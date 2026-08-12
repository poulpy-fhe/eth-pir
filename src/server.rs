//! Server side: the PIR database service and keyword-helper wire service.

use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use poulpy_pir::database::Database;
use poulpy_pir::keyword::{KeywordDirectory, KeywordIndex};
use poulpy_pir::server::Server;

use crate::{
    Address, DefaultBackend, EthPirError, EthQuery, EthResponse, Record, RecordCodec, U256Balance,
    default_shape, record_of,
};

type Payload = poulpy_pir::payload::U512P65536;
type PirServer = Server<DefaultBackend, Payload>;
type PirDatabase = Database<DefaultBackend, Payload>;
type Serving = Arc<Mutex<PirServer>>;

/// ETH PIR server: keyword helper, plaintext master records, and one long-lived
/// PIR server refreshed in place.
///
/// The PIR server is built **once**. Its setup state — the CRS query masks and
/// the multi-GiB online scratch pool — costs seconds and gigabytes but depends
/// only on the shape, so it is paid at `init` and never again. What a refresh
/// rebuilds is only what actually depends on the data: the encoded database and
/// its matching precomputation, which are computed off to the side and swapped
/// in as a pair.
///
/// Generic over the [`RecordCodec`] that lays out each record's payload. The
/// client must be built with the same one — nothing on the wire identifies it.
pub struct EthPirServer<C: RecordCodec = U256Balance> {
    directory: KeywordDirectory<20>,
    /// Plaintext source of truth. Updates land here immediately; they become
    /// *retrievable* only at the next [`rebuild_database`](Self::rebuild_database),
    /// because that is what re-encodes the served database.
    records: Vec<Record>,
    serving: Serving,
    /// The staging database: filled from `records`, precomputed against, then
    /// swapped with the serving one. After each swap it holds the retired
    /// buffer, so the two just ping-pong and no allocation recurs.
    staging: PirDatabase,
    /// Updates applied to `records` since the last rebuild.
    pending: usize,
    /// `fn() -> C` rather than `C`: the codec is a compile-time marker the
    /// server never holds a value of, and this stays `Send + Sync` regardless.
    codec: PhantomData<fn() -> C>,
}

impl<C: RecordCodec> EthPirServer<C> {
    /// Instantiates DB + keyword helper from the initial address -> value map
    /// at the fixed 2 GiB shape.
    pub fn init(map: &std::collections::HashMap<Address, C::Value>) -> Result<Self, EthPirError> {
        let (config, layout) = default_shape();
        Self::init_with(config, layout, map)
    }

    /// `init` at a caller-chosen shape, mainly for tests.
    pub fn init_with(
        config: poulpy_pir::config::Config<Payload>,
        layout: poulpy_pir::database::DatabaseLayout<Payload>,
        map: &std::collections::HashMap<Address, C::Value>,
    ) -> Result<Self, EthPirError> {
        Self::init_with_timed(config, layout, map).map(|(server, _)| server)
    }

    /// [`init_with`](Self::init_with) with a per-step timing breakdown.
    pub fn init_with_timed(
        config: poulpy_pir::config::Config<Payload>,
        layout: poulpy_pir::database::DatabaseLayout<Payload>,
        map: &std::collections::HashMap<Address, C::Value>,
    ) -> Result<(Self, InitTimings), EthPirError> {
        let mut timings = InitTimings::default();

        let t = Instant::now();
        let keys: Vec<Address> = map.keys().copied().collect();
        let mphf = KeywordIndex::build(&keys)?;
        let capacity = layout.num_payloads(config.column_height());
        let directory = KeywordDirectory::new(mphf, capacity, 0)?;
        timings.keyword_index = t.elapsed();

        let t = Instant::now();
        let mut records = zeroed_records(map.len());
        scatter_records(&mut records, &keys, |_, addr| {
            (directory.index(addr), record_of::<C>(addr, &map[addr]))
        });
        timings.records_scatter = t.elapsed();

        // One-time setup: allocate the server, encode the initial database,
        // materialize the CRS query masks, and run a full `offline()` — the only
        // call that also warms the online scratch pool.
        let t = Instant::now();
        let server = PirServer::try_new(config, layout)?;
        #[cfg(feature = "cblas-gemm")]
        let server = server.with_gemm(poulpy_pir::server::CblasDgemm);
        let mut server = server;
        timings.server_alloc = t.elapsed();

        let t = Instant::now();
        server.try_update_shard(0, &records)?;
        timings.database_encode = t.elapsed();

        let t = Instant::now();
        server.generate_query_mask();
        timings.query_mask = t.elapsed();

        let t = Instant::now();
        server.offline();
        timings.offline = t.elapsed();

        let t = Instant::now();
        let staging = server.new_database();
        timings.staging_alloc = t.elapsed();

        Ok((
            Self {
                directory,
                records,
                serving: Arc::new(Mutex::new(server)),
                staging,
                pending: 0,
                codec: PhantomData,
            },
            timings,
        ))
    }

    /// Apply one address -> value update.
    ///
    /// The value is written straight into the plaintext records. New addresses
    /// are appended to the keyword helper immediately and become visible through
    /// delta sync; either way the value becomes *retrievable* only after the
    /// next [`rebuild_database`](Self::rebuild_database), which is what
    /// re-encodes and re-preprocesses the served database.
    pub fn update(&mut self, addr: Address, value: C::Value) -> Result<(), EthPirError> {
        let i = self.directory.index(&addr);
        if i < self.records.len() && self.records[i][..20] == addr {
            self.records[i] = record_of::<C>(&addr, &value);
        } else {
            let appended = self.directory.push(&addr)?;
            debug_assert_eq!(appended, self.records.len());
            self.records.push(record_of::<C>(&addr, &value));
        }
        self.pending += 1;
        Ok(())
    }

    /// Publish every update applied since the last rebuild.
    ///
    /// Re-encodes the served database from the current records, reruns the
    /// query-independent precomputation against it, and swaps the two in as a
    /// pair. Queries are answered throughout; only the final swap needs
    /// exclusive access, and that is two moves.
    ///
    /// The MPHF is left alone. New addresses already live in the keyword
    /// helper's append-only delta, so a client that synced the delta can query
    /// them as soon as this returns — see
    /// [`rebuild_keyword_index`](Self::rebuild_keyword_index) for compacting
    /// that delta away.
    ///
    /// Returns `None` when there was nothing pending.
    pub fn rebuild_database(&mut self) -> Option<RefreshTimings> {
        self.try_rebuild_database()
            .unwrap_or_else(|err| panic!("{err}"))
    }

    /// Fallible variant of [`rebuild_database`](Self::rebuild_database).
    pub fn try_rebuild_database(&mut self) -> Result<Option<RefreshTimings>, EthPirError> {
        if self.pending == 0 {
            return Ok(None);
        }
        Ok(Some(self.refresh()?))
    }

    /// Compact the append-only delta back into a fresh MPHF, then rebuild the
    /// database to match.
    ///
    /// Derives a new MPHF over the complete key set, permutes the records into
    /// its order, re-encodes the database, reruns the precomputation, and only
    /// then publishes the new directory version — clients query by index, so the
    /// served database has to be laid out the new way before the new version
    /// becomes visible.
    ///
    /// This is the expensive path (dominated by MPHF construction) and it is
    /// optional: run it when the delta has grown enough to be worth compacting,
    /// not on every update. Clients must
    /// [`resync`](crate::EthPirClient::resync) afterwards, since every index
    /// moves.
    pub fn rebuild_keyword_index(&mut self) -> Result<KeywordRebuildTimings, EthPirError> {
        let mut timings = KeywordRebuildTimings::default();

        let t = Instant::now();
        let keys = collect_addresses(&self.records);
        timings.collect_keys = t.elapsed();

        let t = Instant::now();
        let next = self.directory.rebuilt(&keys)?;
        timings.mphf_rebuild = t.elapsed();

        let t = Instant::now();
        let mut records = zeroed_records(self.records.len());
        scatter_records(&mut records, &keys, |i, key| {
            (next.index(key), self.records[i])
        });
        timings.permute = t.elapsed();

        self.directory = next;
        self.records = records;
        timings.refresh = self.refresh()?;
        Ok(timings)
    }

    /// Answer one query against the current database.
    pub fn respond(&self, query: &EthQuery) -> EthResponse {
        self.try_respond(query)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    /// Fallible variant of [`respond`](Self::respond).
    pub fn try_respond(&self, query: &EthQuery) -> Result<EthResponse, EthPirError> {
        respond_with(&self.serving, query)
    }

    /// Answer a batch against the current database.
    pub fn respond_batch(&self, queries: &[EthQuery]) -> Vec<EthResponse> {
        self.try_respond_batch(queries)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    /// Fallible variant of [`respond_batch`](Self::respond_batch).
    pub fn try_respond_batch(&self, queries: &[EthQuery]) -> Result<Vec<EthResponse>, EthPirError> {
        respond_batch_with(&self.serving, queries)
    }

    /// A cloneable database-service handle for serving while `&mut self`
    /// update methods run.
    pub fn responder(&self) -> EthPirResponder {
        EthPirResponder {
            serving: self.serving.clone(),
        }
    }

    /// The keyword-helper wire service.
    pub fn keyword(&self) -> KeywordWire<'_> {
        KeywordWire {
            directory: &self.directory,
        }
    }

    /// Addresses currently addressed: MPHF base plus delta.
    pub fn len(&self) -> usize {
        self.directory.len()
    }

    pub fn is_empty(&self) -> bool {
        self.directory.is_empty()
    }

    /// Updates applied since the last refresh, i.e. written to the records but
    /// not yet retrievable.
    pub fn pending(&self) -> usize {
        self.pending
    }

    /// Re-encode the staging database from `records`, precompute against it, and
    /// swap the pair into the live server.
    ///
    /// Only the final `install` needs the server lock, and it is two moves — so
    /// the multi-second precomputation runs while queries are still answered.
    fn refresh(&mut self) -> Result<RefreshTimings, EthPirError> {
        let mut timings = RefreshTimings::default();

        let t = Instant::now();
        self.staging.try_encode_shard(0, &self.records)?;
        timings.database_encode = t.elapsed();

        let t = Instant::now();
        let mut context = self
            .serving
            .lock()
            .map_err(|_| EthPirError::ServerPoisoned)?
            .precomp_context();
        let (precomputation, _) = context.offline_for(&mut self.staging);
        timings.precompute = t.elapsed();

        // `install` swaps: `staging` comes back holding the retired database,
        // which the next refresh refills. Dropping the retired precomputation is
        // what makes this cost more than two moves.
        let t = Instant::now();
        self.serving
            .lock()
            .map_err(|_| EthPirError::ServerPoisoned)?
            .install(&mut self.staging, precomputation);
        timings.install = t.elapsed();

        self.pending = 0;
        Ok(timings)
    }

    /// Where this server's memory goes.
    pub fn memory_report(&self) -> MemoryReport {
        let server = self
            .serving
            .lock()
            .map_err(|_| EthPirError::ServerPoisoned)
            .unwrap_or_else(|err| panic!("{err}"))
            .memory_report();
        MemoryReport {
            serving_database: server.database,
            staging_database: self.staging.allocated_bytes(),
            precomputation: server.precomputation,
            online_scratch_pool: server.online_scratch_pool,
            records: std::mem::size_of_val(self.records.as_slice()),
            keyword_directory: self.keyword().full().len()
                + self.directory.delta_len() * (std::mem::size_of::<Address>() + 24),
        }
    }
}

/// Cloneable database-service handle.
pub struct EthPirResponder {
    serving: Serving,
}

impl Clone for EthPirResponder {
    fn clone(&self) -> Self {
        Self {
            serving: self.serving.clone(),
        }
    }
}

impl EthPirResponder {
    pub fn respond(&self, query: &EthQuery) -> EthResponse {
        self.try_respond(query)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    pub fn try_respond(&self, query: &EthQuery) -> Result<EthResponse, EthPirError> {
        respond_with(&self.serving, query)
    }

    pub fn respond_batch(&self, queries: &[EthQuery]) -> Vec<EthResponse> {
        self.try_respond_batch(queries)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    pub fn try_respond_batch(&self, queries: &[EthQuery]) -> Result<Vec<EthResponse>, EthPirError> {
        respond_batch_with(&self.serving, queries)
    }
}

fn respond_with(serving: &Serving, query: &EthQuery) -> Result<EthResponse, EthPirError> {
    serving
        .lock()
        .map_err(|_| EthPirError::ServerPoisoned)?
        .try_respond(query)
        .map_err(EthPirError::from)
}

fn respond_batch_with(
    serving: &Serving,
    queries: &[EthQuery],
) -> Result<Vec<EthResponse>, EthPirError> {
    serving
        .lock()
        .map_err(|_| EthPirError::ServerPoisoned)?
        .try_respond_batch(queries)
        .map_err(EthPirError::from)
}

// ---------------------------------------------------------------------------
// Bulk record loops.
//
// These are the only parts of the flow eth-pir computes itself; everything else
// is poulpy-pir's, which parallelizes internally. Left serial they dominate
// `init` and keyword compaction, so they are spread over the same worker budget
// poulpy-pir uses.
// ---------------------------------------------------------------------------

/// Default ceiling on the auto-detected worker budget, matching poulpy-pir's.
///
/// `available_parallelism` counts logical CPUs, i.e. two per physical core on an
/// SMT host, which is the wrong default for this workload — see poulpy-pir's
/// `DEFAULT_MAX_THREADS` for the measurements behind the number. `PIR_THREADS`
/// overrides it and is not capped.
const DEFAULT_MAX_THREADS: usize = 64;

/// Worker count for the bulk loops below.
///
/// Follows poulpy-pir's `PIR_THREADS` convention so a run is sized the same way
/// on both sides of the crate boundary.
fn worker_count(items: usize) -> usize {
    if items <= 1 {
        return 1;
    }
    let base = std::env::var("PIR_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&t| t >= 1)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|x| x.get())
                .unwrap_or(1)
                .min(DEFAULT_MAX_THREADS)
        });
    base.clamp(1, items)
}

/// A `*mut Record` that may cross a thread boundary.
///
/// The type asserts nothing on its own; sending it is sound only under the
/// disjointness contract documented on [`scatter_records`], its sole user.
#[derive(Clone, Copy)]
struct RecordsPtr(*mut Record);

// SAFETY: only ever dereferenced inside `scatter_records`, whose workers write
// disjoint indices of a live, already-initialized allocation that outlives the
// thread scope.
unsafe impl Send for RecordsPtr {}
unsafe impl Sync for RecordsPtr {}

/// `len` zeroed records, zeroed in parallel.
///
/// `vec![[0u8; 64]; len]` memsets a gigabyte on one thread; splitting that
/// across the worker budget is most of the cost. (A parallel *first touch* after
/// the `vec!` does nothing — measured — because `vec!` has already committed the
/// pages.)
fn zeroed_records(len: usize) -> Vec<Record> {
    let mut records: Vec<Record> = Vec::with_capacity(len);
    let workers = worker_count(len);
    let ptr = RecordsPtr(records.as_mut_ptr());
    let per = len.div_ceil(workers);
    std::thread::scope(|scope| {
        for w in 0..workers {
            let start = w * per;
            if start >= len {
                break;
            }
            let count = per.min(len - start);
            scope.spawn(move || {
                let ptr = ptr;
                // SAFETY: `start + count <= len <= capacity`, the ranges are
                // disjoint, writing zero bytes over uninitialized memory is
                // defined, and all-zeros is a valid `Record`.
                unsafe { std::ptr::write_bytes(ptr.0.add(start), 0, count) };
            });
        }
    });
    // SAFETY: the loop above covers `0..len` exactly, so every element is now
    // initialized. If a worker panicked, `thread::scope` unwinds before this and
    // the `Vec` is dropped at length 0.
    unsafe { records.set_len(len) };
    records
}

/// Scatter one record per key into `dst`, in parallel.
///
/// `place(i, key)` returns the destination slot for `keys[i]` and the record to
/// put there.
///
/// The workers write disjoint slots because a *minimal perfect* hash over
/// `keys` is a bijection onto `[0, keys.len())`, and `KeywordIndex::build`
/// rejects duplicate keys — so every slot is written exactly once. `dst` is
/// already initialized, so even if that ever stopped holding, the result would
/// be a wrong record rather than undefined behaviour.
fn scatter_records<F>(dst: &mut [Record], keys: &[Address], place: F)
where
    F: Fn(usize, &Address) -> (usize, Record) + Sync,
{
    let len = dst.len();
    assert_eq!(len, keys.len(), "one record slot per key");
    let workers = worker_count(len);
    if workers <= 1 {
        for (i, key) in keys.iter().enumerate() {
            let (slot, record) = place(i, key);
            dst[slot] = record;
        }
        return;
    }

    let ptr = RecordsPtr(dst.as_mut_ptr());
    let per = len.div_ceil(workers);
    std::thread::scope(|scope| {
        for (w, chunk) in keys.chunks(per).enumerate() {
            let place = &place;
            scope.spawn(move || {
                let ptr = ptr;
                for (k, key) in chunk.iter().enumerate() {
                    let (slot, record) = place(w * per + k, key);
                    assert!(slot < len, "keyword index {slot} past {len} record slots");
                    // SAFETY: `slot < len` is checked just above, the
                    // allocation outlives this scope, and slots are disjoint
                    // across workers — see the function docs.
                    unsafe { *ptr.0.add(slot) = record };
                }
            });
        }
    });
}

/// The address prefix of every record, in record order.
fn collect_addresses(records: &[Record]) -> Vec<Address> {
    let mut keys = vec![[0u8; 20]; records.len()];
    let workers = worker_count(records.len());
    if workers <= 1 {
        for (key, record) in keys.iter_mut().zip(records) {
            key.copy_from_slice(&record[..20]);
        }
        return keys;
    }
    let per = records.len().div_ceil(workers);
    std::thread::scope(|scope| {
        for (dst, src) in keys.chunks_mut(per).zip(records.chunks(per)) {
            scope.spawn(move || {
                for (key, record) in dst.iter_mut().zip(src) {
                    key.copy_from_slice(&record[..20]);
                }
            });
        }
    });
    keys
}

/// The keyword-helper service as its own API surface.
pub struct KeywordWire<'a> {
    directory: &'a KeywordDirectory<20>,
}

impl KeywordWire<'_> {
    /// MPHF generation.
    pub fn version(&self) -> u64 {
        self.directory.version()
    }

    /// Full directory blob for bootstrap and post-rebuild resync.
    pub fn full(&self) -> Vec<u8> {
        self.try_full().unwrap_or_else(|err| panic!("{err}"))
    }

    /// Fallible full directory blob writer.
    pub fn try_full(&self) -> std::io::Result<Vec<u8>> {
        let mut blob = Vec::new();
        self.directory.write_to(&mut blob)?;
        Ok(blob)
    }

    /// MPHF parameters alone.
    pub fn mphf(&self) -> Vec<u8> {
        self.try_mphf().unwrap_or_else(|err| panic!("{err}"))
    }

    /// Fallible MPHF-parameter writer.
    pub fn try_mphf(&self) -> std::io::Result<Vec<u8>> {
        let mut blob = Vec::new();
        self.directory.mphf().write_to(&mut blob)?;
        Ok(blob)
    }

    /// The validated append-only tail from position `have`: the delta keys a
    /// client at `have` has not seen yet, in a versioned envelope.
    pub fn tail(&self, have: usize) -> Vec<u8> {
        self.try_tail(have).unwrap_or_else(|err| panic!("{err}"))
    }

    /// Fallible append-only tail writer.
    pub fn try_tail(&self, have: usize) -> std::io::Result<Vec<u8>> {
        let mut tail = Vec::new();
        self.directory.write_delta_envelope_from(&mut tail, have)?;
        Ok(tail)
    }
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

/// Per-step breakdown of [`EthPirServer::init_with_timed`].
///
/// Everything here is paid once. `query_mask` and the scratch-pool share of
/// `offline` are the setup a refresh deliberately does *not* repeat.
#[derive(Clone, Copy, Debug, Default)]
pub struct InitTimings {
    /// MPHF construction over the initial key set.
    pub keyword_index: Duration,
    /// Allocating the plaintext records and scattering them to MPHF order.
    pub records_scatter: Duration,
    /// Allocating the PIR server and its coefficient database.
    pub server_alloc: Duration,
    /// Encoding the records into the database.
    pub database_encode: Duration,
    /// Materializing the CRS query masks. One-time.
    pub query_mask: Duration,
    /// First precomputation, including warming the online scratch pool. The
    /// pool share of this is one-time.
    pub offline: Duration,
    /// Allocating the staging database.
    pub staging_alloc: Duration,
}

impl InitTimings {
    pub fn total(&self) -> Duration {
        self.keyword_index
            + self.records_scatter
            + self.server_alloc
            + self.database_encode
            + self.query_mask
            + self.offline
            + self.staging_alloc
    }
}

/// Per-step breakdown of a database rebuild — the work every rebuild repeats.
#[derive(Clone, Copy, Debug, Default)]
pub struct RefreshTimings {
    /// Encoding the plaintext records into the staging database.
    pub database_encode: Duration,
    /// The query-independent precomputation, run against the staging database
    /// with no server lock held.
    pub precompute: Duration,
    /// Swapping the pair in, plus freeing the retired precomputation.
    pub install: Duration,
}

impl RefreshTimings {
    pub fn total(&self) -> Duration {
        self.database_encode + self.precompute + self.install
    }
}

/// Per-step breakdown of [`EthPirServer::rebuild_keyword_index`].
#[derive(Clone, Copy, Debug, Default)]
pub struct KeywordRebuildTimings {
    /// Reading the address prefix out of every record.
    pub collect_keys: Duration,
    /// Deriving a fresh MPHF over the complete key set. This dominates.
    pub mphf_rebuild: Duration,
    /// Permuting the records into the new MPHF order.
    pub permute: Duration,
    /// The database rebuild that publishes the new order.
    pub refresh: RefreshTimings,
}

impl KeywordRebuildTimings {
    pub fn total(&self) -> Duration {
        self.collect_keys + self.mphf_rebuild + self.permute + self.refresh.total()
    }
}

/// Where a running server's memory goes, in bytes.
///
/// Covers the allocations that scale. Small fixed state is not counted, so
/// `total()` is a floor on RSS rather than a full accounting — and a refresh
/// transiently adds a second `precomputation` between the precompute and the
/// install, which is what sets peak RSS.
#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryReport {
    /// The coefficient database currently being served.
    pub serving_database: usize,
    /// The staging database, holding the previous generation's buffer.
    pub staging_database: usize,
    /// The precomputation matching `serving_database`.
    pub precomputation: usize,
    /// Per-worker online scratch, sized by `PIR_THREADS`. One-time, and
    /// independent of database size.
    pub online_scratch_pool: usize,
    /// Plaintext records: the source of truth updates are written to.
    pub records: usize,
    /// The keyword directory: MPHF parameters plus the append-only delta.
    /// Approximate — the delta's hash index is estimated.
    pub keyword_directory: usize,
}

impl MemoryReport {
    /// Steady-state total, i.e. between refreshes.
    pub fn total(&self) -> usize {
        self.serving_database
            + self.staging_database
            + self.precomputation
            + self.online_scratch_pool
            + self.records
            + self.keyword_directory
    }

    /// Expected peak during a refresh: steady state plus the second
    /// precomputation that exists between the precompute and the install.
    pub fn refresh_peak(&self) -> usize {
        self.total() + self.precomputation
    }
}
