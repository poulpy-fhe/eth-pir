# eth-pir

ETH token-balance PIR service layer based on `poulpy-pir`.

This crate depends on `poulpy-pir` crate and packages its
index-based PIR core plus keyword directory into a fixed-shape service:

- construction: InsPIRe2 recursion,
- default backend: portable `FFT64Ref` for checks, docs, and downstream builds,
- production backend feature: `avx2-fhe` using `FFT64Avx` (AVX2/FMA),
- optional production backend feature: `avx512-fhe` using `FFT64Avx512`,
- optional GEMM feature: `cblas-gemm` for system CBLAS/OpenBLAS offline
  preprocessing,
- optional NUMA feature: `numa-db-interleave` forwards to `poulpy-pir` database
  allocation,
- preset geometry: `InsPIRe2-g32-2GiB-c65536`,
- payload: one `U512P65536` slot per record,
- capacity: 33,554,432 addresses,
- record: 20-byte address, 12 bytes of zero padding, 32-byte little-endian u256
  token balance.

## Run

Default builds use the portable reference backend so the crate compiles
without CPU-specific flags. That path is for checks and documentation, not the
2 GiB production demo:

```sh
cargo test --release --lib
```

Production AVX2/FMA runs use the explicit backend feature and target flags:

```sh
RUSTFLAGS="-C target-feature=+avx2,+fma" \
cargo run --release --features avx2-fhe --example eth_pir
```

On an AVX-512F host:

```sh
RUSTFLAGS="-C target-feature=+avx2,+fma,+avx512f" \
cargo run --release --no-default-features --features avx512-fhe --example eth_pir
```

With system CBLAS/OpenBLAS enabled:

```sh
RUSTFLAGS="-C target-feature=+avx2,+fma,+avx512f" \
cargo run --release --no-default-features --features "avx512-fhe,cblas-gemm" \
  --example eth_pir
```

For single-query latency on multi-socket Linux hosts, add NUMA interleaving:

```sh
RUSTFLAGS="-C target-feature=+avx2,+fma,+avx512f" \
cargo run --release --no-default-features \
  --features "avx512-fhe,cblas-gemm,numa-db-interleave" \
  --example eth_pir
```

### Worker count

The worker budget is the auto-detected CPU count capped at **64**, one per
physical core; see [Performance](#performance) for why SMT is not worth it here.
`PIR_THREADS` overrides it in either direction and is not capped:

```sh
PIR_THREADS=32 RUSTFLAGS="-C target-feature=+avx2,+fma" cargo run --release --features avx2-fhe --example eth_pir
```

It sizes both `poulpy-pir`'s internal parallelism and `eth-pir`'s own bulk record
loops, so one variable scales the whole run. It also sizes the online scratch
pool, which is the largest single allocation the server makes.

Install the optimized pthread OpenBLAS development package before enabling
`cblas-gemm`:

```sh
sudo apt-get update
sudo apt-get install -y libopenblas-pthread-dev
```

On Debian/Ubuntu this provides `/usr/lib/x86_64-linux-gnu/openblas-pthread` and
`libopenblas.so`. Without it, `cblas-gemm` fails during setup because the linker
cannot find `-lopenblas`. The example re-execs once with
`OPENBLAS_NUM_THREADS=1` when the variable is absent, because OpenBLAS sizes its
worker pool before `main`.

`poulpy-pir` exposes `CBLAS_LIB_DIR` and `CBLAS_LIB_NAME` for deliberate custom
BLAS deployments, but this example expects an optimized `cblas_dgemm`; generic
CBLAS providers such as GSL CBLAS are not the intended performance path.

Leave `numa-db-interleave` off for batched serving; `poulpy-pir`'s measured
tuning currently treats interleaving as the single-query-latency choice.

## Performance

One AWS **c8i.32xlarge** (Intel Xeon 6975P-C, 64 cores, 247 GiB), serving
**16 M ETH addresses** from a 2 GiB database with capacity for 33.5 M:

| | |
| --- | --- |
| **private lookup** | **114 ms** |
| **throughput, batched** | **40 queries/s** (batch 128) |
| client download, one-time | 4.05 MiB — 2.116 bits/address (16mio addresses) |
| query / response on the wire | 675 KiB / 192 KiB |
| absorb 1.05 M balance updates | 0.30 s |
| publish them, zero downtime | 4.4 s (5.2 s while serving) |
| server cold start | 17 s |
| memory | 16 GiB steady, 25 GiB peak (30 GiB with batching) |

Nothing is precomputed per client and nothing is cached per query: every lookup
is a full private read of the whole database, and the server never learns which
address was asked for.

**Updates are cheap and never take the service offline.** Balance updates land in
plaintext immediately; publishing them re-encodes the database and reruns the
query-independent precomputation off to the side, then swaps the two in as a
pair under a lock held for microseconds. The example keeps a query load running
across the whole rebuild and asserts every response decodes to either the old or
the new balance — a database swapped in without its matching precomputation
fails the run.

**Batch size is a throughput/latency dial.** Every query in a batch waits for the
whole batch, so 8.7 q/s unbatched becomes 33.7 q/s at batch 64 (1.9 s latency)
and 40 q/s at batch 128 (3.2 s), for about +5 GiB of peak memory. Pick it from
the latency budget.

**Compaction is the one slow path**, and it is optional: `rebuild_keyword_index`
takes 11.4 s, of which 6.8 s is MPHF construction and 4.4 s the database rebuild
it ends with. Until you run it, new addresses stay queryable through a
20 B/address delta, which is cheaper than refetching the 4.05 MiB MPHF for
~212 K inserts.

The worker budget is the detected CPU count capped at 64 — one per physical
core. SMT gains 6% on the offline precompute but costs 10% online latency and
10.6 GiB of peak RSS, so it is off by default; `PIR_THREADS` overrides it.

The example prints the per-phase breakdown behind most of these numbers — init
and rebuild timings, wire sizes, and the memory accounting against measured
VmHWM:

```sh
RUSTFLAGS="-C target-feature=+avx512f,+avx512dq" \
cargo run --release --no-default-features \
  --features "avx512-fhe,cblas-gemm,numa-db-interleave" \
  --example eth_pir
```

It measures batch 32 and 64; the batch-8/128/256 figures and the 64-vs-128
worker comparison come from a wider sweep that is not part of the repository.

## Sync Contract

Cold start and post-rebuild resync use `KeywordWire::full()` and
`EthPirClient::{new,resync}`.

### Keyword Directory Size

The keyword directory lets clients map an ETH address to the PIR record index
without downloading the address set. Its base structure is a minimal perfect
hash function (MPHF): the server builds it once over the address set, ships only
the MPHF parameters, and both sides then compute the same `address -> index`
mapping locally.

The current MPHF parameters cost about **2.116 bits/address** on the wire. In
the 16 M-address example run, client bootstrap downloaded:

```text
4,231,252 bytes = 4.04 MiB = 2.116 bits/address
```

A naive downloadable index would be much larger because the client would need
the addresses, not just integer slots:

```text
16,000,000 addresses

MPHF parameters                 4,231,252 B    4.04 MiB
address list, index by position 320,000,000 B  305.18 MiB
address -> u32 index table      384,000,000 B  366.21 MiB
```

So for 16 M addresses, the MPHF is roughly **76x smaller** than downloading the
address list in slot order, and roughly **91x smaller** than downloading
`address -> u32 index` pairs.

At the full 2 GiB shape capacity of 33,554,432 addresses, the same MPHF density
would be about **8.46 MiB**, while a naive address list would be **640 MiB** and
an `address -> u32 index` table would be **768 MiB**.

Only addresses added after the last MPHF rebuild are sent as a delta overlay.
Each delta key costs 20 bytes until the next index rebuild compacts it back into
the MPHF.

Incremental sync uses `KeywordWire::tail(client.tail_len())` and
`EthPirClient::apply_tail`. The tail is a validated envelope:

```text
magic             8 bytes: "PIRDLT1\0"
directory_version u64 little-endian
base_mphf_len     u64 little-endian
delta_start       u64 little-endian
delta_count       u64 little-endian
capacity          u64 little-endian
keys              delta_count 20-byte addresses
```

The client rejects stale versions, wrong offsets, mismatched base lengths,
capacity mismatches, over-capacity payloads, invalid magic, and duplicate keys
within the delta overlay.

Between MPHF rebuilds, newly inserted addresses live in this append-only delta
overlay. A client can download only the delta tail, learn the new
address-to-index entries, and query those addresses after the server rebuilds the
matching records with `EthPirServer::rebuild_database()`. The MPHF rebuild is a
later compaction step, not a prerequisite for querying recently appended
addresses.

## Rebuilds

Updates are applied to the plaintext records immediately and become retrievable
at the next rebuild. There are two, and they differ only in whether the MPHF is
touched:

- `rebuild_database()` re-encodes the served database from the current records
  and reruns the precomputation. It keeps the current MPHF and its append-only
  delta, so newly inserted addresses become queryable without a client resync.
  This is the common path: ~4.4 s, zero downtime.
- `rebuild_keyword_index()` derives a fresh MPHF over the complete key set,
  permutes the records into its order, then rebuilds the database so the served
  layout matches, and only then publishes the new directory version. Every index
  moves, so clients must `resync()` afterwards. Run it when the delta has grown
  enough to be worth compacting — ~11.4 s, dominated by MPHF construction.

Both return their per-step timings. `rebuild_database` returns `None` when
nothing was pending.

## API

```rust
// ---- Shared types -------------------------------------------------------

pub type Address = [u8; 20];                 // an ETH address: the keyword
pub type Balance = [u8; 32];                 // a token balance: little-endian u256
pub type DefaultBackend = ...;               // FFT64Avx, or FFT64Avx512 with `avx512-fhe`
pub type EthQuery = ...;                     // an encrypted lookup, opaque to the server
pub type EthResponse = ...;                  // the encrypted answer

/// The fixed deployment shape: `InsPIRe2-g32-2GiB-c65536`, 33,554,432 addresses.
pub fn default_shape() -> (Config<U512P65536>, DatabaseLayout<U512P65536>);

pub enum EthPirError {
    NotInSet,                    // the record returned does not carry the queried address
    ServerPoisoned,
    Keyword(KeywordError<20>),
    Pir(poulpy_pir::PirError),
    Io { kind: io::ErrorKind, message: String },
}

// ---- Server -------------------------------------------------------------

impl EthPirServer {
    /// Build the service from an address -> balance map. Derives the MPHF, places
    /// records, encodes the database, materializes the CRS masks and runs the
    /// first precomputation. Everything here is paid once.
    pub fn init(map: &HashMap<Address, Balance>) -> Result<Self, EthPirError>;
    pub fn init_with(config, layout, map) -> Result<Self, EthPirError>;
    pub fn init_with_timed(config, layout, map) -> Result<(Self, InitTimings), EthPirError>;

    /// Apply one balance update. Written to the plaintext records immediately;
    /// retrievable only after the next `rebuild_database`. New addresses are appended
    /// to the keyword delta and are visible to clients that sync it.
    pub fn update(&mut self, addr: Address, value: Balance) -> Result<(), EthPirError>;

    /// Publish everything applied since the last rebuild: re-encode the database,
    /// rerun the precomputation off to the side, swap the pair in. Keeps the
    /// current MPHF, so no client resync is needed. Queries are served
    /// throughout. `None` if nothing was pending.
    pub fn rebuild_database(&mut self) -> Option<RefreshTimings>;
    pub fn try_rebuild_database(&mut self) -> Result<Option<RefreshTimings>, EthPirError>;

    /// Compact the append-only delta into a fresh MPHF, permute the records, and
    /// rebuild the database to match, publishing the new directory version last.
    /// Every index moves, so clients must `resync()` afterwards.
    pub fn rebuild_keyword_index(&mut self) -> Result<KeywordRebuildTimings, EthPirError>;

    /// Answer queries. The server never learns which address was asked for.
    /// Batching amortizes the database pass; see Performance for the tradeoff.
    pub fn respond(&self, query: &EthQuery) -> EthResponse;
    pub fn try_respond(&self, query: &EthQuery) -> Result<EthResponse, EthPirError>;
    pub fn respond_batch(&self, queries: &[EthQuery]) -> Vec<EthResponse>;
    pub fn try_respond_batch(&self, queries: &[EthQuery]) -> Result<Vec<EthResponse>, EthPirError>;

    /// A cloneable, `Send` handle that serves while `&mut self` methods run.
    pub fn responder(&self) -> EthPirResponder;

    /// The keyword-helper wire service.
    pub fn keyword(&self) -> KeywordWire<'_>;

    pub fn len(&self) -> usize;             // addresses addressed: MPHF base + delta
    pub fn is_empty(&self) -> bool;
    pub fn pending(&self) -> usize;         // updates applied but not yet published
    pub fn memory_report(&self) -> MemoryReport;
}

impl EthPirResponder {                      // + Clone
    pub fn respond(&self, query: &EthQuery) -> EthResponse;
    pub fn try_respond(&self, query: &EthQuery) -> Result<EthResponse, EthPirError>;
    pub fn respond_batch(&self, queries: &[EthQuery]) -> Vec<EthResponse>;
    pub fn try_respond_batch(&self, queries: &[EthQuery]) -> Result<Vec<EthResponse>, EthPirError>;
}

// ---- Keyword helper (wire) ----------------------------------------------

impl KeywordWire<'_> {
    pub fn version(&self) -> u64;                    // MPHF generation
    pub fn full(&self) -> Vec<u8>;                   // bootstrap / post-rebuild resync blob
    pub fn try_full(&self) -> io::Result<Vec<u8>>;
    pub fn mphf(&self) -> Vec<u8>;                   // MPHF parameters alone
    pub fn try_mphf(&self) -> io::Result<Vec<u8>>;
    pub fn tail(&self, have: usize) -> Vec<u8>;      // validated append-only tail
    pub fn try_tail(&self, have: usize) -> io::Result<Vec<u8>>;
}

// ---- Client -------------------------------------------------------------

impl EthPirClient {
    /// Bootstrap from the server's full directory blob.
    pub fn new(directory_blob: &[u8]) -> io::Result<Self>;
    pub fn try_new(directory_blob: &[u8]) -> Result<Self, EthPirError>;
    pub fn with_shape(config, layout, directory_blob) -> io::Result<Self>;
    pub fn try_with_shape(config, layout, directory_blob) -> Result<Self, EthPirError>;

    pub fn version(&self) -> u64;
    pub fn tail_len(&self) -> usize;                         // pass to `KeywordWire::tail`
    pub fn apply_tail(&mut self, tail: &[u8]) -> io::Result<()>;
    pub fn try_apply_tail(&mut self, tail: &[u8]) -> Result<(), EthPirError>;
    pub fn resync(&mut self, directory_blob: &[u8]) -> io::Result<()>;
    pub fn try_resync(&mut self, directory_blob: &[u8]) -> Result<(), EthPirError>;

    /// Build a private query for an address, and decrypt the answer. `decrypt`
    /// checks the returned record actually carries the queried address, so a
    /// stale mapping surfaces as `NotInSet` rather than a wrong balance.
    pub fn query(&mut self, addr: Address) -> (EthQuery, LookupState);
    pub fn try_query(&mut self, addr: Address) -> Result<(EthQuery, LookupState), EthPirError>;
    pub fn decrypt(&mut self, response: &EthResponse, lookup: &LookupState)
        -> Result<Balance, EthPirError>;
    pub fn try_decrypt(&mut self, response: &EthResponse, lookup: &LookupState)
        -> Result<Balance, EthPirError>;
}

// ---- Reporting ----------------------------------------------------------

pub struct InitTimings {    // keyword_index, records_scatter, server_alloc,
    ...                     // database_encode, query_mask, offline, staging_alloc
}
pub struct RefreshTimings { // database_encode, precompute, install
    ...
}
pub struct KeywordRebuildTimings {  // collect_keys, mphf_rebuild, permute,
    ...                             // refresh: RefreshTimings
}
pub struct MemoryReport {   // serving_database, staging_database, precomputation,
    ...                     // online_scratch_pool, records, keyword_directory
}
// each has `total()`; `MemoryReport` also has `refresh_peak()`
```

## Repository Layout

- `src/lib.rs`: shared types, default shape, errors.
- `src/server.rs`: `EthPirServer`, `EthPirResponder`, `KeywordWire`.
- `src/client.rs`: `EthPirClient`, lookup state, decrypt verification.
- `examples/eth_pir.rs`: 16 M initial-address demo on the 2 GiB shape, with 1 M
  balance updates, 50 K inserted addresses, batched lookups, a database rebuild
  under concurrent query load, append-only delta lookup, and a keyword-index
  rebuild.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
