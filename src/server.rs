//! Server side: the PIR database service and keyword-helper wire service.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use poulpy_pir::keyword::{KeywordDirectory, KeywordIndex};
use poulpy_pir::server::Server;

use crate::{
    Address, Balance, DefaultBackend, EthPirError, EthQuery, EthResponse, Record, default_shape,
    record_of,
};

type Snapshot = Arc<Mutex<Server<DefaultBackend, poulpy_pir::payload::U512P65536>>>;
type Slot = Arc<RwLock<Snapshot>>;

/// ETH PIR server: keyword helper, plaintext master records, queued updates,
/// and the current serving snapshot.
pub struct EthPirServer {
    config: poulpy_pir::config::Config<poulpy_pir::payload::U512P65536>,
    layout: poulpy_pir::database::DatabaseLayout<poulpy_pir::payload::U512P65536>,
    directory: KeywordDirectory<20>,
    records: Vec<Record>,
    queue: HashMap<Address, Balance>,
    serving: Slot,
    epoch: u64,
}

impl EthPirServer {
    /// Instantiates DB + keyword helper from the initial address -> balance map
    /// at the fixed 2 GiB shape.
    pub fn init(map: &HashMap<Address, Balance>) -> Result<Self, EthPirError> {
        let (config, layout) = default_shape();
        Self::init_with(config, layout, map)
    }

    /// `init` at a caller-chosen shape, mainly for tests.
    pub fn init_with(
        config: poulpy_pir::config::Config<poulpy_pir::payload::U512P65536>,
        layout: poulpy_pir::database::DatabaseLayout<poulpy_pir::payload::U512P65536>,
        map: &HashMap<Address, Balance>,
    ) -> Result<Self, EthPirError> {
        let keys: Vec<Address> = map.keys().copied().collect();
        let mphf = KeywordIndex::build(&keys)?;
        let capacity = layout.num_payloads(config.column_height());
        let directory = KeywordDirectory::new(mphf, capacity, 0)?;
        let mut records = vec![[0u8; 64]; map.len()];
        for (addr, value) in map {
            records[directory.index(addr)] = record_of(addr, value);
        }
        let server = Self::build_snapshot(config, layout, &records);
        Ok(Self {
            config,
            layout,
            directory,
            records,
            queue: HashMap::new(),
            serving: Arc::new(RwLock::new(Arc::new(Mutex::new(server)))),
            epoch: 0,
        })
    }

    /// Queue one address -> balance update.
    ///
    /// New addresses are appended to the keyword helper immediately and become
    /// visible through delta sync. Their balances become retrievable after the
    /// next `flush_queue`.
    pub fn queue_new_update(&mut self, addr: Address, value: Balance) -> Result<(), EthPirError> {
        if !self.queue.contains_key(&addr) {
            let known = self.records[self.directory.index(&addr)][..20] == addr;
            if !known {
                let i = self.directory.push(&addr)?;
                debug_assert_eq!(i, self.records.len());
                self.records.push(record_of(&addr, &[0u8; 32]));
            }
        }
        self.queue.insert(addr, value);
        self.epoch = self.epoch.wrapping_add(1);
        Ok(())
    }

    /// Applies queued updates and swaps in a freshly built serving snapshot.
    pub fn flush_queue(&mut self) {
        if self.apply_queue() {
            self.install_snapshot();
            self.epoch = self.epoch.wrapping_add(1);
        }
    }

    /// Builds a fresh MPHF and permutes records for it, without publishing.
    ///
    /// This is the cheap half of index compaction. It does not change
    /// `keyword()`, the serving snapshot, or the pending queue, so the live
    /// server remains consistent while the work is being measured or staged.
    /// Call [`publish_index_rebuild`](Self::publish_index_rebuild) to rebuild
    /// the PIR snapshot and publish the new directory version.
    pub fn prepare_index_rebuild(&self) -> Result<PreparedIndexRebuild, EthPirError> {
        let keys: Vec<Address> = self
            .records
            .iter()
            .map(|r| r[..20].try_into().expect("record address"))
            .collect();
        let next = self.directory.rebuilt(&keys)?;
        let mut records = vec![[0u8; 64]; self.records.len()];
        for (record, key) in self.records.iter().zip(&keys) {
            records[next.index(key)] = match self.queue.get(key) {
                Some(value) => record_of(key, value),
                None => *record,
            };
        }
        Ok(PreparedIndexRebuild {
            epoch: self.epoch,
            directory: next,
            records,
        })
    }

    /// Publishes a prepared index rebuild and installs its matching snapshot.
    ///
    /// Publishing still has to re-encode the PIR database and rerun offline
    /// preprocessing: a re-derived MPHF changes the indices clients query, so
    /// the serving database must be laid out the same way before the new
    /// directory version becomes visible.
    pub fn publish_index_rebuild(
        &mut self,
        prepared: PreparedIndexRebuild,
    ) -> Result<(), EthPirError> {
        if prepared.epoch != self.epoch {
            return Err(EthPirError::StalePreparedRebuild);
        }
        self.directory = prepared.directory;
        self.records = prepared.records;
        self.queue.clear();
        self.install_snapshot();
        self.epoch = self.epoch.wrapping_add(1);
        Ok(())
    }

    /// Rebuilds the MPHF, permutes records, and publishes the matching snapshot.
    ///
    /// This is the production-safe one-shot API. Use
    /// [`prepare_index_rebuild`](Self::prepare_index_rebuild) plus
    /// [`publish_index_rebuild`](Self::publish_index_rebuild) when measuring the
    /// MPHF rebuild separately from the required database publication work.
    pub fn rebuild_index(&mut self) -> Result<(), EthPirError> {
        let prepared = self.prepare_index_rebuild()?;
        self.publish_index_rebuild(prepared)?;
        Ok(())
    }

    /// Answer one query against the current snapshot.
    pub fn respond(&self, query: &EthQuery) -> EthResponse {
        respond_with_slot(&self.serving, query)
    }

    /// Answer a batch against the current snapshot.
    pub fn respond_batch(&self, queries: &[EthQuery]) -> Vec<EthResponse> {
        respond_batch_with_slot(&self.serving, queries)
    }

    /// A cloneable database-service handle for serving while `&mut self`
    /// update methods run.
    pub fn responder(&self) -> EthPirResponder {
        EthPirResponder {
            slot: self.serving.clone(),
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

    /// Updates waiting for the next flush.
    pub fn pending(&self) -> usize {
        self.queue.len()
    }

    fn apply_queue(&mut self) -> bool {
        if self.queue.is_empty() {
            return false;
        }
        let queue = std::mem::take(&mut self.queue);
        for (addr, value) in queue {
            let i = self.directory.index(&addr);
            debug_assert_eq!(self.records[i][..20], addr, "queue/records misaligned");
            self.records[i] = record_of(&addr, &value);
        }
        true
    }

    fn install_snapshot(&mut self) {
        let server = Self::build_snapshot(self.config, self.layout, &self.records);
        *self.serving.write().unwrap() = Arc::new(Mutex::new(server));
    }

    fn build_snapshot(
        config: poulpy_pir::config::Config<poulpy_pir::payload::U512P65536>,
        layout: poulpy_pir::database::DatabaseLayout<poulpy_pir::payload::U512P65536>,
        records: &[Record],
    ) -> Server<DefaultBackend, poulpy_pir::payload::U512P65536> {
        let server = Server::new(config, layout);
        #[cfg(feature = "cblas-gemm")]
        let server = server.with_gemm(poulpy_pir::server::CblasDgemm);
        let mut server = server;
        server.update_shard(0, records);
        server.generate_query_mask();
        server.offline();
        server
    }
}

/// A staged MPHF compaction that has not been published yet.
///
/// While this value exists, the live server still exposes the old keyword
/// directory and old serving snapshot. It must be passed back to
/// [`EthPirServer::publish_index_rebuild`] before clients should resync to the
/// rebuilt directory version.
pub struct PreparedIndexRebuild {
    epoch: u64,
    directory: KeywordDirectory<20>,
    records: Vec<Record>,
}

/// Cloneable database-service handle.
pub struct EthPirResponder {
    slot: Slot,
}

impl Clone for EthPirResponder {
    fn clone(&self) -> Self {
        Self {
            slot: self.slot.clone(),
        }
    }
}

impl EthPirResponder {
    pub fn respond(&self, query: &EthQuery) -> EthResponse {
        respond_with_slot(&self.slot, query)
    }

    pub fn respond_batch(&self, queries: &[EthQuery]) -> Vec<EthResponse> {
        respond_batch_with_slot(&self.slot, queries)
    }
}

fn respond_with_slot(slot: &Slot, query: &EthQuery) -> EthResponse {
    let snapshot = slot.read().unwrap().clone();
    let mut server = snapshot.lock().unwrap();
    server.respond(query)
}

fn respond_batch_with_slot(slot: &Slot, queries: &[EthQuery]) -> Vec<EthResponse> {
    let snapshot = slot.read().unwrap().clone();
    let mut server = snapshot.lock().unwrap();
    server.respond_batch(queries)
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
        let mut blob = Vec::new();
        self.directory.write_to(&mut blob).expect("write to Vec");
        blob
    }

    /// MPHF parameters alone.
    pub fn mphf(&self) -> Vec<u8> {
        let mut blob = Vec::new();
        self.directory
            .mphf()
            .write_to(&mut blob)
            .expect("write to Vec");
        blob
    }

    /// Versioned append-only delta envelope from position `have`.
    pub fn delta_from(&self, have: usize) -> Vec<u8> {
        let mut tail = Vec::new();
        self.directory
            .write_delta_envelope_from(&mut tail, have)
            .expect("write to Vec");
        tail
    }
}
