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
    /// next [`flush_records`](Self::flush_records).
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

    /// Rebuilds the PIR database from the current keyword helper plus pending
    /// record updates.
    ///
    /// This does **not** rebuild the MPHF. New addresses already live in the
    /// keyword helper's append-only delta, so clients that applied the delta can
    /// query them after this flush. Use
    /// [`flush_keyword_helper`](Self::flush_keyword_helper) only when the delta
    /// should be compacted back into a fresh MPHF.
    pub fn flush_records(&mut self) {
        if self.apply_queue() {
            self.install_snapshot();
            self.epoch = self.epoch.wrapping_add(1);
        }
    }

    /// Backward-compatible name for [`flush_records`](Self::flush_records).
    pub fn flush_queue(&mut self) {
        self.flush_records();
    }

    /// Builds a fresh MPHF and permutes records for it, without publishing.
    ///
    /// This is the cheap half of keyword-helper compaction. It does not change
    /// `keyword()`, the serving snapshot, or the pending queue, so the live
    /// server remains consistent while the work is being measured or staged.
    /// Call [`publish_keyword_helper_flush`](Self::publish_keyword_helper_flush)
    /// to rebuild the PIR snapshot and publish the new directory version.
    pub fn prepare_keyword_helper_flush(&self) -> Result<PreparedKeywordHelperFlush, EthPirError> {
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
        Ok(PreparedKeywordHelperFlush {
            epoch: self.epoch,
            directory: next,
            records,
        })
    }

    /// Backward-compatible name for
    /// [`prepare_keyword_helper_flush`](Self::prepare_keyword_helper_flush).
    pub fn prepare_index_rebuild(&self) -> Result<PreparedKeywordHelperFlush, EthPirError> {
        self.prepare_keyword_helper_flush()
    }

    /// Publishes a prepared keyword-helper flush and installs its matching
    /// snapshot.
    ///
    /// Publishing still has to re-encode the PIR database and rerun offline
    /// preprocessing: a re-derived MPHF changes the indices clients query, so
    /// the serving database must be laid out the same way before the new
    /// directory version becomes visible.
    pub fn publish_keyword_helper_flush(
        &mut self,
        prepared: PreparedKeywordHelperFlush,
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

    /// Backward-compatible name for
    /// [`publish_keyword_helper_flush`](Self::publish_keyword_helper_flush).
    pub fn publish_index_rebuild(
        &mut self,
        prepared: PreparedKeywordHelperFlush,
    ) -> Result<(), EthPirError> {
        self.publish_keyword_helper_flush(prepared)
    }

    /// Rebuilds the keyword helper's MPHF and publishes the matching PIR
    /// database snapshot.
    ///
    /// This is the production-safe compaction API: it rebuilds the MPHF over the
    /// complete current key set, permutes records into the new MPHF order,
    /// rebuilds the PIR database, reruns offline preprocessing, and only then
    /// publishes the new directory version. Use
    /// [`flush_records`](Self::flush_records) for the common append-only path
    /// that keeps the current MPHF plus delta overlay.
    pub fn flush_keyword_helper(&mut self) -> Result<(), EthPirError> {
        let prepared = self.prepare_keyword_helper_flush()?;
        self.publish_keyword_helper_flush(prepared)?;
        Ok(())
    }

    /// Backward-compatible name for
    /// [`flush_keyword_helper`](Self::flush_keyword_helper).
    pub fn rebuild_index(&mut self) -> Result<(), EthPirError> {
        self.flush_keyword_helper()
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

/// A staged keyword-helper compaction that has not been published yet.
///
/// While this value exists, the live server still exposes the old keyword
/// directory and old serving snapshot. It must be passed back to
/// [`EthPirServer::publish_keyword_helper_flush`] before clients should resync
/// to the rebuilt directory version.
pub struct PreparedKeywordHelperFlush {
    epoch: u64,
    directory: KeywordDirectory<20>,
    records: Vec<Record>,
}

/// Backward-compatible name for [`PreparedKeywordHelperFlush`].
pub type PreparedIndexRebuild = PreparedKeywordHelperFlush;

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
