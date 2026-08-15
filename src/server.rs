//! Server side: the PIR database service and keyword-helper wire service.

mod core;
mod keyword;
mod rebuild;
mod records;
mod report;
mod responder;
mod restore;

pub use keyword::{KeywordCheckpoint, KeywordSyncMode, KeywordWire, RestoreReport};
pub use report::{InitTimings, KeywordRebuildTimings, MemoryReport, RefreshTimings};
pub use responder::EthPirResponder;

use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use poulpy_pir::database::Database;
use poulpy_pir::keyword::KeywordDirectory;
use poulpy_pir::server::Server;

use crate::{DefaultBackend, Record, RecordCodec, U256Balance};

type Payload = poulpy_pir::payload::U512P65536;
type PirServer = Server<DefaultBackend, Payload>;
type PirDatabase = Database<DefaultBackend, Payload>;
type Serving = Arc<Mutex<PirServer>>;

/// ETH PIR server: keyword helper, plaintext master records, and one long-lived
/// PIR server refreshed in place.
///
/// The PIR server is built once. Its setup state depends only on the shape, so
/// refreshes rebuild only the encoded database and matching precomputation.
pub struct EthPirServer<C: RecordCodec = U256Balance> {
    directory: KeywordDirectory<20>,
    records: Vec<Record>,
    serving: Serving,
    staging: PirDatabase,
    pending: usize,
    codec: PhantomData<fn() -> C>,
}
