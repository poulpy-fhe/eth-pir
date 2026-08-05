//! Client side: keyword helper, query generation, and response decryption.

use std::io::Result as IoResult;

use poulpy_pir::client::{Client, QueryState};
use poulpy_pir::keyword::KeywordDirectory;
use poulpy_pir::payload::U512P65536;

use crate::{
    Address, Balance, DefaultBackend, EthPirError, EthQuery, EthResponse, address_slot,
    default_shape,
};

/// Per-lookup client state.
pub struct LookupState {
    addr: Address,
    state: QueryState<DefaultBackend>,
}

/// ETH PIR client. Holds a synced keyword directory and PIR query material.
pub struct EthPirClient {
    directory: KeywordDirectory<20>,
    client: Client<DefaultBackend, U512P65536>,
}

impl EthPirClient {
    /// Bootstrap from the server's full directory blob at the fixed 2 GiB shape.
    pub fn new(directory_blob: &[u8]) -> IoResult<Self> {
        let (config, layout) = default_shape();
        Self::with_shape(config, layout, directory_blob)
    }

    /// `new` at a caller-chosen shape, mainly for tests.
    pub fn with_shape(
        config: poulpy_pir::config::Config<U512P65536>,
        layout: poulpy_pir::database::DatabaseLayout<U512P65536>,
        directory_blob: &[u8],
    ) -> IoResult<Self> {
        Ok(Self {
            directory: KeywordDirectory::read_from(&mut { directory_blob })?,
            client: Client::new(config, layout),
        })
    }

    /// The MPHF generation this client holds.
    pub fn version(&self) -> u64 {
        self.directory.version()
    }

    /// How much of the append-only tail this client already holds. Pass it to
    /// [`KeywordWire::tail`](crate::KeywordWire::tail) to fetch only what is
    /// missing.
    pub fn tail_len(&self) -> usize {
        self.directory.delta_len()
    }

    /// Apply a validated tail envelope produced by
    /// [`KeywordWire::tail`](crate::KeywordWire::tail).
    pub fn apply_tail(&mut self, tail: &[u8]) -> IoResult<()> {
        self.directory.apply_delta_envelope(&mut { tail })
    }

    /// Full refresh after bootstrap or server index rebuild.
    pub fn resync(&mut self, directory_blob: &[u8]) -> IoResult<()> {
        self.directory = KeywordDirectory::read_from(&mut { directory_blob })?;
        Ok(())
    }

    /// Build a PIR query for an ETH address.
    pub fn query(&mut self, addr: Address) -> (EthQuery, LookupState) {
        let (query, state) = self.client.query(self.directory.index(&addr));
        (query, LookupState { addr, state })
    }

    /// Decrypt a response and verify the retrieved record belongs to the
    /// queried address.
    pub fn decrypt(
        &mut self,
        response: &EthResponse,
        lookup: &LookupState,
    ) -> Result<Balance, EthPirError> {
        let record = self.client.decode(response, &lookup.state);
        if record[..32] != address_slot(&lookup.addr) {
            return Err(EthPirError::NotInSet);
        }
        Ok(record[32..].try_into().expect("balance half"))
    }
}
