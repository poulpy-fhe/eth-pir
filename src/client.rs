//! Client side: keyword helper, query generation, and response decryption.

use std::io::Result as IoResult;

use poulpy_pir::client::{Client, QueryState};
use poulpy_pir::keyword::KeywordDirectory;
use poulpy_pir::payload::U512P65536;

use crate::{
    Address, Balance, DefaultBackend, EthPirError, EthQuery, EthResponse, address_slot,
    default_shape, eth_error_to_io,
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
        Self::try_new(directory_blob).map_err(eth_error_to_io)
    }

    /// Fallible bootstrap from the server's full directory blob.
    pub fn try_new(directory_blob: &[u8]) -> Result<Self, EthPirError> {
        let (config, layout) = default_shape();
        Self::try_with_shape(config, layout, directory_blob)
    }

    /// `new` at a caller-chosen shape, mainly for tests.
    pub fn with_shape(
        config: poulpy_pir::config::Config<U512P65536>,
        layout: poulpy_pir::database::DatabaseLayout<U512P65536>,
        directory_blob: &[u8],
    ) -> IoResult<Self> {
        Self::try_with_shape(config, layout, directory_blob).map_err(eth_error_to_io)
    }

    /// Fallible `new` at a caller-chosen shape.
    pub fn try_with_shape(
        config: poulpy_pir::config::Config<U512P65536>,
        layout: poulpy_pir::database::DatabaseLayout<U512P65536>,
        directory_blob: &[u8],
    ) -> Result<Self, EthPirError> {
        Ok(Self {
            directory: KeywordDirectory::read_from(&mut { directory_blob })?,
            client: Client::try_new(config, layout)?,
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
        self.try_apply_tail(tail).map_err(eth_error_to_io)
    }

    /// Fallible tail application with the crate's unified error type.
    pub fn try_apply_tail(&mut self, tail: &[u8]) -> Result<(), EthPirError> {
        self.directory.apply_delta_envelope(&mut { tail })?;
        Ok(())
    }

    /// Full refresh after bootstrap or server index rebuild.
    pub fn resync(&mut self, directory_blob: &[u8]) -> IoResult<()> {
        self.try_resync(directory_blob).map_err(eth_error_to_io)
    }

    /// Fallible full refresh after bootstrap or server index rebuild.
    pub fn try_resync(&mut self, directory_blob: &[u8]) -> Result<(), EthPirError> {
        self.directory = KeywordDirectory::read_from(&mut { directory_blob })?;
        Ok(())
    }

    /// Build a PIR query for an ETH address.
    pub fn query(&mut self, addr: Address) -> (EthQuery, LookupState) {
        self.try_query(addr).unwrap_or_else(|err| panic!("{err}"))
    }

    /// Fallible PIR query builder for an ETH address.
    pub fn try_query(&mut self, addr: Address) -> Result<(EthQuery, LookupState), EthPirError> {
        let (query, state) = self.client.try_query(self.directory.index(&addr))?;
        Ok((query, LookupState { addr, state }))
    }

    /// Decrypt a response and verify the retrieved record belongs to the
    /// queried address.
    pub fn decrypt(
        &mut self,
        response: &EthResponse,
        lookup: &LookupState,
    ) -> Result<Balance, EthPirError> {
        self.try_decrypt(response, lookup)
    }

    /// Fallible response decryptor with record-address verification.
    pub fn try_decrypt(
        &mut self,
        response: &EthResponse,
        lookup: &LookupState,
    ) -> Result<Balance, EthPirError> {
        let record = self.client.try_decode(response, &lookup.state)?;
        if record[..32] != address_slot(&lookup.addr) {
            return Err(EthPirError::NotInSet);
        }
        let mut balance = [0u8; 32];
        balance.copy_from_slice(&record[32..]);
        Ok(balance)
    }
}
