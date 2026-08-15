//! Client side: keyword helper, query generation, and response decryption.

use std::io::Result as IoResult;
use std::marker::PhantomData;

use poulpy_pir::client::{Client, QueryState};
use poulpy_pir::keyword::KeywordDirectory;
use poulpy_pir::payload::U512P65536;

use crate::{
    Address, DefaultBackend, EthPirError, EthQuery, EthResponse, RecordCodec, U256Balance,
    default_shape, eth_error_to_io, payload_of,
};

/// Per-lookup client state.
pub struct LookupState {
    addr: Address,
    state: QueryState<DefaultBackend>,
}

/// ETH PIR client. Holds a synced keyword directory and PIR query material.
///
/// Generic over the [`RecordCodec`] the server writes with; the two must agree,
/// since nothing on the wire identifies the layout.
pub struct EthPirClient<C: RecordCodec = U256Balance> {
    directory: KeywordDirectory<20>,
    client: Client<DefaultBackend, U512P65536>,
    /// `fn() -> C` rather than `C`: the codec is a compile-time marker the
    /// client never holds a value of, and this stays `Send + Sync` regardless.
    codec: PhantomData<fn() -> C>,
}

impl<C: RecordCodec> EthPirClient<C> {
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
            codec: PhantomData,
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

    /// The database slot `addr` resolves to through the directory.
    ///
    /// Total, like the MPHF beneath it: an address the server never indexed
    /// still yields a valid slot belonging to some other address. Diagnostics
    /// only — a lookup must still go through [`try_query`](Self::try_query) and
    /// the address check in [`try_decrypt`](Self::try_decrypt).
    pub fn slot(&self, addr: &Address) -> usize {
        self.directory.index(addr)
    }

    /// Serialize a query for transport.
    pub fn encode_query(&self, query: &EthQuery) -> Result<Vec<u8>, EthPirError> {
        let mut bytes = Vec::new();
        query.write_to(self.client.params().module(), &mut bytes)?;
        Ok(bytes)
    }

    /// Build a PIR query for an ETH address, serialized for transport.
    pub fn try_query_bytes(
        &mut self,
        addr: Address,
    ) -> Result<(Vec<u8>, LookupState), EthPirError> {
        let (query, lookup) = self.try_query(addr)?;
        Ok((self.encode_query(&query)?, lookup))
    }

    /// Parse a response received from a server.
    pub fn decode_response(&self, bytes: &[u8]) -> Result<EthResponse, EthPirError> {
        Ok(EthResponse::read_from(
            &mut { bytes },
            self.client.params(),
        )?)
    }

    /// Parse a serialized response and decrypt it in one step.
    pub fn try_decrypt_bytes(
        &mut self,
        bytes: &[u8],
        lookup: &LookupState,
    ) -> Result<C::Value, EthPirError> {
        let response = self.decode_response(bytes)?;
        self.try_decrypt(&response, lookup)
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
    ) -> Result<C::Value, EthPirError> {
        self.try_decrypt(response, lookup)
    }

    /// Fallible response decryptor with record-address verification.
    ///
    /// The address check is the whole reason a record carries its own key: the
    /// keyword index is a *minimal perfect* hash, so it maps an address the
    /// server never indexed onto some arbitrary occupied slot rather than
    /// failing. Comparing the prefix is what turns that into [`NotInSet`]
    /// instead of a confidently wrong value.
    ///
    /// [`NotInSet`]: EthPirError::NotInSet
    pub fn try_decrypt(
        &mut self,
        response: &EthResponse,
        lookup: &LookupState,
    ) -> Result<C::Value, EthPirError> {
        let record = self.client.try_decode(response, &lookup.state)?;
        if record[..20] != lookup.addr {
            return Err(EthPirError::NotInSet);
        }
        Ok(C::decode(payload_of(&record)))
    }
}
