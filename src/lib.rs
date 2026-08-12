//! ETH token-balance PIR service layer over `poulpy-pir`.
//!
//! This crate packages the index-based PIR core and keyword directory into a
//! fixed-shape ETH token-balance lookup service. Default builds use the portable
//! reference backend so CI, docs.rs, and downstream consumers can compile
//! without CPU-specific flags. Production deployments should opt into
//! `avx2-fhe` or `avx512-fhe` and pass matching `RUSTFLAGS`.

mod client;
mod server;

pub use client::{EthPirClient, LookupState};

pub use server::{
    EthPirResponder, EthPirServer, InitTimings, KeywordRebuildTimings, KeywordWire, MemoryReport,
    RefreshTimings,
};

use poulpy_pir::config::{Collapse, Config};
use poulpy_pir::database::DatabaseLayout;
use poulpy_pir::keyword::KeywordError;
use poulpy_pir::payload::U512P65536;

#[cfg(all(feature = "avx2-fhe", feature = "avx512-fhe"))]
compile_error!("features `avx2-fhe` and `avx512-fhe` are mutually exclusive");

#[cfg(feature = "avx512-fhe")]
pub type DefaultBackend = poulpy_cpu_avx512::FFT64Avx512;
#[cfg(all(not(feature = "avx512-fhe"), feature = "avx2-fhe"))]
pub type DefaultBackend = poulpy_cpu_avx::FFT64Avx;
#[cfg(not(any(feature = "avx2-fhe", feature = "avx512-fhe")))]
pub type DefaultBackend = poulpy_cpu_ref::FFT64Ref;

pub type EthQuery = poulpy_pir::server::Query<DefaultBackend>;
pub type EthResponse = poulpy_pir::client::Response<DefaultBackend>;

/// The keyword: a 20-byte ETH address.
pub type Address = [u8; 20];
/// A token balance: little-endian `u256` (byte 0 is least significant).
pub type Balance = [u8; 32];
/// One 64-byte payload: `[address || codec payload]`.
pub(crate) type Record = [u8; 64];

/// Bytes a record leaves to the application, after the address prefix.
///
/// The prefix is not a layout choice the application gets to make: it is what
/// [`EthPirClient::try_decrypt`] compares against to prove the retrieved record
/// really is the one queried. A minimal perfect hash maps an *unknown* key to
/// some arbitrary occupied slot, so without that comparison a lookup for an
/// address the server does not hold would return another address's value with
/// no indication anything was wrong.
pub const PAYLOAD_BYTES: usize = 64 - 20;

/// How an application turns its value into the bytes of a record.
///
/// The PIR layer below is entirely opaque — `P65536<[u8; N]>` encodes a block as
/// `N/2` little-endian 16-bit digits and never inspects it — so the record
/// layout is eth-pir's to define, and this is where an application takes it
/// over.
///
/// A codec never sees the address prefix. That keeps the not-in-set proof an
/// invariant of the crate rather than a rule implementors have to remember: the
/// worst a wrong codec can do is misread its own payload.
pub trait RecordCodec: Send + Sync + 'static {
    /// The value one record carries.
    type Value: Copy + Send + Sync;
    fn encode(value: &Self::Value) -> [u8; PAYLOAD_BYTES];
    fn decode(bytes: &[u8; PAYLOAD_BYTES]) -> Self::Value;
}

/// The original layout: a single little-endian `u256` balance.
///
/// Placed at payload offset 12 — record offset 32 — so records are byte-for-byte
/// identical to those written before the codec existed. Only the not-in-set
/// comparison changed, from 32 bytes to 20, and it is a pure relaxation: the 12
/// bytes it stopped comparing are the constant zero padding this codec still
/// writes, so no record that used to verify stops verifying.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct U256Balance;

impl RecordCodec for U256Balance {
    type Value = Balance;

    fn encode(value: &Balance) -> [u8; PAYLOAD_BYTES] {
        let mut payload = [0u8; PAYLOAD_BYTES];
        payload[12..].copy_from_slice(value);
        payload
    }

    fn decode(bytes: &[u8; PAYLOAD_BYTES]) -> Balance {
        let mut value = [0u8; 32];
        value.copy_from_slice(&bytes[12..]);
        value
    }
}

/// The fixed deployment shape: the 2 GiB `InsPIRe2-g32-2GiB-c65536` geometry
/// carrying 64 B records, for a capacity of 33,554,432 addresses.
pub fn default_shape() -> (Config<U512P65536>, DatabaseLayout<U512P65536>) {
    let collapse = Collapse::Recursion {
        gamma0: 32,
        gamma1: 1024,
        gamma2: 32,
    };
    (
        Config::with_collapse(collapse),
        DatabaseLayout::new(16384, 65536),
    )
}

/// Failure modes of the ETH PIR service layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EthPirError {
    /// The decrypted record's address does not match the queried address.
    NotInSet,
    /// The serving mutex was poisoned by a panic in another thread.
    ServerPoisoned,
    /// Forwarded from the keyword layer.
    Keyword(KeywordError<20>),
    /// Forwarded from the underlying PIR layer.
    Pir(poulpy_pir::PirError),
    /// Forwarded from keyword-directory wire I/O.
    Io {
        kind: std::io::ErrorKind,
        message: String,
    },
}

impl From<KeywordError<20>> for EthPirError {
    fn from(e: KeywordError<20>) -> Self {
        Self::Keyword(e)
    }
}

impl From<poulpy_pir::PirError> for EthPirError {
    fn from(e: poulpy_pir::PirError) -> Self {
        Self::Pir(e)
    }
}

impl From<std::io::Error> for EthPirError {
    fn from(e: std::io::Error) -> Self {
        Self::Io {
            kind: e.kind(),
            message: e.to_string(),
        }
    }
}

impl std::fmt::Display for EthPirError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotInSet => write!(f, "address not in the served set"),
            Self::ServerPoisoned => write!(f, "serving server mutex was poisoned"),
            Self::Keyword(e) => write!(f, "keyword layer: {e}"),
            Self::Pir(e) => write!(f, "PIR layer: {e}"),
            Self::Io { message, .. } => write!(f, "keyword wire I/O: {message}"),
        }
    }
}

impl std::error::Error for EthPirError {}

pub(crate) fn eth_error_to_io(error: EthPirError) -> std::io::Error {
    let kind = match &error {
        EthPirError::Io { kind, .. } => *kind,
        EthPirError::NotInSet | EthPirError::Keyword(_) => std::io::ErrorKind::InvalidData,
        EthPirError::Pir(_) | EthPirError::ServerPoisoned => std::io::ErrorKind::Other,
    };
    std::io::Error::new(kind, error)
}

pub(crate) fn record_of<C: RecordCodec>(addr: &Address, value: &C::Value) -> Record {
    let mut record = [0u8; 64];
    record[..20].copy_from_slice(addr);
    record[20..].copy_from_slice(&C::encode(value));
    record
}

/// The codec's half of a record.
pub(crate) fn payload_of(record: &Record) -> &[u8; PAYLOAD_BYTES] {
    record[20..]
        .try_into()
        .expect("a record is 20 address bytes followed by PAYLOAD_BYTES")
}

#[cfg(test)]
mod tests;
