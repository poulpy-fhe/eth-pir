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
/// One 64-byte payload: `[address || 0^12 || balance_le]`.
pub(crate) type Record = [u8; 64];

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

pub(crate) fn address_slot(addr: &Address) -> [u8; 32] {
    let mut slot = [0u8; 32];
    slot[..20].copy_from_slice(addr);
    slot
}

pub(crate) fn record_of(addr: &Address, value: &Balance) -> Record {
    let mut record = [0u8; 64];
    record[..20].copy_from_slice(addr);
    record[32..].copy_from_slice(value);
    record
}

#[cfg(test)]
mod tests;
