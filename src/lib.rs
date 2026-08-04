//! ETH token-balance PIR service layer over `poulpy-pir`.
//!
//! This crate packages the index-based PIR core and keyword directory into a
//! fixed-shape ETH token-balance lookup service. It uses `FFT64Avx` with the
//! default `avx2-fhe` feature; build with
//! `--no-default-features --features avx512-fhe` on an AVX-512F host to switch
//! the service backend to `FFT64Avx512`.

mod client;
mod server;

pub use client::{EthPirClient, LookupState};
pub use server::{
    EthPirResponder, EthPirServer, KeywordWire, PreparedIndexRebuild, PreparedKeywordHelperFlush,
};

use poulpy_pir::config::{Collapse, Config};
use poulpy_pir::database::DatabaseLayout;
use poulpy_pir::keyword::KeywordError;
use poulpy_pir::payload::U512P65536;

#[cfg(all(feature = "avx2-fhe", feature = "avx512-fhe"))]
compile_error!("features `avx2-fhe` and `avx512-fhe` are mutually exclusive");
#[cfg(not(any(feature = "avx2-fhe", feature = "avx512-fhe")))]
compile_error!("enable exactly one backend feature: `avx2-fhe` or `avx512-fhe`");

#[cfg(feature = "avx2-fhe")]
pub type DefaultBackend = poulpy_cpu_avx::FFT64Avx;
#[cfg(feature = "avx512-fhe")]
pub type DefaultBackend = poulpy_cpu_avx512::FFT64Avx512;

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
    /// A prepared index rebuild was made against an older server state.
    StalePreparedRebuild,
    /// Forwarded from the keyword layer.
    Keyword(KeywordError<20>),
}

impl From<KeywordError<20>> for EthPirError {
    fn from(e: KeywordError<20>) -> Self {
        Self::Keyword(e)
    }
}

impl std::fmt::Display for EthPirError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotInSet => write!(f, "address not in the served set"),
            Self::StalePreparedRebuild => {
                write!(f, "prepared index rebuild no longer matches server state")
            }
            Self::Keyword(e) => write!(f, "keyword layer: {e}"),
        }
    }
}

impl std::error::Error for EthPirError {}

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
