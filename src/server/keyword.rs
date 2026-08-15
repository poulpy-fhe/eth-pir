use poulpy_pir::keyword::KeywordDirectory;

use crate::Address;

/// The keyword-helper service as its own API surface.
pub struct KeywordWire<'a> {
    pub(super) directory: &'a KeywordDirectory<20>,
}

/// The index state a server needs to restart without moving any address.
pub struct KeywordCheckpoint {
    pub directory: Vec<u8>,
    pub version: u64,
    pub keys: Vec<Address>,
}

/// What `EthPirServer::restore` had to do.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RestoreReport {
    pub placed: usize,
    pub appended: usize,
    pub vacant: usize,
}

/// Which keyword-directory payload a client needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeywordSyncMode {
    Full,
    Tail { from: usize },
}

impl KeywordWire<'_> {
    pub fn version(&self) -> u64 {
        self.directory.version()
    }

    pub fn sync_mode(&self, client_version: u64, client_tail_len: usize) -> KeywordSyncMode {
        if client_version == self.version() {
            KeywordSyncMode::Tail {
                from: client_tail_len,
            }
        } else {
            KeywordSyncMode::Full
        }
    }

    pub fn full(&self) -> Vec<u8> {
        self.try_full().unwrap_or_else(|err| panic!("{err}"))
    }

    pub fn try_full(&self) -> std::io::Result<Vec<u8>> {
        let mut blob = Vec::new();
        self.directory.write_to(&mut blob)?;
        Ok(blob)
    }

    pub fn mphf(&self) -> Vec<u8> {
        self.try_mphf().unwrap_or_else(|err| panic!("{err}"))
    }

    pub fn try_mphf(&self) -> std::io::Result<Vec<u8>> {
        let mut blob = Vec::new();
        self.directory.mphf().write_to(&mut blob)?;
        Ok(blob)
    }

    pub fn tail(&self, have: usize) -> Vec<u8> {
        self.try_tail(have).unwrap_or_else(|err| panic!("{err}"))
    }

    pub fn try_tail(&self, have: usize) -> std::io::Result<Vec<u8>> {
        let mut tail = Vec::new();
        self.directory.write_delta_envelope_from(&mut tail, have)?;
        Ok(tail)
    }
}

impl<C: crate::RecordCodec> super::EthPirServer<C> {
    pub fn keyword(&self) -> KeywordWire<'_> {
        KeywordWire {
            directory: &self.directory,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poulpy_pir::keyword::KeywordIndex;

    fn key(byte: u8) -> Address {
        [byte; 20]
    }

    fn wire_for(version: u64) -> KeywordDirectory<20> {
        let keys = [key(1), key(2), key(3)];
        KeywordDirectory::new(KeywordIndex::build(&keys).unwrap(), 16, version).unwrap()
    }

    #[test]
    fn sync_mode_forces_full_resync_after_mphf_generation_change() {
        let directory = wire_for(8);
        let wire = KeywordWire {
            directory: &directory,
        };

        assert_eq!(wire.sync_mode(7, 0), KeywordSyncMode::Full);
        assert_eq!(wire.sync_mode(8, 2), KeywordSyncMode::Tail { from: 2 });
    }
}
