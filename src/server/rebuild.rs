use std::time::Instant;

use poulpy_pir::keyword::KeywordDirectory;

use super::EthPirServer;
use super::records::{records_to_map, scatter_records, zeroed_records};
use super::report::{KeywordRebuildTimings, RefreshTimings};
use crate::{Address, EthPirError, Record, RecordCodec, record_of};

impl<C: RecordCodec> EthPirServer<C> {
    pub fn rebuild_database(&mut self) -> Option<RefreshTimings> {
        self.try_rebuild_database()
            .unwrap_or_else(|err| panic!("{err}"))
    }

    pub fn try_rebuild_database(&mut self) -> Result<Option<RefreshTimings>, EthPirError> {
        if self.pending == 0 {
            return Ok(None);
        }
        Ok(Some(self.refresh()?))
    }

    pub fn rebuild_keyword_index(&mut self) -> Result<KeywordRebuildTimings, EthPirError> {
        let t = Instant::now();
        let map = records_to_map::<C>(&self.records);
        let mut timings = self.rebuild_keyword_index_from(&map)?;
        timings.collect_keys = t.elapsed();
        Ok(timings)
    }

    pub fn rebuild_keyword_index_from(
        &mut self,
        map: &std::collections::HashMap<Address, C::Value>,
    ) -> Result<KeywordRebuildTimings, EthPirError> {
        let mut timings = KeywordRebuildTimings::default();
        let keys = collect_keys::<C>(map, &mut timings);
        let next = rebuilt_directory(&self.directory, &keys, &mut timings)?;
        self.records = scatter_rebuilt_records::<C>(&next, &keys, map, &mut timings);
        self.directory = next;
        timings.refresh = self.refresh()?;
        Ok(timings)
    }

    fn refresh(&mut self) -> Result<RefreshTimings, EthPirError> {
        let mut timings = RefreshTimings::default();
        encode_staging(&mut self.staging, &self.records, &mut timings)?;
        let precomputation = precompute(self, &mut timings)?;
        install(self, precomputation, &mut timings)?;
        self.pending = 0;
        Ok(timings)
    }
}

fn collect_keys<C: RecordCodec>(
    map: &std::collections::HashMap<Address, C::Value>,
    timings: &mut KeywordRebuildTimings,
) -> Vec<Address> {
    let t = Instant::now();
    let keys: Vec<Address> = map.keys().copied().collect();
    timings.collect_keys = t.elapsed();
    keys
}

fn rebuilt_directory(
    previous: &KeywordDirectory<20>,
    keys: &[Address],
    timings: &mut KeywordRebuildTimings,
) -> Result<KeywordDirectory<20>, EthPirError> {
    let t = Instant::now();
    let next = previous.rebuilt(keys)?;
    timings.mphf_rebuild = t.elapsed();
    Ok(next)
}

fn scatter_rebuilt_records<C: RecordCodec>(
    directory: &KeywordDirectory<20>,
    keys: &[Address],
    map: &std::collections::HashMap<Address, C::Value>,
    timings: &mut KeywordRebuildTimings,
) -> Vec<Record> {
    let t = Instant::now();
    let mut records = zeroed_records(keys.len());
    scatter_records(&mut records, keys, |_, addr| {
        (directory.index(addr), record_of::<C>(addr, &map[addr]))
    });
    timings.permute = t.elapsed();
    records
}

fn encode_staging(
    staging: &mut super::PirDatabase,
    records: &[Record],
    timings: &mut RefreshTimings,
) -> Result<(), EthPirError> {
    let t = Instant::now();
    staging.try_encode_shard(0, records)?;
    timings.database_encode = t.elapsed();
    Ok(())
}

fn precompute<C: RecordCodec>(
    server: &mut EthPirServer<C>,
    timings: &mut RefreshTimings,
) -> Result<poulpy_pir::server::ServerPrecomputation<crate::DefaultBackend>, EthPirError> {
    let t = Instant::now();
    let mut context = {
        let serving = server
            .serving
            .lock()
            .map_err(|_| EthPirError::ServerPoisoned)?;
        serving.precomp_context()
    };
    let (precomputation, _) = context.offline_for(&mut server.staging);
    timings.precompute = t.elapsed();
    Ok(precomputation)
}

fn install<C: RecordCodec>(
    server: &mut EthPirServer<C>,
    precomputation: poulpy_pir::server::ServerPrecomputation<crate::DefaultBackend>,
    timings: &mut RefreshTimings,
) -> Result<(), EthPirError> {
    let t = Instant::now();
    server
        .serving
        .lock()
        .map_err(|_| EthPirError::ServerPoisoned)?
        .install(&mut server.staging, precomputation);
    timings.install = t.elapsed();
    Ok(())
}
