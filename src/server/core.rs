use std::marker::PhantomData;
use std::time::Instant;

use poulpy_pir::keyword::{KeywordDirectory, KeywordError, KeywordIndex};

use super::records::{scatter_records, zeroed_records};
use super::report::{InitTimings, MemoryReport};
use super::{EthPirServer, Payload, PirServer};
use crate::{Address, EthPirError, Record, RecordCodec, default_shape, record_of};

impl<C: RecordCodec> EthPirServer<C> {
    /// Instantiates DB + keyword helper from the initial address -> value map
    /// at the fixed 2 GiB shape.
    pub fn init(map: &std::collections::HashMap<Address, C::Value>) -> Result<Self, EthPirError> {
        let (config, layout) = default_shape();
        Self::init_with(config, layout, map)
    }

    /// `init` at a caller-chosen shape, mainly for tests.
    pub fn init_with(
        config: poulpy_pir::config::Config<Payload>,
        layout: poulpy_pir::database::DatabaseLayout<Payload>,
        map: &std::collections::HashMap<Address, C::Value>,
    ) -> Result<Self, EthPirError> {
        Self::init_with_timed(config, layout, map).map(|(server, _)| server)
    }

    /// [`init_with`](Self::init_with) with a per-step timing breakdown.
    pub fn init_with_timed(
        config: poulpy_pir::config::Config<Payload>,
        layout: poulpy_pir::database::DatabaseLayout<Payload>,
        map: &std::collections::HashMap<Address, C::Value>,
    ) -> Result<(Self, InitTimings), EthPirError> {
        let mut timings = InitTimings::default();
        let (directory, keys) = build_directory::<C>(config, layout, map, &mut timings)?;
        let records = scatter_map_records::<C>(&directory, &keys, map, &mut timings);
        let server = Self::assemble_timed(config, layout, directory, records, &mut timings)?;
        Ok((server, timings))
    }

    pub(super) fn assemble(
        config: poulpy_pir::config::Config<Payload>,
        layout: poulpy_pir::database::DatabaseLayout<Payload>,
        directory: KeywordDirectory<20>,
        records: Vec<Record>,
    ) -> Result<Self, EthPirError> {
        let mut timings = InitTimings::default();
        Self::assemble_timed(config, layout, directory, records, &mut timings)
    }

    fn assemble_timed(
        config: poulpy_pir::config::Config<Payload>,
        layout: poulpy_pir::database::DatabaseLayout<Payload>,
        directory: KeywordDirectory<20>,
        records: Vec<Record>,
        timings: &mut InitTimings,
    ) -> Result<Self, EthPirError> {
        let (server, staging) = build_pir_server(config, layout, &records, timings)?;
        Ok(Self {
            directory,
            records,
            serving: std::sync::Arc::new(std::sync::Mutex::new(server)),
            staging,
            pending: 0,
            codec: PhantomData,
        })
    }

    pub fn update(&mut self, addr: Address, value: C::Value) -> Result<(), EthPirError> {
        let i = self.directory.index(&addr);
        if i < self.records.len() && self.records[i][..20] == addr {
            self.records[i] = record_of::<C>(&addr, &value);
        } else {
            self.insert_or_reclaim(i, addr, value)?;
        }
        self.pending += 1;
        Ok(())
    }

    fn insert_or_reclaim(
        &mut self,
        slot: usize,
        addr: Address,
        value: C::Value,
    ) -> Result<(), EthPirError> {
        match self.directory.push(&addr) {
            Ok(appended) => {
                debug_assert_eq!(appended, self.records.len());
                self.records.push(record_of::<C>(&addr, &value));
                Ok(())
            }
            Err(KeywordError::DuplicateKey(key)) if key == addr && slot < self.records.len() => {
                self.records[slot] = record_of::<C>(&addr, &value);
                Ok(())
            }
            Err(e) => Err(e.into()),
        }
    }

    pub fn len(&self) -> usize {
        self.directory.len()
    }

    pub fn is_empty(&self) -> bool {
        self.directory.is_empty()
    }

    pub fn pending(&self) -> usize {
        self.pending
    }

    pub fn memory_report(&self) -> MemoryReport {
        let server = self
            .serving
            .lock()
            .map_err(|_| EthPirError::ServerPoisoned)
            .unwrap_or_else(|err| panic!("{err}"))
            .memory_report();
        MemoryReport {
            serving_database: server.database,
            staging_database: self.staging.allocated_bytes(),
            precomputation: server.precomputation,
            online_scratch_pool: server.online_scratch_pool,
            records: std::mem::size_of_val(self.records.as_slice()),
            keyword_directory: self.keyword().full().len()
                + self.directory.delta_len() * (std::mem::size_of::<Address>() + 24),
        }
    }
}

fn build_directory<C: RecordCodec>(
    config: poulpy_pir::config::Config<Payload>,
    layout: poulpy_pir::database::DatabaseLayout<Payload>,
    map: &std::collections::HashMap<Address, C::Value>,
    timings: &mut InitTimings,
) -> Result<(KeywordDirectory<20>, Vec<Address>), EthPirError> {
    let t = Instant::now();
    let keys: Vec<Address> = map.keys().copied().collect();
    let mphf = KeywordIndex::build(&keys)?;
    let capacity = layout.num_payloads(config.column_height());
    let directory = KeywordDirectory::new(mphf, capacity, 0)?;
    timings.keyword_index = t.elapsed();
    Ok((directory, keys))
}

fn scatter_map_records<C: RecordCodec>(
    directory: &KeywordDirectory<20>,
    keys: &[Address],
    map: &std::collections::HashMap<Address, C::Value>,
    timings: &mut InitTimings,
) -> Vec<Record> {
    let t = Instant::now();
    let mut records = zeroed_records(map.len());
    scatter_records(&mut records, keys, |_, addr| {
        (directory.index(addr), record_of::<C>(addr, &map[addr]))
    });
    timings.records_scatter = t.elapsed();
    records
}

fn build_pir_server(
    config: poulpy_pir::config::Config<Payload>,
    layout: poulpy_pir::database::DatabaseLayout<Payload>,
    records: &[Record],
    timings: &mut InitTimings,
) -> Result<(PirServer, super::PirDatabase), EthPirError> {
    let t = Instant::now();
    let server = PirServer::try_new(config, layout)?;
    #[cfg(feature = "cblas-gemm")]
    let server = server.with_gemm(poulpy_pir::server::CblasDgemm);
    let mut server = server;
    timings.server_alloc = t.elapsed();
    init_serving_database(&mut server, records, timings)?;
    let t = Instant::now();
    let staging = server.new_database();
    timings.staging_alloc = t.elapsed();
    Ok((server, staging))
}

fn init_serving_database(
    server: &mut PirServer,
    records: &[Record],
    timings: &mut InitTimings,
) -> Result<(), EthPirError> {
    let t = Instant::now();
    server.try_update_shard(0, records)?;
    timings.database_encode = t.elapsed();
    let t = Instant::now();
    server.generate_query_mask();
    timings.query_mask = t.elapsed();
    let t = Instant::now();
    server.offline();
    timings.offline = t.elapsed();
    Ok(())
}
