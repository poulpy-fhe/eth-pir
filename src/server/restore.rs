use poulpy_pir::keyword::KeywordDirectory;

use super::keyword::{KeywordCheckpoint, RestoreReport};
use super::records::zeroed_records;
use super::{EthPirServer, Payload};
use crate::{Address, EthPirError, Record, RecordCodec, default_shape, record_of};

impl<C: RecordCodec> EthPirServer<C> {
    pub fn checkpoint(&self) -> Result<KeywordCheckpoint, EthPirError> {
        let mphf_len = self.directory.mphf().len();
        Ok(KeywordCheckpoint {
            directory: self.keyword().try_full()?,
            version: self.directory.version(),
            keys: checkpoint_keys(&self.records[..mphf_len]),
        })
    }

    pub fn restore(
        directory: &[u8],
        keys: &[Address],
        map: &std::collections::HashMap<Address, C::Value>,
    ) -> Result<(Self, RestoreReport), EthPirError> {
        let (config, layout) = default_shape();
        Self::restore_with(config, layout, directory, keys, map)
    }

    pub fn restore_with(
        config: poulpy_pir::config::Config<Payload>,
        layout: poulpy_pir::database::DatabaseLayout<Payload>,
        directory: &[u8],
        keys: &[Address],
        map: &std::collections::HashMap<Address, C::Value>,
    ) -> Result<(Self, RestoreReport), EthPirError> {
        let mut directory = read_checkpoint_directory(directory, keys)?;
        let mut report = RestoreReport::default();
        let mut records = zeroed_records(directory.len());
        place_checkpoint_slots::<C>(keys, map, &mut records, &mut report);
        place_delta_slots::<C>(&directory, map, &mut records, &mut report);
        append_new_slots::<C>(&mut directory, map, &mut records, &mut report)?;
        Self::assemble(config, layout, directory, records).map(|server| (server, report))
    }
}

fn checkpoint_keys(records: &[Record]) -> Vec<Address> {
    records
        .iter()
        .map(|record| {
            let mut addr = [0u8; 20];
            addr.copy_from_slice(&record[..20]);
            addr
        })
        .collect()
}

fn read_checkpoint_directory(
    directory: &[u8],
    keys: &[Address],
) -> Result<KeywordDirectory<20>, EthPirError> {
    let directory = KeywordDirectory::<20>::read_from(&mut { directory })?;
    let mphf_len = directory.mphf().len();
    if keys.len() == mphf_len {
        return Ok(directory);
    }
    Err(EthPirError::Io {
        kind: std::io::ErrorKind::InvalidData,
        message: format!(
            "checkpoint holds {} keys but its MPHF addresses {mphf_len}",
            keys.len()
        ),
    })
}

fn place_checkpoint_slots<C: RecordCodec>(
    keys: &[Address],
    map: &std::collections::HashMap<Address, C::Value>,
    records: &mut [Record],
    report: &mut RestoreReport,
) {
    for (slot, addr) in keys.iter().enumerate() {
        match map.get(addr) {
            Some(value) if *addr != [0u8; 20] => place_record::<C>(records, slot, addr, value),
            _ => report.vacant += 1,
        }
        if records[slot][..20] == *addr && *addr != [0u8; 20] {
            report.placed += 1;
        }
    }
}

fn place_delta_slots<C: RecordCodec>(
    directory: &KeywordDirectory<20>,
    map: &std::collections::HashMap<Address, C::Value>,
    records: &mut [Record],
    report: &mut RestoreReport,
) {
    let mphf_len = directory.mphf().len();
    for (addr, value) in map {
        let slot = directory.index(addr);
        if slot >= mphf_len && slot < records.len() {
            place_record::<C>(records, slot, addr, value);
            report.placed += 1;
        }
    }
}

fn append_new_slots<C: RecordCodec>(
    directory: &mut KeywordDirectory<20>,
    map: &std::collections::HashMap<Address, C::Value>,
    records: &mut Vec<Record>,
    report: &mut RestoreReport,
) -> Result<(), EthPirError> {
    for (addr, value) in map {
        if is_restored_record(directory, records, addr) {
            continue;
        }
        let appended = directory.push(addr)?;
        debug_assert_eq!(appended, records.len());
        records.push(record_of::<C>(addr, value));
        report.appended += 1;
    }
    Ok(())
}

fn is_restored_record(
    directory: &KeywordDirectory<20>,
    records: &[Record],
    addr: &Address,
) -> bool {
    let slot = directory.index(addr);
    slot < records.len() && records[slot][..20] == *addr
}

fn place_record<C: RecordCodec>(
    records: &mut [Record],
    slot: usize,
    addr: &Address,
    value: &C::Value,
) {
    records[slot] = record_of::<C>(addr, value);
}
