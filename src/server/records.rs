use crate::{Address, Record, RecordCodec, payload_of};

const DEFAULT_MAX_THREADS: usize = 64;

fn worker_count(items: usize) -> usize {
    if items <= 1 {
        return 1;
    }
    let base = std::env::var("PIR_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&t| t >= 1)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|x| x.get())
                .unwrap_or(1)
                .min(DEFAULT_MAX_THREADS)
        });
    base.clamp(1, items)
}

#[derive(Clone, Copy)]
struct RecordsPtr(*mut Record);

// SAFETY: only dereferenced while workers write disjoint indices.
unsafe impl Send for RecordsPtr {}
unsafe impl Sync for RecordsPtr {}

pub(super) fn zeroed_records(len: usize) -> Vec<Record> {
    let mut records: Vec<Record> = Vec::with_capacity(len);
    let workers = worker_count(len);
    let ptr = RecordsPtr(records.as_mut_ptr());
    let per = len.div_ceil(workers);
    std::thread::scope(|scope| {
        for w in 0..workers {
            spawn_zero_worker(
                scope,
                ptr,
                w * per,
                per.min(len.saturating_sub(w * per)),
                len,
            );
        }
    });
    // SAFETY: workers cover 0..len exactly; panic unwinds before this line.
    unsafe { records.set_len(len) };
    records
}

fn spawn_zero_worker<'scope>(
    scope: &'scope std::thread::Scope<'scope, '_>,
    ptr: RecordsPtr,
    start: usize,
    count: usize,
    len: usize,
) {
    if start >= len {
        return;
    }
    scope.spawn(move || {
        let ptr = ptr;
        // SAFETY: zero workers write disjoint initialized slots.
        unsafe { std::ptr::write_bytes(ptr.0.add(start), 0, count) };
    });
}

pub(super) fn scatter_records<F>(dst: &mut [Record], keys: &[Address], place: F)
where
    F: Fn(usize, &Address) -> (usize, Record) + Sync,
{
    let len = dst.len();
    assert_eq!(len, keys.len(), "one record slot per key");
    let workers = worker_count(len);
    if workers <= 1 {
        scatter_serial(dst, keys, &place);
        return;
    }
    scatter_parallel(dst, keys, &place, workers);
}

fn scatter_serial<F>(dst: &mut [Record], keys: &[Address], place: &F)
where
    F: Fn(usize, &Address) -> (usize, Record),
{
    for (i, key) in keys.iter().enumerate() {
        let (slot, record) = place(i, key);
        dst[slot] = record;
    }
}

fn scatter_parallel<F>(dst: &mut [Record], keys: &[Address], place: &F, workers: usize)
where
    F: Fn(usize, &Address) -> (usize, Record) + Sync,
{
    let ptr = RecordsPtr(dst.as_mut_ptr());
    let len = dst.len();
    let per = len.div_ceil(workers);
    std::thread::scope(|scope| {
        for (w, chunk) in keys.chunks(per).enumerate() {
            scope.spawn(move || scatter_chunk(ptr, chunk, w * per, len, place));
        }
    });
}

fn scatter_chunk<F>(ptr: RecordsPtr, chunk: &[Address], offset: usize, len: usize, place: &F)
where
    F: Fn(usize, &Address) -> (usize, Record),
{
    for (k, key) in chunk.iter().enumerate() {
        let (slot, record) = place(offset + k, key);
        assert!(slot < len, "keyword index {slot} past {len} record slots");
        // SAFETY: scatter_records' MPHF bijection gives disjoint slots.
        unsafe { *ptr.0.add(slot) = record };
    }
}

pub(super) fn records_to_map<C: RecordCodec>(
    records: &[Record],
) -> std::collections::HashMap<Address, C::Value> {
    let mut map = std::collections::HashMap::with_capacity(records.len());
    for record in records {
        let mut key = [0u8; 20];
        key.copy_from_slice(&record[..20]);
        map.insert(key, C::decode(payload_of(record)));
    }
    map
}
