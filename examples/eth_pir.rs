//! End-to-end ETH PIR demo at the real 2 GiB shape.
//!
//! Run from the `eth-pir` repository root:
//!
//! ```sh
//! cargo test --release --lib
//! ```
//!
//! The full 2 GiB demo is a production AVX workload. Build it explicitly:
//!
//! ```sh
//! RUSTFLAGS="-C target-feature=+avx2,+fma" \
//! cargo run --release --features avx2-fhe --example eth_pir
//! ```
//!
//! On an AVX-512F host, switch to `FFT64Avx512` with:
//!
//! ```sh
//! RUSTFLAGS="-C target-feature=+avx2,+fma,+avx512f" \
//! cargo run --release --no-default-features --features avx512-fhe --example eth_pir
//! ```
//!
//! Add `cblas-gemm` to use the system CBLAS/OpenBLAS dgemm for the offline mask
//! product:
//!
//! ```sh
//! RUSTFLAGS="-C target-feature=+avx2,+fma,+avx512f" \
//! cargo run --release --no-default-features --features "avx512-fhe,cblas-gemm" \
//!   --example eth_pir
//! ```
//!
//! On multi-socket Linux hosts serving single-query latency, add
//! `numa-db-interleave` to interleave the database allocation across NUMA nodes:
//!
//! ```sh
//! RUSTFLAGS="-C target-feature=+avx2,+fma,+avx512f" \
//! cargo run --release --no-default-features \
//!   --features "avx512-fhe,cblas-gemm,numa-db-interleave" \
//!   --example eth_pir
//! ```
//!
//! With `cblas-gemm`, install the optimized pthread OpenBLAS development
//! package first, for example `libopenblas-pthread-dev` on Ubuntu. Generic CBLAS
//! providers such as GSL CBLAS are not the intended performance path. The
//! example re-execs once with `OPENBLAS_NUM_THREADS=1` when that variable is
//! absent, matching `poulpy-pir`'s PIR driver.
//!
//! Always use `--release`: debug builds are dramatically slower. The demo
//! starts with 16 M addresses in the fixed `InsPIRe2-g32-2GiB-c65536` shape,
//! then applies 1 M balance updates and 50 K inserted addresses. Both rebuilds
//! refresh one long-lived server in place: the
//! encoded database and its precomputation are double-buffered, but the
//! server's one-time setup (CRS masks, online scratch pool) is not.
//!
//! What this exercises:
//!
//! - server initialization from deterministic `(address, balance)` data,
//! - client bootstrap from the keyword helper's full directory blob,
//! - a private lookup and address-in-record verification,
//! - 1 M balance updates plus 50 K new-key insertions,
//! - validated keyword delta sync before the rebuild,
//! - database rebuild against the current keyword helper,
//! - append-only delta lookup before the MPHF rebuild,
//! - keyword-index rebuild and client resync,
//! - peak RSS reporting from `/proc/self/status` on Linux.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use eth_pir::{Address, Balance, EthPirClient, EthPirError, EthPirServer, default_shape};

const INITIAL_ADDRESSES: u64 = 16_000_000;
const UPDATED_ADDRESSES: u64 = 1_000_000;
const NEW_ADDRESSES: u64 = 50_000;
const BATCHES: [usize; 2] = [32, 64];

fn main() {
    // - Step 0: normalize the BLAS runtime before any PIR work starts.
    // - With `cblas-gemm`, the pthread OpenBLAS worker pool is sized from the
    //   constructor-time environment, earlier than most Rust code can configure
    //   it.
    // - If the caller did not already choose a thread count, re-exec once with
    //   `OPENBLAS_NUM_THREADS=1`.
    // - The PIR server already parallelizes dgemm calls at a higher level;
    //   letting every OpenBLAS call spawn its own full worker pool causes
    //   oversubscription and can slow unrelated phases too.
    #[cfg(feature = "cblas-gemm")]
    if std::env::var_os("OPENBLAS_NUM_THREADS").is_none() {
        use std::os::unix::process::CommandExt;
        let exe = std::env::current_exe().expect("current_exe for OpenBLAS re-exec");
        let err = std::process::Command::new(exe)
            .args(std::env::args_os().skip(1))
            .env("OPENBLAS_NUM_THREADS", "1")
            .exec();
        panic!("re-exec with OPENBLAS_NUM_THREADS=1 failed: {err}");
    }

    // - Step 1: build deterministic demo state.
    // - Create 16 M synthetic ETH addresses and little-endian u256 balances from
    //   splitmix64-derived bytes.
    // - Exercise the real 2 GiB geometry with reproducible data.
    // - Avoid shipping a huge fixture while still letting assertions recompute
    //   expected values cheaply.
    let t = Instant::now();
    let map: HashMap<Address, Balance> = (0..INITIAL_ADDRESSES)
        .map(|i| {
            let a = address_at(i);
            let b = balance_of(&a);
            (a, b)
        })
        .collect();
    println!(
        "initial map ({INITIAL_ADDRESSES} addresses) : {:?}",
        t.elapsed()
    );

    // - Step 2: initialize the server.
    // - Build the keyword MPHF.
    // - Place each 64-byte record at its private-index slot.
    // - Encode the PIR database, generate the query mask, and run
    //   query-independent offline preprocessing.
    // - After this point the server can answer index PIR queries, while clients
    //   can map ETH addresses to PIR indices through the keyword helper.
    let t = Instant::now();
    let (mut server, init) = {
        let (config, layout) = default_shape();
        EthPirServer::init_with_timed(config, layout, &map).expect("init")
    };
    // - Release the input map after the server copies records into its own
    //   storage, keeping peak memory closer to a production loader after it
    //   publishes its first database.
    drop(map);
    println!("SERVER init (MPHF+fill+offline)  : {:?}", t.elapsed());
    let peak_after_init = peak_rss_bytes();
    println!(
        "  keyword index (MPHF build)     : {:?}",
        init.keyword_index
    );
    println!(
        "  records alloc + scatter        : {:?}",
        init.records_scatter
    );
    println!(
        "  server alloc (DB + NUMA touch) : {:?}   [one-time]",
        init.server_alloc
    );
    println!(
        "  database encode                : {:?}",
        init.database_encode
    );
    println!(
        "  CRS query masks                : {:?}   [one-time]",
        init.query_mask
    );
    println!(
        "  offline (precomp + pool warm)  : {:?}   [pool warm is one-time]",
        init.offline
    );
    println!(
        "  staging database alloc         : {:?}   [one-time]",
        init.staging_alloc
    );

    // - Step 3: bootstrap the client from the keyword helper.
    // - Download the full serialized directory: MPHF parameters, capacity,
    //   version, and any delta overlay keys.
    // - The PIR query itself is by index, but the external API is by ETH
    //   address.
    // - The client needs the same address-to-index mapping as the server before
    //   it can form a private query.
    let t = Instant::now();
    let blob = server.keyword().full();
    let mut client = EthPirClient::new(&blob).expect("client bootstrap");
    println!(
        "CLIENT bootstrap ({} B blob)  : {:?}",
        blob.len(),
        t.elapsed()
    );

    // - Step 4: perform one private lookup and verify the decrypted record.
    // - Map the sample address to an index and create a PIR query.
    // - Have the server answer without learning the index.
    // - Decrypt and check that the returned record contains the requested
    //   address.
    // - Address verification turns MPHF misses or stale mappings into
    //   `NotInSet` instead of returning a wrong balance.
    let sample_existing = address_at(UPDATED_ADDRESSES - 1);
    let t = Instant::now();
    let balance = lookup(&server, &mut client, sample_existing).expect("address is in the set");
    let balance_lookup = t.elapsed();
    println!("lookup (1 query round trip)      : {balance_lookup:?}");
    assert_eq!(balance, balance_of(&sample_existing));
    println!("  balance                        : {}", u256_hex(&balance));

    // - Step 4b: answer a batch of independent lookups in one call.
    // - `respond_batch` streams the plaintext database once for the whole batch
    //   instead of once per query, and runs the per-query FHE tail across the
    //   worker pool.
    // - Addresses are spread across the key space so the batch cannot benefit
    //   from locality a real workload would not have.
    // - The client still builds one private query per address and verifies each
    //   returned record independently; batching is a server-side optimization
    //   with no effect on what a client sees.
    let single_rate = 1.0 / balance_lookup.as_secs_f64();
    for batch in BATCHES {
        let batch_addresses: Vec<Address> = (0..batch)
            .map(|k| address_at(k as u64 * (INITIAL_ADDRESSES / batch as u64)))
            .collect();
        let t = Instant::now();
        let mut queries = Vec::with_capacity(batch);
        let mut states = Vec::with_capacity(batch);
        for addr in &batch_addresses {
            let (query, state) = client.query(*addr);
            queries.push(query);
            states.push(state);
        }
        let batch_build = t.elapsed();

        let t = Instant::now();
        let responses = server.respond_batch(&queries);
        let batch_wall = t.elapsed();

        for ((response, state), addr) in responses.iter().zip(&states).zip(&batch_addresses) {
            let value = client
                .decrypt(response, state)
                .expect("address is in the set");
            assert_eq!(value, balance_of(addr));
        }
        let rate = batch as f64 / batch_wall.as_secs_f64();
        println!(
            "lookup batch of {batch:<3}            : {batch_wall:?}  (all {batch} records verify)"
        );
        println!(
            "  per query (amortized)          : {:?}",
            batch_wall / batch as u32
        );
        println!(
            "  throughput                     : {rate:.1} queries/s   ({:.1}x the single-query rate)",
            rate / single_rate
        );
        println!("  client-side build of {batch:<3} queries : {batch_build:?}");
    }

    // - Step 5: apply many existing-address updates and many insertions.
    // - Update the first 1M existing token balances.
    // - Append 50K new addresses to the keyword delta overlay.
    // - Leave the served database unchanged for now.
    // - Writes land in plaintext immediately, so a production service absorbs
    //   them without rerunning the expensive precomputation on every update.
    let sample_new = address_at(INITIAL_ADDRESSES);
    let t = Instant::now();
    for i in 0..UPDATED_ADDRESSES {
        let addr = address_at(i);
        server
            .update(addr, updated_balance_of(&addr))
            .expect("existing-address update");
    }
    for i in 0..NEW_ADDRESSES {
        let addr = address_at(INITIAL_ADDRESSES + i);
        server
            .update(addr, balance_of(&addr))
            .expect("new-address insert");
    }
    println!(
        "UPDATE ({UPDATED_ADDRESSES} updates + {NEW_ADDRESSES} inserts): {:?}",
        t.elapsed()
    );

    // - Step 6: synchronize only the keyword delta.
    // - Fetch the suffix of keyword entries the client has not seen yet.
    // - Validate the delta envelope before applying it.
    // - New address-to-index mappings are now known to the client, but their
    //   record bodies are still zero until the server rebuilds the database.
    // - The assertions show that old data remains served for updated addresses
    //   and inserted addresses are not retrievable yet.
    let tail = server.keyword().tail(client.tail_len());
    let delta_wire_bytes = tail.len();
    let directory_before_compaction = server.keyword().full().len();
    client.apply_tail(&tail).expect("tail sync");
    assert_eq!(
        lookup(&server, &mut client, sample_existing),
        Ok(balance_of(&sample_existing))
    );
    assert_eq!(
        lookup(&server, &mut client, sample_new),
        Err(EthPirError::NotInSet)
    );
    println!(
        "delta sync: {NEW_ADDRESSES} new keys visible (pending = {})",
        server.pending()
    );

    // - Step 7: publish the pending updates.
    // - Rebuild the PIR database from the current keyword helper: base MPHF plus
    //   append-only delta.
    // - Rerun the precomputation and swap it in with its database, as a pair.
    // - Do not rebuild the MPHF.
    // - Queued writes are not visible to PIR responses until the
    //   encrypted/preprocessed database has been rebuilt.
    // - After the swap, clients can retrieve updated records; inserted records
    //   are exercised in the next step through the append-only delta path.
    // - Serve a continuous query load from another thread across the whole
    //   rebuild, to show it does not take the server offline.
    // - Every response must decode to either the pre-swap or the post-swap
    //   balance: the database and its precomputation are swapped as a pair, so
    //   no query can ever observe one without the other.
    let responder = server.responder();
    let stop = Arc::new(AtomicBool::new(false));
    let load = {
        let stop = stop.clone();
        let blob = blob.clone();
        std::thread::spawn(move || {
            let mut client: EthPirClient = EthPirClient::new(&blob).expect("load-generator client");
            let (mut served, mut before, mut after) = (0u64, 0u64, 0u64);
            let mut worst = std::time::Duration::ZERO;
            while !stop.load(Ordering::Relaxed) {
                let t = Instant::now();
                let (query, state) = client.query(sample_existing);
                let response = responder.respond(&query);
                let balance = client.decrypt(&response, &state).expect("in set");
                worst = worst.max(t.elapsed());
                served += 1;
                if balance == balance_of(&sample_existing) {
                    before += 1;
                } else if balance == updated_balance_of(&sample_existing) {
                    after += 1;
                } else {
                    panic!("torn read: response matched neither the old nor the new balance");
                }
            }
            (served, before, after, worst)
        })
    };

    let t = Instant::now();
    let refresh = server.rebuild_database().expect("updates were pending");
    let rebuild_wall = t.elapsed();
    stop.store(true, Ordering::Relaxed);
    let (served, before, after, worst) = load.join().expect("load generator");
    println!("REBUILD DATABASE                 : {rebuild_wall:?}");
    println!(
        "  database encode from records   : {:?}",
        refresh.database_encode
    );
    println!(
        "  precompute (no server lock)    : {:?}",
        refresh.precompute
    );
    println!("  install + free old precomp     : {:?}", refresh.install);
    println!(
        "  served concurrently            : {served} queries ({before} pre-swap, {after} post-swap), worst {worst:?}"
    );
    assert!(
        before > 0 && after > 0,
        "the load generator should straddle the swap"
    );
    assert_eq!(
        lookup(&server, &mut client, sample_existing),
        Ok(updated_balance_of(&sample_existing))
    );
    println!("  post-rebuild: sample update verifies");
    let peak_after_rebuild = peak_rss_bytes();

    // - Step 8: query an appended address before rebuilding the MPHF.
    // - Keep both server and client on directory version 0.
    // - Use only the append-only delta downloaded in Step 6 to map the new
    //   address to its post-MPHF index.
    // - Demonstrate the intended steady-state path between MPHF rebuilds:
    //   clients fetch small delta tails for new addresses and can query those
    //   addresses after the server rebuilds the database.
    assert_eq!(server.keyword().version(), 0);
    assert_eq!(client.version(), 0);
    let t = Instant::now();
    let inserted_balance =
        lookup(&server, &mut client, sample_new).expect("new address via append-only delta");
    println!("lookup appended address (delta)  : {:?}", t.elapsed());
    assert_eq!(inserted_balance, balance_of(&sample_new));
    println!("  append-only delta lookup verifies");

    // - Step 9: compact the delta back into a fresh MPHF.
    // - Construct a fresh MPHF over the complete key set, permute records into
    //   its order, then rebuild the database so the served layout matches.
    // - The new directory version is published only once that database is live,
    //   because clients query by index and every index moves.
    // - Compacting keeps client lookup state small and lookup behavior
    //   predictable.
    let t = Instant::now();
    let rebuild = server
        .rebuild_keyword_index()
        .expect("keyword index rebuild");
    println!("REBUILD KEYWORD INDEX            : {:?}", t.elapsed());
    println!(
        "  collect keys from records      : {:?}",
        rebuild.collect_keys
    );
    println!(
        "  MPHF rebuild                   : {:?}",
        rebuild.mphf_rebuild
    );
    println!("  permute records to new order   : {:?}", rebuild.permute);
    println!(
        "  database encode from records   : {:?}",
        rebuild.refresh.database_encode
    );
    println!(
        "  precompute (no server lock)    : {:?}",
        rebuild.refresh.precompute
    );
    println!(
        "  install + free old precomp     : {:?}",
        rebuild.refresh.install
    );
    assert_eq!(server.keyword().version(), 1);

    // - Step 11: resynchronize the client after an MPHF rebuild.
    // - Download a fresh full directory.
    // - Replace the client's old version-0 mapping with the version-1 mapping.
    // - After rebuild, indices may change; a client that kept the old MPHF would
    //   query stale positions and fail record verification.
    let t = Instant::now();
    client.resync(&server.keyword().full()).expect("resync");
    println!("CLIENT resync (version 0 -> 1)   : {:?}", t.elapsed());
    assert_eq!(
        lookup(&server, &mut client, sample_existing),
        Ok(updated_balance_of(&sample_existing))
    );
    assert_eq!(
        lookup(&server, &mut client, sample_new),
        Ok(balance_of(&sample_new))
    );
    println!("  post-rebuild: sample lookups verify");
    let peak_after_keyword = peak_rss_bytes();

    // - Step 12: report the wire sizes the deployment actually pays for.
    // - The MPHF blob is the one-time client bootstrap; the delta is what an
    //   incremental sync costs instead, as a naive list of 20-byte addresses.
    // - Their ratio is what decides how long the append-only path stays
    //   cheaper than re-downloading a rebuilt MPHF.
    let (config, layout) = default_shape();
    let mphf_bytes = server.keyword().mphf().len();
    let full_bytes = server.keyword().full().len();
    let query_bytes = config.query_size(layout).total_size();
    let response_bytes = config.response_size(layout).total_size();
    println!();
    println!("WIRE SIZES");
    println!(
        "  client query                     : {:>12}",
        bytes(query_bytes)
    );
    println!(
        "  server response                  : {:>12}",
        bytes(response_bytes)
    );
    println!(
        "  keyword MPHF (client bootstrap)  : {:>12}   {:.3} bits/key over {} keys",
        bytes(mphf_bytes),
        mphf_bytes as f64 * 8.0 / server.len() as f64,
        server.len()
    );
    println!(
        "  keyword delta, {NEW_ADDRESSES} inserts     : {:>12}   naive 20 B/key list + 48 B envelope",
        bytes(delta_wire_bytes)
    );
    println!(
        "  full directory before compaction : {:>12}   MPHF + that delta overlay",
        bytes(directory_before_compaction)
    );
    println!(
        "  full directory after compaction  : {:>12}   delta folded back into the MPHF",
        bytes(full_bytes)
    );
    println!(
        "  -> a delta stays cheaper than refetching the MPHF for ~{} inserts",
        mphf_bytes / 20
    );

    // - Step 13: account for the memory the server holds.
    // - `MemoryReport` names the allocations that scale; VmHWM is what the
    //   kernel actually saw.
    // - The gap between `refresh_peak` and VmHWM is the transient working set
    //   of the precomputation plus the example's own 16 M-address fixtures.
    let mem = server.memory_report();
    println!();
    println!("MEMORY BREAKDOWN");
    println!(
        "  online scratch pool              : {:>12}   one-time, sized by PIR_THREADS, not by the DB",
        bytes(mem.online_scratch_pool)
    );
    println!(
        "  serving database                 : {:>12}",
        bytes(mem.serving_database)
    );
    println!(
        "  staging database                 : {:>12}   the retired buffer, refilled next refresh",
        bytes(mem.staging_database)
    );
    println!(
        "  precomputation (mask side)       : {:>12}   counted buffers only, see note",
        bytes(mem.precomputation)
    );
    println!(
        "  plaintext records                : {:>12}",
        bytes(mem.records)
    );
    println!(
        "  keyword directory (approx)       : {:>12}",
        bytes(mem.keyword_directory)
    );
    println!("  --------------------------------------------------");
    println!(
        "  accounted steady state           : {:>12}",
        bytes(mem.total())
    );
    println!(
        "  + second precomp during refresh  : {:>12}   between precompute and install",
        bytes(mem.precomputation)
    );
    println!(
        "  = expected refresh peak          : {:>12}",
        bytes(mem.refresh_peak())
    );
    if let Some(peak) = peak_rss_bytes() {
        let peak = peak as usize;
        println!("  measured peak (VmHWM)            : {:>12}", bytes(peak));
        println!(
            "  unaccounted                      : {:>12}",
            bytes(peak.saturating_sub(mem.refresh_peak()))
        );
    }

    // - Which phase actually set the high-water mark. VmHWM only ever rises, so
    //   a phase that does not move it added nothing to the peak.
    println!();
    println!("  high-water mark after each phase:");
    for (label, value) in [
        ("init", peak_after_init),
        ("database rebuild", peak_after_rebuild),
        ("keyword compaction", peak_after_keyword),
    ] {
        if let Some(v) = value {
            println!("    {label:<28} : {:>12}", bytes(v as usize));
        }
    }
    println!();
    println!("  The unaccounted remainder is transient. Candidates, largest first:");
    println!("  the scratch each offline worker allocates for its own parallel region,");
    println!("  the precomputation's BSGS giant-step FFT plans (type-erased, so the");
    println!("  report cannot size them), and this example's own {INITIAL_ADDRESSES}-entry");
    println!("  input HashMap, freed right after init but counted in the high-water mark.");
    println!();
    println!("RESULT                           : OK");
}

// - Run one complete address lookup round trip.
// - Keep the main flow readable while preserving the service boundary: the
//   client creates a private query and keeps decrypt state; the server only sees
//   the encrypted query; the client verifies the returned record before exposing
//   a balance.
fn lookup(
    server: &EthPirServer,
    client: &mut EthPirClient,
    addr: Address,
) -> Result<Balance, EthPirError> {
    let (query, state) = client.query(addr);
    let response = server.respond(&query);
    client.decrypt(&response, &state)
}

// - Deterministically derive a 20-byte ETH-like address from an integer.
// - Treat this as test data, not an address-generation scheme.
// - Keep the mapping stable so the example can re-create expected keys without
//   storing fixtures.
fn address_at(i: u64) -> Address {
    let mut key = [0u8; 20];
    let mut z = i.wrapping_add(0x9e3779b97f4a7c15);
    for chunk in key.chunks_mut(8) {
        z = z.wrapping_add(0x9e3779b97f4a7c15);
        let x = splitmix64(z);
        chunk.copy_from_slice(&x.to_le_bytes()[..chunk.len()]);
    }
    key
}

// - Deterministically derive a 32-byte little-endian balance from an address.
// - Match the record format, which stores balances as little-endian u256 values.
// - Derive balances from keys so every address has a stable expected value for
//   assertions.
fn balance_of(key: &Address) -> Balance {
    let mut z = u64::from_le_bytes(key[..8].try_into().unwrap());
    let mut value = [0u8; 32];
    for word in 0..4 {
        z = z.wrapping_add(0x9e3779b97f4a7c15);
        value[word * 8..][..8].copy_from_slice(&splitmix64(z).to_le_bytes());
    }
    value
}

// - Deterministically derive the post-update value for an existing address.
// - Keep it different from `balance_of` while staying cheap to recompute during
//   verification.
fn updated_balance_of(key: &Address) -> Balance {
    let mut value = balance_of(key);
    for (i, b) in value.iter_mut().enumerate() {
        *b ^= 0xa5u8.wrapping_add((i as u8).wrapping_mul(17));
    }
    value
}

// - Use a small, fast deterministic mixer only for demo data generation.
// - Spread nearby integer inputs across visually unrelated bytes, making
//   accidental record/index mixups easier for assertions to catch.
fn splitmix64(z: u64) -> u64 {
    let mut x = z;
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

// - Display the little-endian u256 balance in conventional big-endian hex.
fn u256_hex(le: &Balance) -> String {
    let be: String = le.iter().rev().map(|b| format!("{b:02x}")).collect();
    format!("0x{be}")
}

// - Format a byte count in the largest unit that keeps it readable.
fn bytes(n: usize) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut value = n as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{n} B")
    } else {
        format!("{value:.3} {}", UNITS[unit])
    }
}

// - Return Linux's peak resident set size counter, if this platform exposes it.
// - Return `None` on other platforms, keeping the example portable while still
//   giving useful memory telemetry on the deployment target.
fn peak_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kib: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kib * 1024);
        }
    }
    None
}
