//! End-to-end ETH PIR demo at the real 2 GiB shape.
//!
//! Run from the `eth-pir` repository root:
//!
//! ```sh
//! cargo run --release --example eth_pir
//! ```
//!
//! The default `avx2-fhe` feature uses `FFT64Avx` (AVX2/FMA). This repository
//! includes a local `.cargo/config.toml` that sets `-C target-feature=+avx2,+fma`
//! for x86/x86_64 builds. The equivalent explicit command is:
//!
//! ```sh
//! RUSTFLAGS="-C target-feature=+avx2,+fma" \
//! cargo run --release --example eth_pir
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
//! then queues 1 M balance updates and 50 K inserted addresses. Flush and
//! rebuilt-index publication briefly hold two serving snapshots; expect peak RSS
//! to be roughly double the previous 1 GiB run.
//!
//! What this exercises:
//!
//! - server initialization from deterministic `(address, balance)` data,
//! - client bootstrap from the keyword helper's full directory blob,
//! - a private lookup and address-in-record verification,
//! - 1 M queued balance updates plus 50 K new-key insertions,
//! - validated keyword delta sync before flush,
//! - full snapshot flush,
//! - MPHF rebuild preparation,
//! - rebuilt-index publication and client resync,
//! - peak RSS reporting from `/proc/self/status` on Linux.

use std::collections::HashMap;
use std::time::Instant;

use eth_pir::{Address, Balance, EthPirClient, EthPirError, EthPirServer};

const INITIAL_ADDRESSES: u64 = 16_000_000;
const UPDATED_ADDRESSES: u64 = 1_000_000;
const NEW_ADDRESSES: u64 = 50_000;

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

    // - Step 2: initialize the server snapshot.
    // - Build the keyword MPHF.
    // - Place each 64-byte record at its private-index slot.
    // - Encode the PIR database, generate the query mask, and run
    //   query-independent offline preprocessing.
    // - After this point the server can answer index PIR queries, while clients
    //   can map ETH addresses to PIR indices through the keyword helper.
    let t = Instant::now();
    let mut server = EthPirServer::init(&map).expect("init");
    // - Release the input map after the server copies records into its own
    //   storage, keeping peak memory closer to a production loader after it
    //   publishes a serving snapshot.
    drop(map);
    println!("SERVER init (MPHF+fill+offline)  : {:?}", t.elapsed());

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
    println!("lookup (1 query round trip)      : {:?}", t.elapsed());
    assert_eq!(balance, balance_of(&sample_existing));
    println!("  balance                        : {}", u256_hex(&balance));

    // - Step 5: queue many existing-address updates and many insertions.
    // - Update the first 1M existing token balances.
    // - Append 50K new addresses to the keyword delta overlay.
    // - Leave the serving PIR snapshot unchanged for now.
    // - Queueing lets production services absorb writes without rebuilding the
    //   expensive offline snapshot on every update.
    let sample_new = address_at(INITIAL_ADDRESSES);
    let t = Instant::now();
    for i in 0..UPDATED_ADDRESSES {
        let addr = address_at(i);
        server
            .queue_new_update(addr, updated_balance_of(&addr))
            .expect("queue existing-address update");
    }
    for i in 0..NEW_ADDRESSES {
        let addr = address_at(INITIAL_ADDRESSES + i);
        server
            .queue_new_update(addr, balance_of(&addr))
            .expect("queue new-address insert");
    }
    println!(
        "QUEUE ({UPDATED_ADDRESSES} updates + {NEW_ADDRESSES} inserts): {:?}",
        t.elapsed()
    );

    // - Step 6: synchronize only the keyword delta.
    // - Fetch the suffix of keyword entries the client has not seen yet.
    // - Validate the delta envelope before applying it.
    // - New address-to-index mappings are now known to the client, but their
    //   record bodies are still zero until the server flushes queued updates.
    // - The assertions show that old data remains served for updated addresses
    //   and inserted addresses are not retrievable yet.
    let tail = server.keyword().delta_from(client.delta_len());
    client.apply_delta(&tail).expect("delta sync");
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

    // - Step 7: flush queued writes into a new serving snapshot.
    // - Apply pending balances to plaintext records.
    // - Re-encode the database, rerun offline preprocessing, and atomically swap
    //   the responder snapshot.
    // - Queued writes are not visible to PIR responses until the
    //   encrypted/preprocessed database has been rebuilt.
    // - After the swap, clients using the already-applied keyword delta can
    //   retrieve updated and newly inserted records.
    let t = Instant::now();
    server.flush_queue();
    println!("FLUSH (re-encode + offline)      : {:?}", t.elapsed());
    assert_eq!(
        lookup(&server, &mut client, sample_existing),
        Ok(updated_balance_of(&sample_existing))
    );
    assert_eq!(
        lookup(&server, &mut client, sample_new),
        Ok(balance_of(&sample_new))
    );
    println!("  post-flush: sample update + sample insert verify");

    // - Step 8: prepare a rebuilt keyword index.
    // - Construct a fresh MPHF over the complete key set.
    // - Permute records into the new MPHF order.
    // - Keep the live keyword directory and serving snapshot unchanged.
    // - Compacting the delta overlay back into the MPHF keeps lookup state small
    //   and lookup behavior predictable.
    let t = Instant::now();
    let prepared = server.prepare_index_rebuild().expect("prepare rebuild");
    println!("REBUILD INDEX (MPHF+permute)     : {:?}", t.elapsed());
    assert_eq!(server.keyword().version(), 0);

    // - Step 9: publish the rebuilt index.
    // - Re-encode the database in the prepared MPHF order.
    // - Rerun offline preprocessing and atomically swap the responder snapshot.
    // - Publish the directory version only after the PIR snapshot matches it.
    // - This phase is required for correctness: clients query PIR by index, so
    //   a new MPHF cannot be exposed while the server still serves the old
    //   physical layout.
    let t = Instant::now();
    server
        .publish_index_rebuild(prepared)
        .expect("publish rebuild");
    println!("PUBLISH INDEX (re-encode+offline): {:?}", t.elapsed());
    assert_eq!(server.keyword().version(), 1);

    // - Step 10: resynchronize the client after an MPHF rebuild.
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

    // - Step 11: report peak resident memory when running on Linux.
    // - Read `/proc/self/status` and print VmHWM if available.
    // - This example intentionally exercises a large real-shape database.
    // - Peak memory is one of the first production capacity signals to check
    //   when changing backend, BLAS, NUMA, or snapshotting behavior.
    if let Some(peak) = peak_rss_bytes() {
        println!(
            "PEAK MEMORY (VmHWM)              : {:.3} GiB",
            peak as f64 / (1u64 << 30) as f64
        );
    }
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
