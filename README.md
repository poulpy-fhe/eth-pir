# eth-pir

ETH token-balance PIR service layer based on `poulpy-pir`.

This crate depends on the adjacent `../poulpy-pir` crate and packages its
index-based PIR core plus keyword directory into a fixed-shape service:

- construction: InsPIRe2 recursion,
- default backend feature: `avx2-fhe` using `FFT64Avx` (AVX2/FMA),
- optional backend feature: `avx512-fhe` using `FFT64Avx512`,
- optional GEMM feature: `cblas-gemm` for system CBLAS/OpenBLAS offline
  preprocessing,
- optional NUMA feature: `numa-db-interleave` forwards to `poulpy-pir` database
  allocation,
- preset geometry: `InsPIRe2-g32-2GiB-c65536`,
- payload: one `U512P65536` slot per record,
- capacity: 33,554,432 addresses,
- record: 20-byte address, 12 bytes of zero padding, 32-byte little-endian u256
  token balance.

## Run

The default `avx2-fhe` feature uses `FFT64Avx` (AVX2/FMA). This repository
includes a local `.cargo/config.toml` that passes the required `+avx2,+fma`
target features on x86/x86_64, so the short command works from this repository
root:

```sh
cargo run --release --example eth_pir
```

The equivalent explicit command is:

```sh
RUSTFLAGS="-C target-feature=+avx2,+fma" \
cargo run --release --example eth_pir
```

On an AVX-512F host:

```sh
RUSTFLAGS="-C target-feature=+avx2,+fma,+avx512f" \
cargo run --release --no-default-features --features avx512-fhe --example eth_pir
```

With system CBLAS/OpenBLAS enabled:

```sh
RUSTFLAGS="-C target-feature=+avx2,+fma,+avx512f" \
cargo run --release --no-default-features --features "avx512-fhe,cblas-gemm" \
  --example eth_pir
```

For single-query latency on multi-socket Linux hosts, add NUMA interleaving:

```sh
RUSTFLAGS="-C target-feature=+avx2,+fma,+avx512f" \
cargo run --release --no-default-features \
  --features "avx512-fhe,cblas-gemm,numa-db-interleave" \
  --example eth_pir
```

Install the optimized pthread OpenBLAS development package before enabling
`cblas-gemm`:

```sh
sudo apt-get update
sudo apt-get install -y libopenblas-pthread-dev
```

On Debian/Ubuntu this provides `/usr/lib/x86_64-linux-gnu/openblas-pthread` and
`libopenblas.so`. Without it, `cblas-gemm` fails during setup because the linker
cannot find `-lopenblas`. The example re-execs once with
`OPENBLAS_NUM_THREADS=1` when the variable is absent, because OpenBLAS sizes its
worker pool before `main`.

`poulpy-pir` exposes `CBLAS_LIB_DIR` and `CBLAS_LIB_NAME` for deliberate custom
BLAS deployments, but this example expects an optimized `cblas_dgemm`; generic
CBLAS providers such as GSL CBLAS are not the intended performance path.

Leave `numa-db-interleave` off for batched serving; `poulpy-pir`'s measured
tuning currently treats interleaving as the single-query-latency choice.

## Sync Contract

Cold start and post-rebuild resync use `KeywordWire::full()` and
`EthPirClient::{new,resync}`.

Incremental sync uses `KeywordWire::delta_from(client.delta_len())` and
`EthPirClient::apply_delta`. The delta is a validated envelope:

```text
magic             8 bytes: "PIRDLT1\0"
directory_version u64 little-endian
base_mphf_len     u64 little-endian
delta_start       u64 little-endian
delta_count       u64 little-endian
capacity          u64 little-endian
keys              delta_count 20-byte addresses
```

The client rejects stale versions, wrong offsets, mismatched base lengths,
capacity mismatches, over-capacity payloads, invalid magic, and duplicate keys
within the delta overlay.

## Index Rebuilds

`EthPirServer::prepare_index_rebuild()` rebuilds the MPHF and permutes the
plaintext record vector without publishing anything. The live keyword helper and
serving PIR snapshot remain on the old version, so this phase can be measured
separately.

`EthPirServer::publish_index_rebuild(prepared)` installs the prepared directory
and rebuilds the matching PIR snapshot. This database rebuild is required before
clients resync: a new MPHF changes the indices clients query, and those indices
must match the physical database layout.

`EthPirServer::rebuild_index()` remains the one-shot safe API and performs both
phases.

## Repository Layout

- `src/lib.rs`: shared types, default shape, errors.
- `src/server.rs`: `EthPirServer`, `EthPirResponder`, `KeywordWire`.
- `src/client.rs`: `EthPirClient`, lookup state, decrypt verification.
- `examples/eth_pir.rs`: 16 M initial-address demo on the 2 GiB shape, with 1 M
  balance updates and 50 K inserted addresses.
