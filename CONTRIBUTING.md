# Contributing

Before opening a PR, run:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --release --lib
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test --release --lib --features avx2-fhe
cargo doc --no-deps
```

Keep public APIs fallible for malformed wire data and shape mismatches. Panic
wrappers may remain for examples and quick experiments, but production-facing
paths should expose `EthPirError`.
