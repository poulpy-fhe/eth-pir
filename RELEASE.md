# Release Checklist

1. Publish the matching `poulpy-pir` version to crates.io first. `eth-pir`
   depends on `poulpy-pir = "0.1.0"`, with a local path used only for adjacent
   repository development.
2. Update `CHANGELOG.md` and the crate version in `Cargo.toml`.
3. Run portable checks:

   ```sh
   cargo fmt --check
   cargo clippy --all-targets -- -D warnings
   cargo test --release --lib
   cargo doc --no-deps
   ```

4. Run the production AVX2 check:

   ```sh
   RUSTFLAGS="-C target-feature=+avx2,+fma" \
   cargo test --release --lib --features avx2-fhe
   ```

5. Package the crate:

   ```sh
   cargo package --allow-dirty
   ```

6. Tag and publish only after the package contents and generated documentation
   match the intended release.
