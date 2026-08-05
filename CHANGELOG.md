# Changelog

## 0.1.1

- Added the Apache 2.0 `LICENSE` file, which was already included it in the published package,
  and added a license section to the README for more explicite licensing.

## 0.1.0

- Added release metadata and release-readiness documentation.
- Made portable reference-backend builds the default.
- Added fallible `try_*` APIs around query, response, rebuild, and keyword wire
  operations while keeping existing convenience wrappers.
- Removed hidden local AVX2/FMA rustflags; production AVX builds now pass
  target features explicitly.
