# Changelog

## 0.1.0

- Added release metadata and release-readiness documentation.
- Made portable reference-backend builds the default.
- Added fallible `try_*` APIs around query, response, rebuild, and keyword wire
  operations while keeping existing convenience wrappers.
- Removed hidden local AVX2/FMA rustflags; production AVX builds now pass
  target features explicitly.
