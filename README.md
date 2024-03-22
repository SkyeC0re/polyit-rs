# poly_it

[![maintenance status: passively-maintained](https://img.shields.io/badge/maintenance-passively--maintained-yellowgreen.svg)](https://doc.rust-lang.org/cargo/reference/manifest.html#the-badges-section)
[![license](https://img.shields.io/crates/l/polynomial.svg)](LICENSE)
[![crates.io](https://img.shields.io/crates/v/polynomial.svg)](https://crates.io/crates/poly_it)
[![docs.rs](https://img.shields.io/docsrs/polynomial/latest)](https://docs.rs/poly_it/latest/)
[![rust 1.70.0+ badge](https://img.shields.io/badge/rust-1.70.0+-93450a.svg)](https://doc.rust-lang.org/cargo/reference/manifest.html#the-rust-version-field)
[![Rust CI](https://github.com/SkyeC0re/polyit-rs/actions/workflows/rust-ci.yml/badge.svg)](https://github.com/SkyeC0re/polyit-rs/actions/workflows/rust-ci.yml)
[![codecov](https://codecov.io/gh/SkyeC0re/polyit-rs/branch/master/graph/badge.svg?token=UIj6XoEUBm)](https://codecov.io/gh/SkyeC0re/polyit-rs)

A no-std library for manipulating polynomials with iterator support and minimal allocation.

[Documentation](https://docs.rs/poly_it/latest/)

## How to use?

Add this to your `Cargo.toml`:

```toml
[dependencies]
poly_it = "0.1.0"
```

## no_std environments

The library can be used in a `no_std` environment, so long as a global allocator is present.

## Minimum supported Rust version (MSRV)

The minimum supported Rust version is **Rust 1.70.0**.
At least the last 3 versions of stable Rust are supported at any given time.

While a crate is pre-release status (0.x.x) it may have its MSRV bumped in a patch release.
Once a crate has reached 1.x, any MSRV bump will be accompanied with a new minor version.

## Acknowledgements

This library started out as fork of [polynomial-rs](https://github.com/gifnksm/polynomial-rs),
but was rewritten to accommodate iterators and minimal allocation.