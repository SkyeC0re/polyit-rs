# poly_it

[![maintenance status: passively-maintained](https://img.shields.io/badge/maintenance-passively--maintained-yellowgreen.svg)](https://doc.rust-lang.org/cargo/reference/manifest.html#the-badges-section)
[![license](https://img.shields.io/crates/l/poly_it.svg)](LICENSE)
[![crates.io](https://img.shields.io/crates/v/poly_it.svg)](https://crates.io/crates/poly_it)
[![docs.rs](https://img.shields.io/docsrs/poly_it/latest)](https://docs.rs/poly_it/latest/)
[![rust 1.70.0+ badge](https://img.shields.io/badge/rust-1.70.0+-93450a.svg)](https://doc.rust-lang.org/cargo/reference/manifest.html#the-rust-version-field)
[![Rust CI](https://github.com/SkyeC0re/polyit-rs/actions/workflows/rust-ci.yml/badge.svg)](https://github.com/SkyeC0re/polyit-rs/actions/workflows/rust-ci.yml)
[![codecov](https://codecov.io/gh/SkyeC0re/polyit-rs/branch/master/graph/badge.svg?token=UIj6XoEUBm)](https://codecov.io/gh/SkyeC0re/polyit-rs)

A `no_std` library for manipulating polynomials with slice support and minimal allocation.

[Documentation](https://docs.rs/poly_it/latest/)

## How to use?

Add this to your `Cargo.toml`:

```toml
[dependencies]
poly_it = "0.2.1"
```

## no_std environments

The library is `no_std` by default but assumes a global allocator. It can also be used entirely without one if the default features
are disabled and either of the `tinyvec` or `arrayvec` features are enabled.

## Minimum supported Rust version (MSRV)

The minimum supported Rust version is **Rust 1.56.0**.

## Acknowledgements

This library started out as fork of [polynomial-rs](https://github.com/gifnksm/polynomial-rs),
but was rewritten to accommodate iterators and minimal allocation.