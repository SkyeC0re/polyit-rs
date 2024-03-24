//! # poly_it
//! A no-std library for manipulating polynomials with slice support and minimal allocation.
//!
//! At the end of the day the classical representation method for the polynomial is
//! its coefficients and this library leverages this by means of slices.  
//! Take this example case:
//! ```
//! extern crate poly_it;
//! use poly_it::Polynomial;
//!
//! pub fn main() {
//!     let p1 = Polynomial::new(vec![1, 2, 3]);
//!     let p2 = Polynomial::new(vec![3, 2, 1]);
//!     let arr = [1, 2];
//!     let vec = vec![1, 2, 3];
//!
//!     assert_eq!(
//!         format!("{}", p2 - &p1),
//!         "2 + -2x^2"
//!     );
//!
//!     assert_eq!(
//!         format!("{}", p1.clone() + arr.as_slice()),
//!         "2 + 4x + 3x^2"
//!     );
//!
//!     assert_eq!(
//!         format!("{}", p1 * vec.as_slice()),
//!         "1 + 4x + 10x^2 + 12x^3 + 9x^4"
//!     );
//! }
//! ```
//!
//! This example illustrates several things:
//! 1. Binary operations with slices.
//! 2. Using operations with owned or referenced polynomials.
//! 3. That `+ -` is possibly heretical.
//!
//! As to the third point, this has to do with future proofing in cases where the underlying numerical type
//! might not have ordering, but for the second point, see below for a quick summary as to the how and the why.
//!
//! ## Minimal Allocation
//!
//! This crate attempts to minimize allocations by reusing the underlying allocated space of the polynomials when an
//! owned [Polynomial](struct@Polynomial) is passed to a unary or binary operation. If more than one owned polynomial is passed,
//! as would be the case with:
//! ```text
//! let x = Polynomial::new(vec![1, 2, 3]) * Polynomial::new(vec![3, 2, 1]);
//! ```
//! the polynomial whose data vector has the highest capacity will be selected. If a new allocation is desired, use
//! references.
#![no_std]
#![warn(bad_style)]
#![warn(missing_docs)]
#![warn(trivial_casts)]
#![warn(trivial_numeric_casts)]
#![warn(unused)]
#![warn(unused_extern_crates)]
#![warn(unused_import_braces)]
#![warn(unused_qualifications)]
#![warn(unused_results)]
#![forbid(unsafe_code)]

extern crate alloc;
use alloc::{vec, vec::Vec};
use core::fmt::Display;
use core::marker::PhantomData;
use core::ops::{Add, Div, Mul, Neg, Sub};
use core::{fmt, mem};
use num_traits::{Float, FloatConst, FromPrimitive, One, Zero};

pub use num_traits;

/// Represents a slice like storage type.
pub trait Storage<T>: Clone {
    /// Create a new storage instance.
    fn new() -> Self;

    /// Create a new storage instance with at least `capacity` capacity.
    /// Is allowed to panic if the storage type cannot support the requested capacity.
    fn with_capacity(capacity: usize) -> Self;

    /// Clears all data in the storage.
    fn clear(&mut self);

    /// Push an element to storage.
    fn push(&mut self, value: T);

    /// Pop the last element from storage.
    fn pop(&mut self) -> Option<T>;

    /// Return an immutable refrence to the data.
    fn as_slice(&self) -> &[T];

    /// Return a mutable reference to the data.
    fn as_mut_slice(&mut self) -> &mut [T];

    /// Return the length of the data. Should always concide with the
    /// the slice's length.
    #[inline]
    fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// Return the capacity of the storage type.
    fn capacity(&self) -> usize;
}

impl<T> Storage<T> for Vec<T>
where
    T: Clone,
{
    #[inline]
    fn new() -> Self {
        Self::new()
    }

    #[inline]
    fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity(capacity)
    }

    #[inline]
    fn clear(&mut self) {
        self.clear();
    }

    #[inline]
    fn push(&mut self, value: T) {
        self.push(value);
    }

    #[inline]
    fn pop(&mut self) -> Option<T> {
        self.pop()
    }

    #[inline]
    fn len(&self) -> usize {
        self.len()
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.capacity()
    }

    fn as_slice(&self) -> &[T] {
        self.as_slice()
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

/// A polynomial.
#[derive(Eq, PartialEq, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Polynomial<T, S: Storage<T> = Vec<T>> {
    data: S,
    _type: PhantomData<T>,
}

impl<T: Zero, S: Storage<T>> Polynomial<T, S> {
    /// Creates a new `Polynomial` from a `Vec` of coefficients. Automatically
    /// trims all trailing zeroes from the data.
    ///
    /// # Examples
    ///
    /// ```
    /// use poly_it::Polynomial;
    /// let poly = Polynomial::new(vec![1., 2., 3., 0.]);
    /// assert_eq!("1.00 + 2.00x + 3.00x^2", format!("{:.2}", poly));
    /// ```
    #[inline]
    pub fn new(data: S) -> Self {
        let mut p = Self {
            data,
            _type: PhantomData,
        };
        p.trim();
        p
    }

    /// Trims all trailing zeros from the polynomial's coefficients.
    /// # Examples
    ///
    /// ```
    /// use poly_it::Polynomial;
    /// let mut poly = Polynomial::new(vec![1, 0, -1]) + Polynomial::new(vec![0, 0, 1]);
    /// poly.trim();
    /// assert_eq!(poly.coeffs(), &[1]);
    /// ```
    #[inline]
    pub fn trim(&mut self) {
        while let Some(true) = self.data.as_slice().last().map(T::is_zero) {
            let _ = self.data.pop();
        }
    }

    /// Get an immutable reference to the polynomial's coefficients.
    #[inline]
    pub fn coeffs(&self) -> &[T] {
        self.data.as_slice()
    }

    /// Get a mutable reference to the polynomial's coefficients.
    #[inline]
    pub fn coeffs_mut(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }
}

// impl<T, S> Into<S> for Polynomial<T, S>
// where
//     S: Storage<T>,
// {
//     #[inline]
//     fn into(self) -> S {
//         self.data
//     }
// }

impl<T, S> Display for Polynomial<T, S>
where
    T: Display + Zero,
    S: Storage<T>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut previous_nonzero_term = false;
        for (i, ci) in self.data.as_slice().into_iter().enumerate() {
            if ci.is_zero() {
                continue;
            }
            if i == 0 {
                ci.fmt(f)?;
            } else {
                if previous_nonzero_term {
                    f.write_str(" + ")?;
                }
                ci.fmt(f)?;
                f.write_str("x")?;
                if i > 1 {
                    f.write_fmt(format_args!("^{}", i))?;
                }
            }
            previous_nonzero_term = true;
        }
        if !previous_nonzero_term {
            T::zero().fmt(f)?;
        }

        Ok(())
    }
}

impl<T, S> Polynomial<T, S>
where
    T: Zero
        + One
        + Add<T, Output = T>
        + Neg<Output = T>
        + Sub<T, Output = T>
        + Mul<T, Output = T>
        + Div<T, Output = T>
        + Clone,
    S: Storage<T> + Clone,
{
    /// Generates the polynomial $P(x)$ that fits a number of points `N` in a least squares sense which minimizes:
    /// $$\sum_{i=1}^N\left[P(x_i) - y_i\right]^2$$
    ///
    /// Returns `None` if a unique solution does not exist (such as if `deg < N + 1`).
    ///
    /// Based on:
    /// [P. A. Gorry, General least-squares smoothing and differentiation by the convolution (Savitzky-Golay) method, Anal. Chem., vol. 62, no. 6, pp. 570-573, Mar. 1990.](https://pubs.acs.org/doi/10.1021/ac00205a007)
    ///
    /// # Examples
    ///
    /// ```
    /// use poly_it::Polynomial;
    /// // Noisy second degree polynomial data
    /// let xys: Vec<(f64, f64)> = (0..100).into_iter().map(|x| {
    ///     let xf = x as f64;
    ///     (xf, 1. + 2. * xf + 3. * xf * xf + xf.sin())
    /// }).collect();
    /// let poly = Polynomial::<f64>::least_squares_fit(
    ///     2,
    ///     xys.iter().copied()
    ///  ).unwrap();
    /// println!("{:.3}", poly);
    /// ```
    #[inline(always)]
    pub fn least_squares_fit(
        deg: usize,
        samples: impl Iterator<Item = (T, T)> + Clone,
    ) -> Option<Self> {
        Self::least_squares_fit_weighted(deg, samples.map(|(x, y)| (x, y, T::one())))
    }

    /// Adapted from [least_squares_fit](fn@Polynomial::least_squares_fit) to allow for non-uniform fitting weights.
    /// Generates the polynomial $P(x)$ that fits a number of points `N` in a least squares sense which minimizes:
    /// $$\sum_{i=1}^N w_i \left[P(x_i) - y_i\right]^2$$
    /// when $W = \sum_{i=1}^N w_i$ is positive. If $W$ is negative it will instead maximize the above value.
    /// No solution exists if $W = 0$.
    ///
    /// Returns `None` if a unique solution does not exist (such as if `deg < N + 1`).
    /// # Examples
    ///
    /// ```
    /// use poly_it::Polynomial;
    /// // Noisy second degree polynomial data weighted inversly by its deviation
    /// let xyws: Vec<(f64, f64, f64)> = (0..100).map(|x| {
    ///     let xf = x as f64;
    ///     return (xf, 1. + 2. * xf + 3. * xf * xf + xf.sin(), 1.0 - xf.sin().abs())
    /// }).collect();
    /// let poly = Polynomial::<f64>::least_squares_fit_weighted(
    ///     2,
    ///     xyws.iter().copied()
    ///     ).unwrap();
    /// println!("{:.3}", poly);
    /// ```
    pub fn least_squares_fit_weighted(
        deg: usize,
        samples: impl Iterator<Item = (T, T, T)> + Clone,
    ) -> Option<Self> {
        let (mut d_0, gamma_0, mut b_0, data_len) = samples.clone().fold(
            (T::zero(), T::zero(), T::zero(), 0usize),
            |mut acc, (xi, yi, wi)| {
                acc.0 = acc.0 + wi.clone() * yi;
                acc.1 = acc.1 + wi.clone();
                acc.2 = acc.2 + wi * xi;
                acc.3 += 1;
                acc
            },
        );

        if data_len < deg + 1 || gamma_0.is_zero() {
            return None;
        }

        b_0 = b_0 / gamma_0.clone();
        d_0 = d_0 / gamma_0.clone();

        let mut p_data = S::with_capacity(deg + 1);

        if deg == 0 {
            p_data.push(d_0);
            return Some(Polynomial::new(p_data));
        }

        for _ in 0..(deg + 1) {
            p_data.push(T::zero());
        }
        let mut p_km1 = p_data.clone();
        let mut p_km1 = p_km1.as_mut_slice();
        let mut p_k = p_data.clone();
        let mut p_k = p_k.as_mut_slice();
        let p_data_slice = p_data.as_mut_slice();
        p_data_slice[0] = d_0;
        p_k[0] = T::one();

        let mut gamma_k = gamma_0;
        let mut b_k = b_0;
        let mut minus_c_k = T::zero();
        let mut kp1 = 1;

        loop {
            // Overwrite p_{k-1} with p_{k+1}
            for (p_km1i, p_ki) in p_km1[0..kp1].iter_mut().zip(p_k[0..kp1].iter().cloned()) {
                *p_km1i = minus_c_k.clone() * p_km1i.clone() - b_k.clone() * p_ki;
            }
            for (p_km1i, p_kim1) in p_km1[1..(kp1 + 1)]
                .iter_mut()
                .zip(p_k[0..kp1].iter().cloned())
            {
                *p_km1i = p_km1i.clone() + p_kim1;
            }

            let (mut d_kp1, gamma_kp1, mut b_kp1) = samples.clone().fold(
                (T::zero(), T::zero(), T::zero()),
                |mut acc, (xi, yi, wi)| {
                    let px = eval_slice_horner(&p_km1[0..(kp1 + 1)], xi.clone());
                    let wipx = wi * px.clone();
                    acc.0 = acc.0 + yi * wipx.clone();

                    let wipxpx = wipx * px;
                    acc.1 = acc.1 + wipxpx.clone();
                    acc.2 = acc.2 + xi * wipxpx;

                    acc
                },
            );

            if gamma_kp1.is_zero() {
                return None;
            }

            d_kp1 = d_kp1 / gamma_kp1.clone();

            for (pi, p_km1i) in p_data_slice
                .iter_mut()
                .zip(p_km1[0..(kp1 + 1)].iter().cloned())
            {
                *pi = pi.clone() + d_kp1.clone() * p_km1i;
            }

            if kp1 == deg {
                break;
            }

            // Delay the remaining work left for b_{k+1} until it is certain to be needed.
            b_kp1 = b_kp1 / gamma_kp1.clone();

            kp1 += 1;
            b_k = b_kp1;
            minus_c_k = -(gamma_kp1.clone() / gamma_k);
            gamma_k = gamma_kp1;

            // Reorient the offsets
            mem::swap(&mut p_k, &mut p_km1);
        }
        let mut p = Polynomial::new(p_data);
        p.trim();
        Some(p)
    }

    /// Creates the [Lagrange polynomial] that fits a number of points.
    ///
    /// [Lagrange polynomial]: https://en.wikipedia.org/wiki/Lagrange_polynomial
    ///
    /// Returns `None` if any two x-coordinates are the same.
    ///
    /// # Examples
    ///
    /// ```
    /// use poly_it::Polynomial;
    /// let poly = Polynomial::<f64>::lagrange(
    ///     [1., 2., 3.].iter()
    ///     .copied()
    ///     .zip([10., 40., 90.].iter().copied())
    /// ).unwrap();
    /// assert_eq!("10.0x^2", format!("{:.1}",poly));
    /// ```
    pub fn lagrange(samples: impl Iterator<Item = (T, T)> + Clone) -> Option<Self> {
        let mut res = Polynomial::new(S::new());
        let mut li = Polynomial::new(S::new());
        for (i, (x, y)) in samples.clone().enumerate() {
            li.data.clear();
            li.data.push(T::one());
            let mut denom = T::one();
            for (j, (x2, _)) in samples.clone().enumerate() {
                if i != j {
                    li = li * [-x2.clone(), T::one()].as_slice();
                    let diff = x.clone() - x2.clone();
                    if diff.is_zero() {
                        return None;
                    }
                    denom = denom * diff;
                }
            }
            let scalar = y.clone() / denom;
            li = li * [scalar].as_slice();
            res = res + &li;
        }
        res.trim();
        Some(res)
    }
}

impl<T, S> Polynomial<T, S>
where
    T: Zero + One + FloatConst + Float + FromPrimitive,
    S: Storage<T>,
{
    /// [Chebyshev approximation] fits a function $f(x)$ over to a polynomial by taking $n-1$
    /// samples of $f$ on some interval $[a, b]$.
    ///
    /// [Chebyshev approximation]: https://en.wikipedia.org/wiki/Approximation_theory#Chebyshev_approximation
    ///
    /// This attempts to minimize the maximum error.
    ///
    /// Returns `None` if $n < 1$ or if the Gauss–Chebyshev zeros collapse in $[a, b]$ due to
    /// floating point inaccuracies.
    ///
    /// # Examples
    ///
    /// ```
    /// use poly_it::Polynomial;
    /// use std::f64::consts::PI;
    /// let p = Polynomial::<f64>::chebyshev(&f64::sin, 7, PI/4., -PI/4.).unwrap();
    /// assert!((p.eval(0.) - (0.0_f64).sin()).abs() < 0.0001);
    /// assert!((p.eval(0.1) - (0.1_f64).sin()).abs() < 0.0001);
    /// assert!((p.eval(-0.1) - (-0.1_f64).sin()).abs() < 0.0001);
    /// ```
    #[inline]
    pub fn chebyshev<F: Fn(T) -> T>(f: &F, n: usize, a: T, b: T) -> Option<Self> {
        if n < 1 {
            return None;
        }

        let mut samples = Vec::with_capacity(n);

        let pi_over_n = T::PI() / T::from(n)?;
        let two = T::one() + T::one();
        let half = T::one() / two;
        let x_avg = (b + a) * half;
        let x_half_delta = (b - a).abs() * half;

        for k in 0..n {
            let x = x_avg - x_half_delta * T::cos((T::from(k)? + half) * pi_over_n);
            samples.push((x, f(x)));
        }

        Polynomial::lagrange(samples.iter().copied())
    }
}

#[inline]
fn eval_slice_horner<T>(slice: &[T], x: T) -> T
where
    T: Zero + Mul<Output = T> + Clone,
{
    let mut result = T::zero();
    for ci in slice.into_iter().rev().cloned() {
        result = result * x.clone() + ci.clone();
    }

    result
}

impl<T, S> Polynomial<T, S>
where
    T: Zero + Mul<Output = T> + Clone,
    S: Storage<T>,
{
    /// Evaluates the polynomial at a point.
    ///
    /// # Examples
    ///
    /// ```
    /// use poly_it::Polynomial;
    /// let poly = Polynomial::new(vec![1, 2, 3]);
    /// assert_eq!(1, poly.eval(0));
    /// assert_eq!(6, poly.eval(1));
    /// assert_eq!(17, poly.eval(2));
    /// ```
    #[inline(always)]
    pub fn eval(&self, x: T) -> T {
        eval_slice_horner(self.data.as_slice(), x)
    }
}

impl<T> Neg for Polynomial<T>
where
    T: Neg<Output = T> + Clone,
{
    type Output = Polynomial<T>;

    #[inline]
    fn neg(mut self) -> Self::Output {
        self.data.iter_mut().for_each(|c| *c = -c.clone());
        self
    }
}

impl<'a, T> Neg for &'a Polynomial<T>
where
    T: Neg<Output = T> + Zero + Clone,
{
    type Output = Polynomial<T>;

    #[inline]
    fn neg(self) -> Self::Output {
        -self.clone()
    }
}

macro_rules! forward_ref_iter_binop {
    (impl $imp:ident, $method:ident) => {
        impl<'a, T, S> $imp<&'a [T]> for &'a Polynomial<T, S>
        where
            T: 'a + $imp<T, Output = T> + Zero + Clone,
            S: Storage<T>,
        {
            type Output = Polynomial<T, S>;

            #[inline(always)]
            fn $method(self, other: &[T]) -> Self::Output {
                $imp::$method(self.clone(), other)
            }
        }
    };
}

macro_rules! forward_iter_ref_binop {
    (impl $imp:ident, $method:ident) => {
        impl<'a, T, S> $imp<&'a Polynomial<T, S>> for &'a [T]
        where
            T: $imp<T, Output = T> + Zero + Clone,
            S: Storage<T>,
        {
            type Output = Polynomial<T, S>;

            #[inline(always)]
            fn $method(self, other: &'a Polynomial<T, S>) -> Self::Output {
                $imp::$method(self, other.clone())
            }
        }
    };
}

macro_rules! forward_iter_val_val_binop {
    (impl $imp:ident, $method:ident) => {
        impl<T, S> $imp<Polynomial<T, S>> for Polynomial<T, S>
        where
            T: $imp<T, Output = T> + Zero + Clone,
            S: Storage<T>,
        {
            type Output = Polynomial<T, S>;

            #[inline(always)]
            fn $method(self, other: Polynomial<T, S>) -> Self::Output {
                if self.data.capacity() >= other.data.capacity() {
                    $imp::$method(self, other.data.as_slice())
                } else {
                    $imp::$method(self.data.as_slice(), other)
                }
            }
        }
    };
}

macro_rules! forward_iter_ref_val_binop {
    (impl $imp:ident, $method:ident) => {
        impl<'a, T, S> $imp<Polynomial<T, S>> for &'a Polynomial<T, S>
        where
            T: $imp<T, Output = T> + Zero + Clone,
            S: Storage<T>,
        {
            type Output = Polynomial<T, S>;

            #[inline(always)]
            fn $method(self, other: Polynomial<T, S>) -> Self::Output {
                $imp::$method(self.data.as_slice(), other)
            }
        }
    };
}

macro_rules! forward_iter_val_ref_binop {
    (impl $imp:ident, $method:ident) => {
        impl<'a, T, S> $imp<&'a Polynomial<T, S>> for Polynomial<T, S>
        where
            T: $imp<T, Output = T> + Zero + Clone,
            S: Storage<T>,
        {
            type Output = Polynomial<T, S>;

            #[inline(always)]
            fn $method(self, other: &'a Polynomial<T, S>) -> Self::Output {
                $imp::$method(self, other.data.as_slice())
            }
        }
    };
}

macro_rules! forward_iter_ref_ref_binop {
    (impl $imp:ident, $method:ident) => {
        impl<'a, 'b, T, S> $imp<&'b Polynomial<T, S>> for &'a Polynomial<T, S>
        where
            T: $imp<T, Output = T> + Zero + Clone,
            S: Storage<T>,
        {
            type Output = Polynomial<T, S>;

            #[inline(always)]
            fn $method(self, other: &'b Polynomial<T, S>) -> Self::Output {
                if self.data.len() >= other.data.len() {
                    $imp::$method(self, other.data.as_slice())
                } else {
                    $imp::$method(self.data.as_slice(), other)
                }
            }
        }
    };
}

macro_rules! forward_iter_all_binop {
    (impl $imp:ident, $method:ident) => {
        forward_ref_iter_binop!(impl $imp, $method);
        forward_iter_ref_binop!(impl $imp, $method);
        forward_iter_val_val_binop!(impl $imp, $method);
        forward_iter_ref_val_binop!(impl $imp, $method);
        forward_iter_val_ref_binop!(impl $imp, $method);
        forward_iter_ref_ref_binop!(impl $imp, $method);
    };
}

forward_iter_all_binop!(impl Add, add);
forward_iter_all_binop!(impl Sub, sub);
forward_iter_all_binop!(impl Mul, mul);

impl<'a, T, S> Add<&'a [T]> for Polynomial<T, S>
where
    T: Zero + Add<T, Output = T> + Clone,
    S: Storage<T>,
{
    type Output = Polynomial<T, S>;

    fn add(mut self, other: &'a [T]) -> Self::Output {
        let p_data = &mut self.data;
        for (pi, si) in p_data
            .as_mut_slice()
            .into_iter()
            .zip(other.into_iter().cloned())
        {
            *pi = pi.clone() + si;
        }
        if other.len() > p_data.len() {
            for si in other[p_data.len()..].iter().cloned() {
                p_data.push(si);
            }
        }
        self.trim();
        self
    }
}

impl<'a, T, S> Add<Polynomial<T, S>> for &'a [T]
where
    T: Zero + Add<T, Output = T> + Clone,
    S: Storage<T>,
{
    type Output = Polynomial<T, S>;
    #[inline(always)]
    fn add(self, other: Polynomial<T, S>) -> Self::Output {
        other + self
    }
}

impl<'a, T, S> Sub<&'a [T]> for Polynomial<T, S>
where
    T: Zero + Sub<T, Output = T> + Clone,
    S: Storage<T>,
{
    type Output = Polynomial<T, S>;

    fn sub(mut self, other: &'a [T]) -> Self::Output {
        let p_data = &mut self.data;
        for (pi, si) in p_data
            .as_mut_slice()
            .into_iter()
            .zip(other.into_iter().cloned())
        {
            *pi = pi.clone() - si;
        }
        if other.len() > p_data.len() {
            for si in other[p_data.len()..].iter().cloned() {
                p_data.push(T::zero() - si);
            }
        }
        self.trim();
        self
    }
}

impl<'a, T, S> Sub<Polynomial<T, S>> for &'a [T]
where
    T: Zero + Sub<T, Output = T> + Clone,
    S: Storage<T>,
{
    type Output = Polynomial<T, S>;

    fn sub(self, mut other: Polynomial<T, S>) -> Self::Output {
        let p_data = &mut other.data;
        for (pi, si) in p_data
            .as_mut_slice()
            .into_iter()
            .zip(self.into_iter().cloned())
        {
            *pi = si - pi.clone();
        }
        if self.len() > p_data.len() {
            for si in self[p_data.len()..].iter().cloned() {
                p_data.push(si);
            }
        } else {
            for pi in &mut p_data.as_mut_slice()[self.len()..] {
                *pi = T::zero() - pi.clone();
            }
        }
        other.trim();
        other
    }
}

impl<'a, T, S> Mul<&'a [T]> for Polynomial<T, S>
where
    T: Zero + Mul<T, Output = T> + Clone,
    S: Storage<T>,
{
    type Output = Polynomial<T, S>;

    fn mul(mut self, other: &'a [T]) -> Self::Output {
        let data_slice = self.data.as_mut_slice();
        let mut ai = match data_slice.last() {
            Some(v) => v.clone(),
            None => return self,
        };
        let last_index = data_slice.len() - 1;
        let b0 = match other.get(0) {
            Some(v) => v.clone(),
            None => {
                self.data.clear();
                return self;
            }
        };
        let other = &other[1..];

        data_slice[last_index] = ai.clone() * b0.clone();
        let _ = data_slice;
        other
            .into_iter()
            .cloned()
            .for_each(|bj| self.data.push(ai.clone() * bj));

        let data_slice = self.data.as_mut_slice();
        for i in (0..last_index).rev() {
            ai = data_slice[i].clone();
            data_slice[i] = ai.clone() * b0.clone();
            data_slice[(i + 1)..]
                .iter_mut()
                .zip(other.into_iter().cloned())
                .for_each(|(v, bj)| *v = v.clone() + ai.clone() * bj.clone());
        }
        self.trim();
        self
    }
}

impl<'a, T, S> Mul<Polynomial<T, S>> for &'a [T]
where
    T: Zero + Mul<T, Output = T> + Clone,
    S: Storage<T>,
{
    type Output = Polynomial<T, S>;
    #[inline]
    fn mul(self, other: Polynomial<T, S>) -> Self::Output {
        other * self
    }
}

impl<T: Zero + Clone> Zero for Polynomial<T> {
    #[inline]
    fn zero() -> Self {
        Self {
            data: vec![],
            _type: PhantomData,
        }
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.data.is_empty()
    }
}

impl<T: Zero + One + Clone> One for Polynomial<T> {
    #[inline]
    fn one() -> Self {
        Self {
            data: vec![One::one()],
            _type: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use core::ops::{Add, Mul, Sub};

    use super::Polynomial;
    use alloc::{vec, vec::Vec};

    #[test]
    fn new() {
        fn check(dst: Vec<i32>, src: Vec<i32>) {
            assert_eq!(dst, Polynomial::new(src).data);
        }
        check(vec![1, 2, 3], vec![1, 2, 3]);
        check(vec![1, 2, 3], vec![1, 2, 3, 0, 0]);
        check(vec![], vec![0, 0, 0]);
    }

    macro_rules! test_binop {
        (impl $imp:ident, $method:ident, $a:expr, $b:expr, $res:expr) => {
            let a = Polynomial::new(($a).into_iter().collect::<Vec<_>>());
            let b = Polynomial::new(($b).into_iter().collect::<Vec<_>>());
            let res = Polynomial::new(($res).into_iter().collect::<Vec<_>>());
            assert_eq!($imp::$method(a.clone(), b.clone()), res);
            assert_eq!($imp::$method(a.clone(), &b), res);
            assert_eq!($imp::$method(&a, b.clone()), res);
            assert_eq!($imp::$method(&a, &b), res);
            assert_eq!($imp::$method(a.clone(), b.coeffs()), res);
            assert_eq!($imp::$method(a.coeffs(), b.clone()), res);
            assert_eq!($imp::$method(&a, b.coeffs()), res);
            assert_eq!($imp::$method(a.coeffs(), &b), res);
        };
    }

    #[test]
    fn neg() {
        let p = Polynomial::new([1, 2, 3].to_vec());
        let p_neg = Polynomial::new([-1, -2, -3].to_vec());
        assert_eq!(-(&p), p_neg);
        assert_eq!(-p.clone(), p_neg);
        assert_eq!(-(-p.clone()), p);
    }

    #[test]
    fn add() {
        let empty: [i32; 0] = [];
        test_binop!(impl Add, add, empty, empty, empty);
        test_binop!(impl Add, add, empty, [1], [1]);
        test_binop!(impl Add, add, [1], empty, [1]);
        test_binop!(impl Add, add, [1, 2, 3], [-1, -2, -3], empty);
        test_binop!(impl Add, add, [0, 2, 4], [1, 3, 5], [1, 5, 9]);
        test_binop!(impl Add, add, [1, 2, 3], [3, 2, 1], [4, 4, 4]);
    }

    #[test]
    fn sub() {
        let empty: [i32; 0] = [];
        test_binop!(impl Sub, sub, empty, empty, empty);
        test_binop!(impl Sub, sub, empty, [1], [-1]);
        test_binop!(impl Sub, sub, [1], empty, [1]);
        test_binop!(impl Sub, sub, [1, 2, 3], [1, 2, 3], empty);
        test_binop!(impl Sub, sub, [0, 2, 4], [1, 3, 5], [-1, -1, -1]);
        test_binop!(impl Sub, sub, [1, 2, 3], [3, 2, 1], [-2, 0, 2]);
    }

    #[test]
    fn mul() {
        let empty: [i32; 0] = [];
        test_binop!(impl Mul, mul, empty, empty, empty);
        test_binop!(impl Mul, mul, empty, [1], empty);
        test_binop!(impl Mul, mul, [1], empty, empty);
        test_binop!(impl Mul, mul, [0], [1, 2], empty);
        test_binop!(impl Mul, mul, [1, 2], [0], empty);
        test_binop!(impl Mul, mul, [1], [1], [1]);
        test_binop!(impl Mul, mul, [1, -3], [1, -3], [1, -6, 9]);
        test_binop!(impl Mul, mul, [1, 1], [1, 0, 1], [1, 1, 1, 1]);
        test_binop!(impl Mul, mul, [0, 0, 1], [0, -1], [0, 0, 0, -1]);
    }

    #[test]
    fn eval() {
        fn check<F: Fn(i32) -> i32>(pol: &[i32], f: F) {
            for n in -10..10 {
                assert_eq!(f(n), Polynomial::new(pol.to_vec()).eval(n));
            }
        }
        check(&[], |_x| 0);
        check(&[1], |_x| 1);
        check(&[1, 1], |x| x + 1);
        check(&[0, 1], |x| x);
        check(&[10, -10, 10], |x| 10 * x * x - 10 * x + 10);
    }

    #[test]
    fn least_squares() {
        fn check(max_deg: usize, xyws: impl Iterator<Item = (f64, f64, f64)> + Clone) {
            const JITTER: f64 = 1e-7;
            for deg in 0..=max_deg {
                let mut p =
                    Polynomial::<_, Vec<_>>::least_squares_fit_weighted(deg, xyws.clone()).unwrap();

                let diff: f64 = xyws
                    .clone()
                    .map(|(xi, yi, wi)| wi * (p.eval(xi) - yi).powi(2))
                    .sum();

                for i in 0..p.data.len() {
                    for sgn in [-1., 1.] {
                        let bckp = p.data[i];
                        p.data[i] += sgn * JITTER;
                        let jitter_diff: f64 = xyws
                            .clone()
                            .map(|(xi, yi, wi)| wi * (p.eval(xi) - yi).powi(2))
                            .sum();

                        assert!(
                            diff <= jitter_diff,
                            "Jitter caused better fit: {:?}>{:?}, deg={}, i={}",
                            diff,
                            jitter_diff,
                            deg,
                            i
                        );

                        p.data[i] = bckp
                    }
                }
            }
        }

        let xs: Vec<f64> = (0..50).map(|x| x as f64 / 50.).collect();

        check(2, xs.iter().map(|&x| (x, x.powi(2), 1.)));
        check(3, xs.iter().map(|&x| (x, x.powi(4) - x + 3., x)));
        check(5, xs.iter().map(|&x| (x, x.ln_1p(), 1. - x)));

        assert_eq!(
            Polynomial::<_, Vec<_>>::least_squares_fit(1, [(0., 0.), (0., 1.)].into_iter()),
            None
        );
    }

    #[test]
    fn lagrange() {
        // Evaluate the lagrange polynomial at the x coordinates.
        // The error should be close to zero.
        fn check(xs: impl Iterator<Item = f64> + Clone, p: Polynomial<f64>) {
            let p_test = Polynomial::<_, Vec<_>>::lagrange(xs.map(|xi| (xi, p.eval(xi)))).unwrap();
            assert!(p_test.data.len() == p.data.len());
            p_test
                .data
                .into_iter()
                .zip(p.data.into_iter())
                .for_each(|(c_test, c)| assert!((c_test - c).abs() < 1e-9));
        }

        // Squares
        check([1., 2., 3.].iter().copied(), Polynomial::new(vec![0., 10.]));
        // Cubes
        check(
            [-1., 0., 1., 2.].iter().copied(),
            Polynomial::new(vec![0., 0., 0., 1.]),
        );
        // Non linear x.
        check(
            [1., 9., 10., 11.].iter().copied(),
            Polynomial::new(vec![-1., 2., -3., 4.]),
        );

        // Test double x failure case.
        assert_eq!(
            Polynomial::<f64>::lagrange(
                [1., 9., 9., 11.]
                    .iter()
                    .copied()
                    .zip([1., 2., 3., 4.].iter().copied())
            ),
            None
        );
    }

    #[test]
    fn chebyshev() {
        // Construct a Chebyshev approximation for a function
        // and evaulate it at 100 points.
        fn check<F: Fn(f64) -> f64>(f: &F, n: usize, xmin: f64, xmax: f64) {
            let p = Polynomial::<f64>::chebyshev(f, n, xmin, xmax).unwrap();
            for i in 0..=100 {
                let x = xmin + (i as f64) * ((xmax - xmin) / 100.0);
                let diff = (f(x) - p.eval(x)).abs();
                assert!(diff < 1e-4);
            }
        }

        // Approximate some common functions.
        use core::f64::consts::PI;
        check(&f64::sin, 7, -PI / 2., PI / 2.);
        check(&f64::cos, 7, 0., PI / 4.);
        check(&f64::ln, 5, 2., 1.);

        // Test n >= 1 condition
        assert!(Polynomial::<f64>::chebyshev(&f64::exp, 0, 0., 1.).is_none());
    }
}
