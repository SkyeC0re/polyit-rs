//! # poly_it
//! A no-std library for manipulating polynomials with slice support and minimal allocation (or no allocation).
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
//!
//! ## Stack Storage
//!
//! By default this crate has the `alloc` feature enabled and sets the default storage mechanism for [`Polynomial`](struct@Polynomial)
//! to `Vec`. It is however possible to have array backing for a polynomial with the `tinyvec` feature:
//! ```
//! extern crate poly_it;
//! use poly_it::{Polynomial, storage::tinyvec::ArrayVec};
//!
//!
//! pub fn main() {
//!     let p1 = Polynomial::new(ArrayVec::from([1, 2, 3]));
//! }
//!```
//!
//! ## Polynomial Division
//!
//! Polynomial division is defined similar to polynomial long division, in the sense that for two polynomials $A$ and $B$, $A / B$
//! produces a quotient $Q$ and a remainder $R$ such that:
//! $$
//! A \approx BQ + R
//! $$
//! up to the precision of the underlying numeric type (integer types can be seen as infinitely precise as their results
//! conform to their mathematical closures i.e. 5 + 6 is exactly 11 and no precision loss occurs).
//!
//! A notable consequence of this is that $R$ may be of the same degree as $A$, even if $B$ has a degree greater than 0.
//! As an example case, consider the division $(-10x + 1) / (7x + 2)$ for the 32 bit integer type.
//! This will produce $Q = -1$ and $R = -3x + 3$.
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

pub mod storage;

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use core::fmt::{Debug, Display};
use core::marker::PhantomData;
use core::ops::{Add, Div, Mul, Neg, Sub};
use core::{fmt, mem};
use num_traits::{Float, FloatConst, FromPrimitive, One, Zero};

pub use num_traits;
use storage::{Storage, StorageProvider};

/// A polynomial.
#[cfg(feature = "alloc")]
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Polynomial<T, S: Storage<T> = Vec<T>> {
    data: S,
    _type: PhantomData<T>,
}

/// A polynomial.
#[cfg(not(feature = "alloc"))]
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Polynomial<T, S: Storage<T>> {
    data: S,
    _type: PhantomData<T>,
}

impl<T, S> Debug for Polynomial<T, S>
where
    T: Debug,
    S: Storage<T>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Polynomial")
            .field("data", &self.data.as_slice())
            .finish()
    }
}

impl<T, S1, S2> PartialEq<Polynomial<T, S2>> for Polynomial<T, S1>
where
    T: PartialEq,
    S1: Storage<T>,
    S2: Storage<T>,
{
    fn eq(&self, other: &Polynomial<T, S2>) -> bool {
        self.data.as_slice() == other.data.as_slice()
    }
}

impl<T, S> Eq for Polynomial<T, S>
where
    T: Eq,
    S: Storage<T>,
{
}

impl<T: Zero, S: Storage<T>> Polynomial<T, S> {
    /// Creates a new `Polynomial` from a storage of coefficients. Automatically
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
    T: Zero + One + Mul<T, Output = T> + Clone,
    S: Storage<T> + Clone,
{
    /// Computes the derivative in place.
    pub fn deriv_mut(&mut self) {
        let data = self.data.as_mut_slice();
        let mut i = 1;
        let mut carry = T::zero();
        while i < data.len() {
            carry = carry + T::one();
            data[i - 1] = carry.clone() * data[i].clone();
            i += 1;
        }

        let _ = self.data.pop();
    }

    /// Computes the derivate.
    #[inline]
    pub fn deriv(&self) -> Self {
        let mut ret = self.clone();
        ret.deriv_mut();
        ret
    }
}

impl<T, S> Polynomial<T, S>
where
    T: Zero + One + Div<T, Output = T> + Clone,
    S: Storage<T> + Clone,
{
    /// Computes the antiderivative at $C = 0$ in place.
    pub fn antideriv_mut(&mut self) {
        if self.data.len() == 0 {
            return;
        }
        self.data.push(T::zero());
        let data = self.data.as_mut_slice();

        let mut i = 1;
        let mut carry = T::zero();
        let mut ci = T::zero();
        mem::swap(&mut ci, &mut data[0]);
        while i < data.len() {
            carry = carry + T::one();
            mem::swap(&mut ci, &mut data[i]);
            data[i] = data[i].clone() / carry.clone();
            i += 1;
        }
    }

    /// Computes the antiderivative at $C = 0$.
    #[inline]
    pub fn antideriv(&self) -> Self {
        let mut ret = self.clone();
        ret.antideriv_mut();
        ret
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

        let mut p_data = S::Provider::new().storage_with_capacity(deg + 1);

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
        let mut provider = S::Provider::new();
        let mut res = Polynomial::new(provider.new_storage());
        let mut li = Polynomial::new(provider.new_storage());
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
    S::Provider: StorageProvider<(T, T)> + StorageProvider<T>,
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

        let mut samples = <S::Provider as StorageProvider<(T, T)>>::storage_with_capacity(
            &mut <S::Provider as StorageProvider<(T, T)>>::new(),
            n,
        );

        let pi_over_n = T::PI() / T::from(n)?;
        let two = T::one() + T::one();
        let mut k_phalf = T::one() / two;
        let x_avg = (b + a) * k_phalf;
        let x_half_delta = (b - a).abs() * k_phalf;

        for _k in 0..n {
            let x = x_avg - x_half_delta * T::cos(k_phalf * pi_over_n);
            samples.push((x, f(x)));
            k_phalf = k_phalf + T::one();
        }

        Polynomial::lagrange(samples.as_slice().into_iter().copied())
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

impl<T, S> Neg for Polynomial<T, S>
where
    T: Neg<Output = T> + Clone,
    S: Storage<T>,
{
    type Output = Polynomial<T, S>;

    #[inline]
    fn neg(mut self) -> Self::Output {
        self.data
            .as_mut_slice()
            .into_iter()
            .for_each(|c| *c = -c.clone());
        self
    }
}

impl<'a, T, S> Neg for &'a Polynomial<T, S>
where
    T: Neg<Output = T> + Zero + Clone,
    S: Storage<T>,
{
    type Output = Polynomial<T, S>;

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

impl<T, S> Div<&[T]> for Polynomial<T, S>
where
    T: Zero + Mul<T, Output = T> + Div<T, Output = T> + Sub<T, Output = T> + PartialEq + Clone,
    S: Storage<T>,
{
    type Output = (Polynomial<T, S>, Polynomial<T, S>);

    fn div(mut self, other: &[T]) -> Self::Output {
        self.trim();

        let rem_data = self.coeffs_mut();

        let mut cutoff = other.len();
        for elem in other.into_iter().rev() {
            if elem != &T::zero() {
                break;
            }
            cutoff -= 1;
        }

        let div_data = &other[..cutoff];
        let main_divisor = match div_data.last().filter(|_| div_data.len() <= rem_data.len()) {
            Some(v) => v.clone(),
            None => return (Polynomial::new(S::Provider::new().new_storage()), self),
        };
        let dd_lm1 = div_data.len() - 1;
        let mut ret = Polynomial::new(
            S::Provider::new().storage_with_capacity((rem_data.len() - div_data.len()) + 1),
        );

        for i in (dd_lm1..rem_data.len()).rev() {
            let val = rem_data[i].clone() / main_divisor.clone();
            for (r, d) in rem_data[(i - dd_lm1)..=i].iter_mut().zip(div_data.iter()) {
                *r = r.clone() - val.clone() * d.clone()
            }
            ret.data.push(val);
        }
        ret.coeffs_mut().reverse();

        ret.trim();
        self.trim();

        (ret, self)
    }
}

impl<T, S> Div<Polynomial<T, S>> for &[T]
where
    T: Zero + Mul<T, Output = T> + Div<T, Output = T> + Sub<T, Output = T> + PartialEq + Clone,
    S: Storage<T>,
{
    type Output = (Polynomial<T, S>, Polynomial<T, S>);

    #[inline]
    fn div(self, other: Polynomial<T, S>) -> Self::Output {
        let mut owned_data = S::Provider::new().storage_with_capacity(self.len());
        for elem in self {
            owned_data.push(elem.clone());
        }

        Polynomial::new(owned_data) / other.coeffs()
    }
}

impl<T, S1, S2> Div<Polynomial<T, S2>> for Polynomial<T, S1>
where
    T: Zero + Mul<T, Output = T> + Div<T, Output = T> + Sub<T, Output = T> + PartialEq + Clone,
    S1: Storage<T>,
    S2: Storage<T>,
{
    type Output = (Polynomial<T, S1>, Polynomial<T, S1>);

    #[inline(always)]
    fn div(self, other: Polynomial<T, S2>) -> Self::Output {
        self / other.coeffs()
    }
}

impl<T, S1, S2> Div<Polynomial<T, S2>> for &Polynomial<T, S1>
where
    T: Zero + Mul<T, Output = T> + Div<T, Output = T> + Sub<T, Output = T> + PartialEq + Clone,
    S1: Storage<T>,
    S2: Storage<T>,
{
    type Output = (Polynomial<T, S2>, Polynomial<T, S2>);

    #[inline(always)]
    fn div(self, other: Polynomial<T, S2>) -> Self::Output {
        self.coeffs() / other
    }
}

impl<T, S1, S2> Div<&Polynomial<T, S2>> for Polynomial<T, S1>
where
    T: Zero + Mul<T, Output = T> + Div<T, Output = T> + Sub<T, Output = T> + PartialEq + Clone,
    S1: Storage<T>,
    S2: Storage<T>,
{
    type Output = (Polynomial<T, S1>, Polynomial<T, S1>);

    #[inline(always)]
    fn div(self, other: &Polynomial<T, S2>) -> Self::Output {
        self / other.coeffs()
    }
}

impl<T, S1, S2> Div<&Polynomial<T, S2>> for &Polynomial<T, S1>
where
    T: Zero + Mul<T, Output = T> + Div<T, Output = T> + Sub<T, Output = T> + PartialEq + Clone,
    S1: Storage<T>,
    S2: Storage<T>,
{
    type Output = (Polynomial<T, S1>, Polynomial<T, S1>);

    #[inline(always)]
    fn div(self, other: &Polynomial<T, S2>) -> Self::Output {
        self.clone() / other.coeffs()
    }
}

impl<T, S> Div<&[T]> for &Polynomial<T, S>
where
    T: Zero + Mul<T, Output = T> + Div<T, Output = T> + Sub<T, Output = T> + PartialEq + Clone,
    S: Storage<T>,
{
    type Output = (Polynomial<T, S>, Polynomial<T, S>);

    #[inline(always)]
    fn div(self, other: &[T]) -> Self::Output {
        self.clone() / other
    }
}

impl<T, S> Div<&Polynomial<T, S>> for &[T]
where
    T: Zero + Mul<T, Output = T> + Div<T, Output = T> + Sub<T, Output = T> + PartialEq + Clone,
    S: Storage<T>,
{
    type Output = (Polynomial<T, S>, Polynomial<T, S>);

    #[inline(always)]
    fn div(self, other: &Polynomial<T, S>) -> Self::Output {
        self / other.clone()
    }
}
