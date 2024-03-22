//! # poly_it
//! A no-std library for manipulating polynomials with iterator support and minimal allocation.
//!
//! At the end of the day the classical representation method for the polynomial is
//! it coefficients and this library leverages this by means of clone-able iterators.  
//! Take this example case:
//! ```
//! extern crate poly_it;
//! use poly_it::{Polynomial, Coverable};
//!
//! pub fn main() {
//!     let p1 = Polynomial::new(vec![1, 2, 3]);
//!     let p2 = Polynomial::new(vec![3, 2, 1]);
//!     let arr_iter = [1, 2];
//!     let vec_iter = vec![1, 2, 3];
//!
//!     assert_eq!(
//!         format!("{}", p2 - &p1),
//!         "2 + -2x^2"
//!     );
//!
//!     assert_eq!(
//!         format!("{}", p1.clone() + arr_iter.into_iter().covered()),
//!         "2 + 4x + 3x^2"
//!     );
//!
//!     assert_eq!(
//!         format!("{}", p1 * vec_iter.iter().copied().covered()),
//!         "1 + 4x + 10x^2 + 12x^3 + 9x^4"
//!     );
//! }
//! ```
//!
//! This example illustrates several things:
//! 1. The use `into_iter()` and `iter().copied()` type iterators
//! 2. The covering mechanism used by this crate
//! 3. Using operations with owned or referenced polynomials.
//! 4. That `+ -` is possibly heretical.
//!
//! As to the fourth point, this has to do with future proofing in cases where the underlying numerical type
//! might not have ordering, but for the first three, see below for a quick summary as to the how and the why.
//!
//! ## Iterator Support
//!
//! ### `into_iter()` vs `iter().cloned()`/`iter().copied()`:
//!
//! Many iterable types gives the user the option between using `into_iter()` to produce elements of
//! type some type `T` or `iter()` to produce `&T`. The issue with the former is that for iterable types which are expensive
//! to clone, this hinders performance as each time the iterator is cloned, the underlying data is cloned with it (as is the case
//! with [Vec](struct@Vec)). For these types the user is urged to use the `iter().cloned()`/`iter().copied()` pattern instead
//! such that the underlying iterable type gets reused on a clone of the iterator.
//!
//! **Caveats**
//!
//! This crate assumes the following for any iterator passed to it during a function call:
//! 1. The iterator is fixed for the duration of the function. That is to say that it will always produce the
//!    same list of elements regardless of how long the function waits before iterating over it.
//! 2. Any clones of the iterator will produce the same elements as the iterator.
//!
//! ### Covered iterators
//!
//! Rust allows the following implementation of a trait:
//! ```text
//! impl ForeignTrait<LocalType> for ForeignType
//! ```
//! but not:
//! ```text
//! impl<T: SomeTrait> ForeignTrait<LocalType> for T
//! ```
//! (see [this issue](https://github.com/rust-lang/rust/issues/63599) for more details). This means that it
//! is impossible to implement binary operations with iterators on the left hand side. One such example would be:
//! ```text
//! impl<T: Add<T, Output=T>, I: Iterator<Item=T>> Add<Polynomial<T>> for I
//! ```
//! even though we could manually implement (in a similar fashion as
//! [nalgedra  did for the primitive numeric types](https://docs.rs/nalgebra/latest/src/nalgebra/base/ops.rs.html#548)):
//! ```text
//! impl<T: Add<T, Output=T>> Add<Polynomial<T>> for SomeConcreteIterator
//! ```
//! for all iterators and nested and chained iterator combinations...
//!
//! We will not be doing this.
//!
//! Instead we propose a simple covering mechanism, such that by wrapping the iterators in a local type (see [Covered](struct@Covered))
//! we are able to implement:
//! ```text
//! impl<T: Add<T, Output=T>, I: Iterator<Item=T>> Add<Polynomial<T>> for LocalWrapperType<I>
//! ```
//!
//! Note: Although not required we also enforce this when the Polynomial is the LHS for consistency.
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

extern crate alloc;
use alloc::{vec, vec::Vec};
use core::fmt::Display;
use core::ops::{Add, Deref, DerefMut, Div, Mul, Neg, Sub};
use core::{fmt, mem};
use num_traits::{Float, FloatConst, FromPrimitive, One, Zero};

pub use num_traits;

/// A struct for covering generic objects under this crate.
/// Currently it is not possible to implement the following in Rust:  
/// `impl<T> ForeignTrait<LocalType> for T`  
/// but it is possible to implement:  
/// `impl<T> ForeignTrait<LocalType> for LocalType<T>`  
#[repr(transparent)]
pub struct Covered<T>(pub T);

impl<T> Deref for Covered<T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Covered<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// A convenience trait for the [Covered](struct@Covered) which
/// allows any sized type to be covered under this crate by calling [covered](fn@Coverable::covered)
/// on it.
pub trait Coverable {
    /// Covers an object under this crate.
    fn covered(self) -> Covered<Self>
    where
        Self: Sized;
}

impl<T: Sized> Coverable for T {
    #[inline(always)]
    fn covered(self) -> Covered<Self>
    where
        Self: Sized,
    {
        Covered(self)
    }
}

/// A polynomial.
#[derive(Eq, PartialEq, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Polynomial<T> {
    data: Vec<T>,
}

impl<T: Zero> Polynomial<T> {
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
    pub fn new(data: Vec<T>) -> Self {
        let mut p = Self { data };
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
        while let Some(true) = self.data.last().map(T::is_zero) {
            let _ = self.data.pop();
        }
    }

    /// Get an immutable reference to the polynomial's coefficients.
    #[inline]
    pub fn coeffs(&self) -> &[T] {
        &self.data
    }

    /// Get a mutable reference to the polynomial's coefficients.
    #[inline]
    pub fn coeffs_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T> Into<Vec<T>> for Polynomial<T> {
    #[inline]
    fn into(self) -> Vec<T> {
        self.data
    }
}

impl<T: Display> Display for Polynomial<T>
where
    T: Zero,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut previous_nonzero_term = false;
        for i in 0..self.data.len() {
            let ci = &self.data[i];
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

impl<T> Polynomial<T>
where
    T: Zero
        + One
        + Add<T, Output = T>
        + Neg<Output = T>
        + Sub<T, Output = T>
        + Mul<T, Output = T>
        + Div<T, Output = T>
        + Clone,
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
    /// let poly = Polynomial::least_squares_fit(
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
    /// let poly = Polynomial::least_squares_fit_weighted(
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

        let mut p_data = Vec::with_capacity(deg + 1);
        p_data.push(d_0);

        if deg == 0 {
            return Some(Polynomial::new(p_data));
        }

        let mut p_km1_offset = data_len;
        let mut p_k_offset = data_len + (deg + 1);
        // Stores x_i^{k + 1} data, P_k data and P_{k-1} data.
        let mut computation_data = Vec::with_capacity(data_len + ((deg + 1) << 1));
        computation_data.extend(samples.clone().map(|(xi, _, _)| xi));
        computation_data.extend((0usize..(deg + 1) << 1).map(|_| T::zero()));

        computation_data[p_k_offset] = T::one();

        let eval_slice = |slice: &[T], x: T| -> T {
            slice
                .into_iter()
                .rev()
                .cloned()
                .reduce(|acc, c_i| x.clone() * acc + c_i)
                .unwrap_or(T::zero())
        };

        let mut gamma_k = gamma_0;
        let mut b_k = b_0;
        let mut minus_c_k = T::zero();
        let mut kp1 = 1;

        loop {
            p_data.push(Zero::zero());

            // Overwrite p_{k-1} with p_{k+1}
            for i in 0..(kp1 - 1) {
                computation_data[p_km1_offset + i] =
                    minus_c_k.clone() * computation_data[p_km1_offset + i].clone();
            }
            for i in 0..kp1 {
                computation_data[p_km1_offset + i] = computation_data[p_km1_offset + i].clone()
                    - b_k.clone() * computation_data[p_k_offset + i].clone();
                computation_data[p_km1_offset + i + 1] = computation_data[p_km1_offset + i + 1]
                    .clone()
                    + computation_data[p_k_offset + i].clone();
            }

            let (mut d_kp1, gamma_kp1, mut b_kp1) = samples.clone().enumerate().fold(
                (T::zero(), T::zero(), T::zero()),
                |mut acc, (i, (xi, yi, wi))| {
                    let weighted_eval = wi
                        * eval_slice(
                            &computation_data[p_km1_offset..(p_km1_offset + kp1 + 1)],
                            xi.clone(),
                        );

                    acc.0 = acc.0 + yi * weighted_eval.clone();
                    let mut xi_powk = computation_data[i].clone();
                    acc.1 = acc.1 + xi_powk.clone() * weighted_eval.clone();

                    // Update x_i^k to x_i^{k+1}
                    xi_powk = xi_powk * xi;
                    computation_data[i] = xi_powk.clone();

                    acc.2 = acc.2 + xi_powk * weighted_eval;
                    acc
                },
            );

            if gamma_kp1.is_zero() {
                return None;
            }

            d_kp1 = d_kp1 / gamma_kp1.clone();

            for i in 0..(kp1 + 1) {
                p_data[i] =
                    p_data[i].clone() + d_kp1.clone() * computation_data[p_km1_offset + i].clone();
            }

            if kp1 == deg {
                break;
            }

            // Delay the remaining work left for b_{k+1} until it is certain to be needed.
            b_kp1 = b_kp1 / gamma_kp1.clone() + computation_data[p_km1_offset + kp1 - 1].clone();

            kp1 += 1;
            b_k = b_kp1;
            minus_c_k = -(gamma_kp1.clone() / gamma_k);
            gamma_k = gamma_kp1;

            // Reorient the offsets
            mem::swap(&mut p_k_offset, &mut p_km1_offset);
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
        let mut res = Polynomial::new(vec![Zero::zero()]);
        let mut li = Polynomial::new(Vec::new());
        for (i, (x, y)) in samples.clone().enumerate() {
            li.data.clear();
            li.data.push(T::one());
            let mut denom = T::one();
            for (j, (x2, _)) in samples.clone().enumerate() {
                if i != j {
                    li = li * [-x2.clone(), T::one()].into_iter().covered();
                    let diff = x.clone() - x2.clone();
                    if diff.is_zero() {
                        return None;
                    }
                    denom = denom * diff;
                }
            }
            let scalar = y.clone() / denom;
            li = li * [scalar].into_iter().covered();
            res = res + &li;
        }
        res.trim();
        Some(res)
    }
}

impl<T> Polynomial<T>
where
    T: Zero + One + FloatConst + Float + FromPrimitive,
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
    /// let p = Polynomial::chebyshev(&f64::sin, 7, PI/4., -PI/4.).unwrap();
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

impl<T: Zero + Mul<Output = T> + Clone> Polynomial<T> {
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
    #[inline]
    pub fn eval(&self, x: T) -> T {
        let mut result: T = Zero::zero();
        for n in self.data.iter().rev() {
            result = result * x.clone() + n.clone();
        }
        result
    }
}

impl<T> Neg for Polynomial<T>
where
    T: Neg<Output = T> + Clone,
{
    type Output = Polynomial<T>;

    #[inline]
    fn neg(mut self) -> Polynomial<T> {
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
    fn neg(self) -> Polynomial<T> {
        -self.clone()
    }
}

macro_rules! forward_ref_iter_binop {
    (impl $imp:ident, $method:ident) => {
        impl<'a, T, I> $imp<Covered<I>> for &'a Polynomial<T>
        where
            T: 'a + $imp<T, Output = T> + Zero + Clone,
            I: Iterator<Item = T> + Clone,
        {
            type Output = Polynomial<T>;

            #[inline(always)]
            fn $method(self, other: Covered<I>) -> Polynomial<T> {
                $imp::$method(self.clone(), other)
            }
        }
    };
}

macro_rules! forward_iter_ref_binop {
    (impl $imp:ident, $method:ident) => {
        impl<'a, T, I> $imp<&'a Polynomial<T>> for Covered<I>
        where
            T: 'a + $imp<T, Output = T> + Zero + Clone,
            I: Iterator<Item = T> + Clone,
        {
            type Output = Polynomial<T>;

            #[inline(always)]
            fn $method(self, other: &'a Polynomial<T>) -> Polynomial<T> {
                $imp::$method(self, other.clone())
            }
        }
    };
}

macro_rules! forward_iter_val_val_binop {
    (impl $imp:ident, $method:ident) => {
        impl<T> $imp<Polynomial<T>> for Polynomial<T>
        where
            T: $imp<T, Output = T> + Zero + Clone,
        {
            type Output = Polynomial<T>;

            #[inline(always)]
            fn $method(self, other: Polynomial<T>) -> Polynomial<T> {
                if self.data.capacity() >= other.data.capacity() {
                    $imp::$method(self, other.data.iter().cloned().covered())
                } else {
                    $imp::$method(self.data.iter().cloned().covered(), other)
                }
            }
        }
    };
}

macro_rules! forward_iter_ref_val_binop {
    (impl $imp:ident, $method:ident) => {
        impl<'a, T> $imp<Polynomial<T>> for &'a Polynomial<T>
        where
            T: $imp<T, Output = T> + Zero + Clone,
        {
            type Output = Polynomial<T>;

            #[inline(always)]
            fn $method(self, other: Polynomial<T>) -> Polynomial<T> {
                $imp::$method(self.data.iter().cloned().covered(), other)
            }
        }
    };
}

macro_rules! forward_iter_val_ref_binop {
    (impl $imp:ident, $method:ident) => {
        impl<'a, T> $imp<&'a Polynomial<T>> for Polynomial<T>
        where
            T: $imp<T, Output = T> + Zero + Clone,
        {
            type Output = Polynomial<T>;

            #[inline(always)]
            fn $method(self, other: &'a Polynomial<T>) -> Polynomial<T> {
                $imp::$method(self, other.data.iter().cloned().covered())
            }
        }
    };
}

macro_rules! forward_iter_ref_ref_binop {
    (impl $imp:ident, $method:ident) => {
        impl<'a, 'b, T> $imp<&'b Polynomial<T>> for &'a Polynomial<T>
        where
            T: $imp<T, Output = T> + Zero + Clone,
        {
            type Output = Polynomial<T>;

            #[inline(always)]
            fn $method(self, other: &'b Polynomial<T>) -> Polynomial<T> {
                if self.data.len() >= other.data.len() {
                    $imp::$method(self, other.data.iter().cloned().covered())
                } else {
                    $imp::$method(self.data.iter().cloned().covered(), other)
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

impl<T, I> Add<Covered<I>> for Polynomial<T>
where
    T: Zero + Add<T, Output = T> + Clone,
    I: Iterator<Item = T> + Clone,
{
    type Output = Polynomial<T>;

    fn add(mut self, iter: Covered<I>) -> Polynomial<T> {
        let poly_len = self.data.len();
        for (j, bj) in iter.0.enumerate() {
            if j < poly_len {
                self.data[j] = self.data[j].clone() + bj;
            } else {
                self.data.push(bj);
            }
        }
        self.trim();
        self
    }
}

impl<T, I> Add<Polynomial<T>> for Covered<I>
where
    T: Zero + Add<T, Output = T> + Clone,
    I: Iterator<Item = T> + Clone,
{
    type Output = Polynomial<T>;
    #[inline(always)]
    fn add(self, poly: Polynomial<T>) -> Polynomial<T> {
        poly + self
    }
}

impl<T, I> Sub<Covered<I>> for Polynomial<T>
where
    T: Zero + Sub<T, Output = T> + Clone,
    I: Iterator<Item = T> + Clone,
{
    type Output = Polynomial<T>;

    fn sub(mut self, iter: Covered<I>) -> Polynomial<T> {
        let poly_len = self.data.len();
        for (j, bj) in iter.0.enumerate() {
            if j < poly_len {
                self.data[j] = self.data[j].clone() - bj;
            } else {
                self.data.push(T::zero() - bj);
            }
        }
        self.trim();
        self
    }
}

impl<T, I> Sub<Polynomial<T>> for Covered<I>
where
    T: Zero + Sub<T, Output = T> + Clone,
    I: Iterator<Item = T> + Clone,
{
    type Output = Polynomial<T>;

    fn sub(self, mut poly: Polynomial<T>) -> Polynomial<T> {
        let poly_len = poly.data.len();
        let mut j = 0;
        for bj in self.0 {
            if j < poly_len {
                poly.data[j] = bj - poly.data[j].clone();
            } else {
                poly.data.push(bj);
            }
            j += 1;
        }
        while j < poly_len {
            poly.data[j] = T::zero() - poly.data[j].clone();
            j += 1;
        }

        poly.trim();
        poly
    }
}

impl<T, I> Mul<Covered<I>> for Polynomial<T>
where
    T: Zero + Mul<T, Output = T> + Clone,
    I: Iterator<Item = T> + Clone,
{
    type Output = Polynomial<T>;

    fn mul(mut self, mut iter: Covered<I>) -> Polynomial<T> {
        let mut ai = match self.data.last() {
            Some(v) => v.clone(),
            None => return self,
        };
        let last_index = self.data.len() - 1;
        let b0 = match iter.next() {
            Some(v) => v.clone(),
            None => {
                self.data.clear();
                return self;
            }
        };

        self.data[last_index] = ai.clone() * b0.clone();
        iter.clone().for_each(|bj| self.data.push(ai.clone() * bj));
        for i in (0..last_index).rev() {
            ai = self.data[i].clone();
            self.data[i] = ai.clone() * b0.clone();
            self.data[(i + 1)..]
                .iter_mut()
                .zip(iter.clone())
                .for_each(|(v, bj)| *v = v.clone() + ai.clone() * bj);
        }
        self.trim();
        self
    }
}

impl<T, I> Mul<Polynomial<T>> for Covered<I>
where
    T: Zero + Mul<T, Output = T> + Clone,
    I: Iterator<Item = T> + Clone,
{
    type Output = Polynomial<T>;
    #[inline]
    fn mul(self, poly: Polynomial<T>) -> Polynomial<T> {
        poly * self
    }
}

impl<T: Zero + Clone> Zero for Polynomial<T> {
    #[inline]
    fn zero() -> Self {
        Self { data: vec![] }
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
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use super::{Coverable, Polynomial};
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

    #[test]
    fn neg_add_sub() {
        fn check(a: &[i32], b: &[i32], c: &[i32]) {
            fn check_eq(a: &Polynomial<i32>, b: &Polynomial<i32>) {
                assert_eq!(*a, *b);
                assert_eq!(-a, -b);
            }
            fn check_add(sum: &Polynomial<i32>, a: &Polynomial<i32>, b: &Polynomial<i32>) {
                check_eq(sum, &(a + b));
                check_eq(sum, &(b + a));
            }
            fn check_sub(sum: &Polynomial<i32>, a: &Polynomial<i32>, b: &Polynomial<i32>) {
                check_eq(a, &(sum - b));
                check_eq(b, &(sum - a));
            }

            let a = &Polynomial::new(a.to_vec());
            let b = &Polynomial::new(b.to_vec());
            let c = &Polynomial::new(c.to_vec());
            check_add(c, a, b);
            check_add(&(-c), &(-a), &(-b));
            check_sub(c, a, b);
            check_sub(&(-c), &(-a), &(-b));
        }
        check(&[], &[], &[]);
        check(&[], &[1], &[1]);
        check(&[1], &[1], &[2]);
        check(&[1, 0, 1], &[1], &[2, 0, 1]);
        check(&[1, 0, -1], &[-1, 0, 1], &[]);
    }

    #[test]
    fn mul() {
        fn check(a: &[i32], b: &[i32], c: &[i32]) {
            let a_p = Polynomial::new(a.to_vec());
            let b_p = Polynomial::new(b.to_vec());
            let c_p = Polynomial::new(c.to_vec());
            assert_eq!(c_p, &a_p * &b_p);
            assert_eq!(c_p, &b_p * &a_p);
            assert_eq!(c_p, &a_p * b.into_iter().copied().covered());
            assert_eq!(c_p, a.into_iter().copied().covered() * &b_p);
        }
        check(&[], &[], &[]);
        check(&[0, 0], &[], &[]);
        check(&[0, 0], &[1], &[]);
        check(&[1, 0], &[1], &[1]);
        check(&[1, 0, 1], &[1], &[1, 0, 1]);
        check(&[1, 1], &[1, 1], &[1, 2, 1]);
        check(&[1, 1], &[1, 0, 1], &[1, 1, 1, 1]);
        check(&[0, 0, 1], &[0, 0, 1], &[0, 0, 0, 0, 1]);
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
                let mut p = Polynomial::least_squares_fit_weighted(deg, xyws.clone()).unwrap();

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
            Polynomial::least_squares_fit(1, [(0., 0.), (0., 1.)].into_iter()),
            None
        );
    }

    #[test]
    fn lagrange() {
        // Evaluate the lagrange polynomial at the x coordinates.
        // The error should be close to zero.
        fn check(xs: impl Iterator<Item = f64> + Clone, p: Polynomial<f64>) {
            let p_test = Polynomial::lagrange(xs.map(|xi| (xi, p.eval(xi)))).unwrap();
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
    #[cfg(any(feature = "std", feature = "libm"))]
    fn chebyshev() {
        // Construct a Chebyshev approximation for a function
        // and evaulate it at 100 points.
        fn check<F: Fn(f64) -> f64>(f: &F, n: usize, xmin: f64, xmax: f64) {
            let p = Polynomial::chebyshev(f, n, xmin, xmax).unwrap();
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
        assert!(Polynomial::chebyshev(&f64::exp, 0, 0., 1.).is_none());
    }
}
