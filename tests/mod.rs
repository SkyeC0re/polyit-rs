#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use core::ops::{Add, Mul, Sub};
use num_traits::Zero;
use paste::paste;
use poly_it::{
    storage::{Storage, StorageProvider},
    Polynomial,
};
use std::ops::Div;

fn poly_from_slice<T, S>(slice: &[T]) -> Polynomial<T, S>
where
    S: Storage<T>,
    T: Zero + Clone,
{
    let mut p = S::Provider::new().storage_with_capacity(slice.len());
    for x in slice {
        p.push(x.clone());
    }

    Polynomial::new(p)
}

macro_rules! test_all_with_storage {
    ($prefix:ident, $storage:ty) => {
        paste! {
            #[test]
            fn [<$prefix _new>]() {
                test_new::<$storage<_>>()
            }

            #[test]
            fn [<$prefix _eq>]() {
                test_eq::<$storage<_>>()
            }

            #[test]
            fn [<$prefix _neg>]() {
                test_neg::<$storage<_>>()
            }

            #[test]
            fn [<$prefix _add>]() {
                test_add::<$storage<_>>()
            }

            #[test]
            fn [<$prefix _sub>]() {
                test_sub::<$storage<_>>()
            }

            #[test]
            fn [<$prefix _mul>]() {
                test_mul::<$storage<_>>()
            }

            #[test]
            fn [<$prefix _div>]() {
                test_div::<$storage<_>>()
            }

            #[test]
            fn [<$prefix _eval>]() {
                test_eval::<$storage<_>>()
            }

            #[test]
            fn [<$prefix _least_squares>]() {
                test_least_squares::<$storage<_>>()
            }

            #[test]
            fn [<$prefix _lagrange>]() {
                test_lagrange::<$storage<_>>()
            }

            #[test]
            fn [<$prefix _chebyshev>]() {
                test_chebyshev::<$storage<_>>()
            }

            #[test]
            fn [<$prefix _deriv>]() {
                test_deriv::<$storage<_>>()
            }

            #[test]
            fn [<$prefix _antideriv>]() {
                test_antideriv::<$storage<_>>()
            }
        }
    };
}

#[cfg(feature = "alloc")]
test_all_with_storage!(vec, Vec);
#[cfg(feature = "tinyvec")]
type TinyVecArr<T> = tinyvec::ArrayVec<[T; 10]>;
#[cfg(feature = "tinyvec")]
test_all_with_storage!(tinyvec_arr, TinyVecArr);

#[cfg(all(feature = "tinyvec", feature = "alloc"))]
type TinyVec<T> = tinyvec::TinyVec<[T; 3]>;
#[cfg(all(feature = "tinyvec", feature = "alloc"))]
test_all_with_storage!(tinyvec, TinyVec);

#[cfg(feature = "arrayvec")]
type ArrayVec<T> = arrayvec::ArrayVec<T, 10>;
#[cfg(feature = "arrayvec")]
test_all_with_storage!(arrayvec, ArrayVec);

#[cfg(feature = "smallvec")]
type SmallVec<T> = smallvec::SmallVec<[T; 3]>;
#[cfg(feature = "smallvec")]
test_all_with_storage!(smallvec, SmallVec);

#[cfg(feature = "alloc")]
#[test]
fn test_display() {
    assert_eq!(&format!("{}", Polynomial::new(Vec::<f64>::new())), "0");
    assert_eq!(
        &format!("{:.3}", Polynomial::new(vec![-1f64, 1f64])),
        "-1.000 + 1.000x"
    );
    assert_eq!(
        &format!("{:.1}", Polynomial::new(vec![1f64, 0f64, -2f64, 3f64])),
        "1.0 + -2.0x^2 + 3.0x^3"
    );
}

fn test_new<S: Storage<i32>>() {
    #[track_caller]
    fn check<S: Storage<i32>>(dst: &[i32], src: &[i32]) {
        let mut data = S::Provider::new().new_storage();
        for &elem in src {
            data.push(elem);
        }
        assert_eq!(dst, Polynomial::new(data).coeffs());
    }
    check::<S>(&[1, 2, 3], &[1, 2, 3]);
    check::<S>(&[1, 2, 3], &[1, 2, 3, 0, 0]);
    check::<S>(&[], &[0, 0, 0]);
}

fn test_eq<S: Storage<i32>>() {
    #[track_caller]
    fn check<S: Storage<i32>>(a: &[i32], b: &[i32], eq: bool) {
        let mut a_data = S::Provider::new().new_storage();
        for &elem in a {
            a_data.push(elem);
        }
        let mut b_data = S::Provider::new().new_storage();
        for &elem in b {
            b_data.push(elem);
        }
        assert_eq!(Polynomial::new(a_data) == Polynomial::new(b_data), eq);
    }

    check::<S>(&[], &[0, 0], true);
    check::<S>(&[1], &[], false);
    check::<S>(&[1, 2], &[1, 2, 3], false);
    check::<S>(&[0, 0, 1], &[0, 0, 1, 0, 0], true);
    check::<S>(&[1, 2, 3], &[1], false);
}

macro_rules! test_binop {
    (impl $imp:ident, $method:ident, $a:ident, $b:ident, $res:ident) => {
        assert_eq!($imp::$method($a.clone(), $b.clone()), $res);
        assert_eq!($imp::$method($a.clone(), &$b), $res);
        assert_eq!($imp::$method(&$a, $b.clone()), $res);
        assert_eq!($imp::$method(&$a, &$b), $res);
        assert_eq!($imp::$method($a.clone(), $b.coeffs()), $res);
        assert_eq!($imp::$method($a.coeffs(), $b.clone()), $res);
        assert_eq!($imp::$method(&$a, $b.coeffs()), $res);
        assert_eq!($imp::$method($a.coeffs(), &$b), $res);
    };
}

fn test_neg<S: Storage<i32>>() {
    let mut p = S::Provider::new().new_storage();
    for x in 1..=3 {
        p.push(x);
    }
    let mut p_neg = p.clone();
    for x in p_neg.as_mut_slice() {
        *x = -*x;
    }
    let p = Polynomial::new(p);
    let p_neg = Polynomial::new(p_neg);
    assert_eq!(-(&p), p_neg);
    assert_eq!(-p.clone(), p_neg);
    assert_eq!(-(-p.clone()), p);
}

fn test_add<S: Storage<i32>>() {
    let empty: [i32; 0] = [];
    #[track_caller]
    fn check<S: Storage<i32>>(a: &[i32], b: &[i32], res: &[i32]) {
        let a = poly_from_slice::<_, S>(a);
        let b = poly_from_slice::<_, S>(b);
        let res = poly_from_slice::<_, S>(res);
        test_binop!(impl Add, add, a, b, res);
    }

    check::<S>(&empty, &empty, &empty);
    check::<S>(&empty, &[1], &[1]);
    check::<S>(&[1], &empty, &[1]);
    check::<S>(&[1, 2, 3], &[-1, -2, -3], &empty);
    check::<S>(&[0, 2, 4], &[1, 3, 5], &[1, 5, 9]);
    check::<S>(&[1, 2, 3], &[3, 2, 1], &[4, 4, 4]);
}

fn test_sub<S: Storage<i32>>() {
    let empty: [i32; 0] = [];
    #[track_caller]
    fn check<S: Storage<i32>>(a: &[i32], b: &[i32], res: &[i32]) {
        let a = poly_from_slice::<_, S>(a);
        let b = poly_from_slice::<_, S>(b);
        let res = poly_from_slice::<_, S>(res);
        test_binop!(impl Sub, sub, a, b, res);
    }

    check::<S>(&empty, &empty, &empty);
    check::<S>(&empty, &[1], &[-1]);
    check::<S>(&[1], &empty, &[1]);
    check::<S>(&[1, 2, 3], &[1, 2, 3], &empty);
    check::<S>(&[0, 2, 4], &[1, 3, 5], &[-1, -1, -1]);
    check::<S>(&[1, 2, 3], &[3, 2, 1], &[-2, 0, 2]);
}

fn test_mul<S: Storage<i32>>() {
    let empty: [i32; 0] = [];
    #[track_caller]
    fn check<S: Storage<i32>>(a: &[i32], b: &[i32], res: &[i32]) {
        let a = poly_from_slice::<_, S>(a);
        let b = poly_from_slice::<_, S>(b);
        let res = poly_from_slice::<_, S>(res);
        test_binop!(impl Mul, mul, a, b, res);
    }

    check::<S>(&empty, &empty, &empty);
    check::<S>(&empty, &[1], &empty);
    check::<S>(&[1], &empty, &empty);
    check::<S>(&[0], &[1, 2], &empty);
    check::<S>(&[1, 2], &[0], &empty);
    check::<S>(&[1], &[1], &[1]);
    check::<S>(&[1, -3], &[1, -3], &[1, -6, 9]);
    check::<S>(&[1, 1], &[1, 0, 1], &[1, 1, 1, 1]);
    check::<S>(&[0, 0, 1], &[0, -1], &[0, 0, 0, -1]);
}

fn test_div<S: Storage<i32>>() {
    let empty: [i32; 0] = [];
    #[track_caller]
    fn check<S: Storage<i32>>(a: &[i32], b: &[i32], res: &[i32], rem: &[i32]) {
        let a = poly_from_slice::<_, S>(a);
        let b = poly_from_slice::<_, S>(b);
        let res = (poly_from_slice::<_, S>(res), poly_from_slice::<_, S>(rem));
        test_binop!(impl Div, div, a, b, res);
    }

    check::<S>(&empty, &empty, &empty, &empty);
    check::<S>(&empty, &[1], &empty, &empty);
    check::<S>(&[1], &empty, &empty, &[1]);
    check::<S>(&[0], &[1, 2], &empty, &empty);
    check::<S>(&[1, 2], &[0], &empty, &[1, 2]);
    check::<S>(&[1], &[1], &[1], &[0]);
    check::<S>(&[1, -3, 6], &[3, 5], &[-1, 1], &[4, -1, 1]);
    // div  5 ->  5 , 7, 26, -8, 0
    // div -2 ->  5, -7, 24, -2, 0
    // div  8 -> 61,  1,  0, -2, 0
    check::<S>(
        &[5, 7, -9, -13, 15],
        &[-7, -1, 3],
        &[8, -2, 5],
        &[61, 1, 0, -2],
    );
    check::<S>(&[0, 0, 1], &[0, -1], &[0, -1], &empty);
}

fn test_eval<S: Storage<i32>>() {
    #[track_caller]
    fn check<F: Fn(i32) -> i32, S: Storage<i32>>(pol: &[i32], f: F) {
        for n in -10..10 {
            assert_eq!(f(n), poly_from_slice::<_, S>(pol).eval(n));
        }
    }
    check::<_, S>(&[], |_x| 0);
    check::<_, S>(&[1], |_x| 1);
    check::<_, S>(&[1, 1], |x| x + 1);
    check::<_, S>(&[0, 1], |x| x);
    check::<_, S>(&[10, -10, 10], |x| 10 * x * x - 10 * x + 10);
}

fn test_least_squares<S: Storage<f64>>() {
    #[track_caller]
    fn check<S: Storage<f64>>(max_deg: usize, xyws: impl Iterator<Item = (f64, f64, f64)> + Clone) {
        const JITTER: f64 = 1e-7;
        for deg in 0..=max_deg {
            let mut p = Polynomial::<_, S>::least_squares_fit_weighted(deg, xyws.clone()).unwrap();

            let diff: f64 = xyws
                .clone()
                .map(|(xi, yi, wi)| wi * (p.eval(xi) - yi).powi(2))
                .sum();

            for i in 0..p.coeffs().len() {
                for sgn in [-1., 1.] {
                    let bckp = p.coeffs_mut()[i];
                    p.coeffs_mut()[i] += sgn * JITTER;
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

                    p.coeffs_mut()[i] = bckp
                }
            }
        }
    }

    let xs = (0..50).map(|x| x as f64 / 50.);

    check::<S>(2, xs.clone().map(|x| (x, x.powi(2), 1.)));
    check::<S>(3, xs.clone().map(|x| (x, x.powi(4) - x + 3., x)));
    check::<S>(5, xs.clone().map(|x| (x, x.ln_1p(), 1. - x)));

    assert_eq!(
        Polynomial::<_, S>::least_squares_fit(1, [(0., 0.), (0., 1.)].into_iter()),
        None
    );
}

fn test_lagrange<S: Storage<f64>>() {
    // Evaluate the lagrange polynomial at the x coordinates.
    // The error should be close to zero.
    #[track_caller]
    fn check<S: Storage<f64>>(xs: impl Iterator<Item = f64> + Clone, p: Polynomial<f64, S>) {
        let p_test = Polynomial::<_, S>::lagrange(xs.map(|xi| (xi, p.eval(xi)))).unwrap();
        assert!(p_test.coeffs().len() == p.coeffs().len());
        p_test
            .coeffs()
            .into_iter()
            .zip(p.coeffs().into_iter())
            .for_each(|(c_test, c)| assert!((c_test - c).abs() < 1e-9));
    }

    // Squares
    check::<S>(
        [1., 2., 3.].iter().copied(),
        poly_from_slice::<_, S>(&[0., 10.]),
    );
    // Cubes
    check::<S>(
        [-1., 0., 1., 2.].iter().copied(),
        poly_from_slice::<_, S>(&[0., 0., 0., 1.]),
    );
    // Non linear x.
    check::<S>(
        [1., 9., 10., 11.].iter().copied(),
        poly_from_slice::<_, S>(&[-1., 2., -3., 4.]),
    );

    // Test double x failure case.
    assert_eq!(
        Polynomial::<f64, S>::lagrange(
            [1., 9., 9., 11.]
                .iter()
                .copied()
                .zip([1., 2., 3., 4.].iter().copied())
        ),
        None
    );
}

fn test_chebyshev<S>()
where
    S: Storage<f64>,
    S::Provider: StorageProvider<(f64, f64)> + StorageProvider<f64>,
{
    // Construct a Chebyshev approximation for a function
    // and evaulate it at 100 points.
    #[track_caller]
    fn check<F, S>(f: &F, n: usize, xmin: f64, xmax: f64)
    where
        F: Fn(f64) -> f64,
        S: Storage<f64>,
        S::Provider: StorageProvider<(f64, f64)> + StorageProvider<f64>,
    {
        let p = Polynomial::<f64, S>::chebyshev(f, n, xmin, xmax).unwrap();
        for i in 0..=100 {
            let x = xmin + (i as f64) * ((xmax - xmin) / 100.0);
            let diff = (f(x) - p.eval(x)).abs();
            assert!(diff < 1e-4);
        }
    }

    // Approximate some common functions.
    use core::f64::consts::PI;
    check::<_, S>(&f64::sin, 7, -PI / 2., PI / 2.);
    check::<_, S>(&f64::cos, 7, 0., PI / 4.);
    check::<_, S>(&f64::ln, 5, 2., 1.);

    // Test n >= 1 condition
    assert!(Polynomial::<f64, S>::chebyshev(&f64::exp, 0, 0., 1.).is_none());
}

fn test_deriv<S>()
where
    S: Storage<i32>,
{
    let empty: [i32; 0] = [];
    #[track_caller]
    fn check<S: Storage<i32>>(a: &[i32], deriv_a: &[i32]) {
        let a = poly_from_slice::<_, S>(a);
        let deriv_a = poly_from_slice::<_, S>(deriv_a);
        assert_eq!(a.deriv(), deriv_a);
    }

    check::<S>(&empty, &empty);
    check::<S>(&[1], &empty);
    check::<S>(&[1, 3, 5, 7], &[3, 10, 21]);
}

fn test_antideriv<S>()
where
    S: Storage<i32>,
{
    let empty: [i32; 0] = [];
    #[track_caller]
    fn check<S: Storage<i32>>(a: &[i32], antideriv_a: &[i32]) {
        let a = poly_from_slice::<_, S>(a);
        let antideriv_a = poly_from_slice::<_, S>(antideriv_a);
        assert_eq!(a.antideriv(), antideriv_a);
    }

    check::<S>(&empty, &empty);
    check::<S>(&[2], &[0, 2]);
    check::<S>(&[1, 4, 9, 16], &[0, 1, 2, 3, 4]);
}
