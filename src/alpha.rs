use nalgebra::RealField;

macro_rules! assert_alpha {
    ($alpha:ident) => {
        for value in $alpha.iter() {
            assert_positive!(*value, alpha);
            assert!(*value <= T::one(), "alpha value should be less than one.");
        }
    };
}

macro_rules! assert_positive {
    ($value:expr) => {
        assert_positive!($value, $value);
    };

    ($value:expr, $name:pat) => {
        assert!(
            $value > T::zero(),
            stringify!($name should be greater than zero)
        );
    };
}

/// Calculate smoothing factor as follows
///
/// `(1 + rate / (2 * π * cutoff)) ^ (-1)`
///
/// # Arguments
///
/// * `rate` - sampling frequency
/// * `cutoff` - frequency cutoff
///
/// # Panics
///
/// This functions panics if:
///
/// * `rate` or `cutoff` are negative or zero;
/// * calculated value is not finite.
#[inline]
pub fn get_alpha<T: RealField>(rate: T, cutoff: T) -> T {
    assert_positive!(rate);
    assert_positive!(cutoff);

    unsafe { get_alpha_unchecked(rate, cutoff) }
}

/// Same as [`get_alpha`] function but without argument checks.
#[inline]
pub unsafe fn get_alpha_unchecked<T: RealField>(rate: T, cutoff: T) -> T {
    (T::one() + rate / (T::two_pi() * cutoff)).recip()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_unit_alpha() {
        assert_abs_diff_eq!(get_alpha(1.0, f64::INFINITY), 1.0);
    }

    #[test]
    fn test_half_alpha() {
        assert_abs_diff_eq!(get_alpha(1.0, (2.0 * std::f64::consts::PI).recip()), 0.5);
    }

    #[test]
    fn test_epsilon_alpha() {
        assert_abs_diff_eq!(get_alpha(f64::INFINITY, f64::EPSILON), f64::EPSILON);
    }

    #[test]
    #[should_panic]
    fn test_zero_rate() {
        get_alpha(0.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_zero_cutoff() {
        get_alpha(1.0, 0.0);
    }

    #[test]
    #[should_panic]
    fn test_negative_rate() {
        get_alpha(-f64::EPSILON, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_negative_cutoff() {
        get_alpha(1.0, -f64::EPSILON);
    }
}