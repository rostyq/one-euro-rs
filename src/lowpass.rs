use std::ops::Neg;

use nalgebra::{RealField, SVector};

macro_rules! assert_alpha {
    ($alpha:ident, $T:ty) => {
        for value in $alpha.iter() {
            assert!(*value >= T::zero());
            assert!(*value <= T::one());
        }
    };
}

/// Low-pass filter state.
#[derive(Clone, Copy, Debug)]
pub struct LowPassFilter<T: RealField, const D: usize> {
    state: SVector<T, D>,
}

impl<T: RealField, const D: usize> LowPassFilter<T, D> {
    /// Initialize low-pass filter state.
    pub fn new(state: SVector<T, D>) -> Self {
        Self { state }
    }

    /// Filter new sample (see [`filter`] function) and set result as current state.
    /// 
    /// # Arguments
    /// 
    /// * `sample` - new signal
    /// * `alpha` - smoothing factor
    /// 
    /// # Panics
    /// 
    /// See [`filter`].
    pub fn update(&mut self, sample: &SVector<T, D>, alpha: &SVector<T, D>) {
        self.state = filter(sample, &self.state, alpha);
    }

    /// Same as [`LowPassFilter::update`] but without alpha check.
    /// 
    /// # Safety
    /// 
    /// See [`filter_unchecked`].
    pub unsafe fn update_unchecked(&mut self, sample: &SVector<T, D>, alpha: &SVector<T, D>) {
        self.state = filter_unchecked(sample, &self.state, alpha);
    }

    /// Current state.
    pub fn state(&self) -> &SVector<T, D> {
        &self.state
    }
}

impl<T: RealField, const D: usize> AsRef<SVector<T, D>> for LowPassFilter<T, D> {
    fn as_ref(&self) -> &SVector<T, D> {
        self.state()
    }
}

/// Filter signal as follows
///
///     current * alpha + (1 - alpha) * previous
///
/// # Arguments
///
/// * `current` - raw signal
/// * `previous` - previous signal
/// * `alpha` - smoothing factor
/// 
/// # Panics
/// 
/// This function will panic if any value in `alpha` is out of \[0, 1\] range.
#[inline]
pub fn filter<T: RealField, const D: usize>(
    current: &SVector<T, D>,
    previous: &SVector<T, D>,
    alpha: &SVector<T, D>,
) -> SVector<T, D> {
    assert_alpha!(alpha, T);
    unsafe { filter_unchecked(current, previous, alpha) }
}

/// Same as [`filter`] but without smoothing factor (`alpha`) check.
/// 
/// # Safety
/// 
/// Each value in `alpha` should be in \[0, 1\] range.
#[inline]
pub unsafe fn filter_unchecked<T: RealField, const D: usize>(
    current: &SVector<T, D>,
    previous: &SVector<T, D>,
    alpha: &SVector<T, D>,
) -> SVector<T, D> {
    current.component_mul(alpha) + previous.component_mul(&(alpha.neg().add_scalar(T::one())))
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector1;

    use super::*;

    #[test]
    fn test_lowpass_filter() {
        let previous = Vector1::new(1.0);
        let current = Vector1::new(2.0);

        assert_abs_diff_eq!(filter(&current, &previous, &[0.0].into()), [1.0].into());
        assert_abs_diff_eq!(filter(&current, &previous, &[1.0].into()), [2.0].into());
        assert_abs_diff_eq!(filter(&current, &previous, &[0.5].into()), [1.5].into());
    }

    #[test]
    fn test_lowpass_filter_state() {
        let mut filter = LowPassFilter::new(Vector1::new(1.0));

        filter.update(&[2.0].into(), &[0.0].into());
        assert_abs_diff_eq!(filter.state, [1.0].into());

        filter.update(&[2.0].into(), &[1.0].into());
        assert_abs_diff_eq!(filter.state, [2.0].into());

        filter.update(&[3.0].into(), &[0.5].into());
        assert_abs_diff_eq!(filter.state, [2.5].into());
    }
}
