use std::ops::Neg;

use nalgebra::{RealField, SVector};

macro_rules! assert_alpha {
    ($alpha:ident, $T:ty) => {
        for value in $alpha.iter() {
            if *value < T::zero() || *value > T::one() {
                panic!("`alpha` value should be in range [0, 1]");
            }
        }
    };
}

/// Low-pass filter state.
#[derive(Clone, Copy, Debug)]
pub struct LowPassState<T: RealField, const D: usize> (SVector<T, D>);

impl<T: RealField, const D: usize> LowPassState<T, D> {
    /// Initialize low-pass filter state.
    pub fn new(state: SVector<T, D>) -> Self {
        Self(state)
    }

    /// Update state using [`filter`] function.
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
        self.0 = filter(sample, &self.0, alpha);
    }

    /// Same as [`LowPassState::update`] but without alpha check.
    /// 
    /// # Safety
    /// 
    /// See [`filter_unchecked`].
    pub unsafe fn update_unchecked(&mut self, sample: &SVector<T, D>, alpha: &SVector<T, D>) {
        self.0 = filter_unchecked(sample, &self.0, alpha);
    }

    /// Current state.
    pub fn data(&self) -> &SVector<T, D> {
        &self.0
    }
}

impl<T: RealField, const D: usize> AsRef<SVector<T, D>> for LowPassState<T, D> {
    fn as_ref(&self) -> &SVector<T, D> {
        self.data()
    }
}

impl<T: RealField, const D: usize> From<SVector<T, D>> for LowPassState<T, D> {
    fn from(value: SVector<T, D>) -> Self {
        LowPassState::new(value)
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
    fn test_lowpass_state() {
        let mut state = LowPassState::new(Vector1::new(1.0));

        state.update(&[2.0].into(), &[0.0].into());
        assert_abs_diff_eq!(state.0, [1.0].into());

        state.update(&[2.0].into(), &[1.0].into());
        assert_abs_diff_eq!(state.0, [2.0].into());

        state.update(&[3.0].into(), &[0.5].into());
        assert_abs_diff_eq!(state.0, [2.5].into());
    }
}
