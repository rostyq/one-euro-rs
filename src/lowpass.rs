use std::ops::Neg;

use nalgebra::{RealField, SVector};

/// Low-pass filter state.
#[derive(Clone, Copy, Debug)]
pub struct LowPassFilter<T: RealField, const D: usize> {
    pub state: SVector<T, D>,
}

impl<T: RealField, const D: usize> LowPassFilter<T, D> {
    /// Initializes low-pass filter state.
    pub fn new(state: SVector<T, D>) -> Self {
        Self { state }
    }

    /// Updates with [`filter`] function using its state as `previous` value.
    pub fn update(&mut self, sample: &SVector<T, D>, alpha: &SVector<T, D>) {
        self.state = filter(sample, &self.state, alpha);
    }
}

/// Filter 1D signal as follows
///
///     current * alpha + (1 - alpha) * previous
///
/// where:
///
/// * `current` - Raw signal
/// * `previous` - Previous signal
/// * `alpha` - Smoothing factor
#[inline]
pub fn filter<T: RealField, const D: usize>(
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
