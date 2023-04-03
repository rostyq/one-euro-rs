use nalgebra::{RealField, SVector};

use crate::lowpass::LowPassState;
use crate::alpha::{get_alpha, get_alpha_unchecked};

/// 1€ Filter state.
#[derive(Clone, Copy, Debug)]
pub struct OneEuroState<T: RealField, const D: usize> {
    raw: SVector<T, D>,
    filtered: LowPassState<T, D>,
    derivate: LowPassState<T, D>,
}

impl<T: RealField, const D: usize> OneEuroState<T, D> {
    /// Initializes 1€ Filter state.
    #[inline]
    pub fn new(state: SVector<T, D>) -> Self {
        Self {
            raw: state.to_owned(),
            filtered: state.to_owned().into(),
            derivate: SVector::<T, D>::zeros().into(),
        }
    }

    /// Current derivate.
    #[inline]
    fn derivate(&self) -> &SVector<T, D> {
        self.derivate.as_ref()
    }

    /// Current state.
    #[inline]
    pub fn data(&self) -> &SVector<T, D> {
        self.filtered.as_ref()
    }

    /// Current raw (not filtered) state.
    #[inline]
    pub fn raw(&self) -> &SVector<T, D> {
        &self.raw
    }

    /// Calculate frequency cutoff:
    ///
    /// `intercept + slope * derivate`
    ///
    /// where `derivate` is value from [`derivate`].
    ///
    /// # Arguments
    ///     
    /// * `intercept` - minimal cutoff
    /// * `slope` - cutoff coefficient
    #[inline]
    fn get_cutoff(&self, intercept: T, slope: T) -> SVector<T, D> {
        self.derivate().abs().scale(slope).add_scalar(intercept)
    }

    /// Update state.
    ///
    /// # Arguments
    ///
    /// * `raw` - new unfiltered signal
    /// * `alpha` - smoothing factor for raw signal derivate
    /// * `rate` - signal sampling frequency
    /// * `mincutoff` - minimal value for frequency cutoff
    /// * `beta` - slope for frequency cutoff
    ///
    /// # Panics
    ///
    /// This function panics if:
    ///
    /// * any value in `alpha` is not in \(0, 1\] range
    /// * `rate` or `mincutoff` are negative or zero
    /// * `beta` is negative
    #[inline]
    pub fn update(
        &mut self,
        raw: &SVector<T, D>,
        alpha: &SVector<T, D>,
        rate: T,
        mincutoff: T,
        beta: T,
    ) {
        self.derivate
            .update(&(raw - &self.raw).scale(rate.to_owned()), alpha);

        let alpha = self
            .get_cutoff(mincutoff, beta)
            .map(|v| get_alpha(rate.to_owned(), v));

        // get_alpha is checked
        unsafe { self.filtered.update_unchecked(raw, &alpha) }

        self.raw = raw.to_owned();
    }

    /// Same as [`OneEuroState::update`] but without safety checks.
    ///
    /// # Safety
    ///
    /// Calculation is valid if:
    ///
    /// * each value in `alpha` is in \(0, 1\] range
    /// * `rate` and `mincutoff` are positive
    /// * `beta` is not negative
    #[inline]
    pub unsafe fn update_unchecked(
        &mut self,
        raw: &SVector<T, D>,
        alpha: &SVector<T, D>,
        rate: T,
        mincutoff: T,
        beta: T,
    ) {
        self.derivate
            .update_unchecked(&(raw - &self.raw).scale(rate.to_owned()), alpha);

        let alpha = self
            .get_cutoff(mincutoff, beta)
            .map(|v| get_alpha_unchecked(rate.to_owned(), v));

        self.filtered.update_unchecked(raw, &alpha);

        self.raw = raw.to_owned();
    }
}

impl<T: RealField, const D: usize> AsRef<SVector<T, D>> for OneEuroState<T, D> {
    #[inline]
    fn as_ref(&self) -> &SVector<T, D> {
        self.data()
    }
}

impl<T: RealField, const D: usize> AsRef<[T; D]> for OneEuroState<T, D> {
    #[inline]
    fn as_ref(&self) -> &[T; D] {
        &self.data().data.0[0]
    }
}

impl<T: RealField, const D: usize> From<SVector<T, D>> for OneEuroState<T, D> {
    #[inline]
    fn from(value: SVector<T, D>) -> Self {
        Self::new(value)
    }
}

impl<T: RealField, const D: usize> From<[T; D]> for OneEuroState<T, D> {
    #[inline]
    fn from(value: [T; D]) -> Self {
        Self::new(value.into())
    }
}
