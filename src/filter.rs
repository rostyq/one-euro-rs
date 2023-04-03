use nalgebra::{RealField, SVector};

use crate::state::OneEuroState;
use crate::alpha::get_alpha_unchecked;

/// 1€ Filter parameters.
#[derive(Debug)]
pub struct OneEuroFilter<T: RealField, const D: usize> {
    rate: T,
    beta: T,
    dcutoff: T,
    mincutoff: T,
    alpha: SVector<T, D>,
}

impl<T: RealField, const D: usize> OneEuroFilter<T, D> {
    /// Sampling frequency.
    #[inline]
    pub fn rate(&self) -> T {
        self.rate.to_owned()
    }

    /// Slope for frequency cutoff.
    #[inline]
    pub fn beta(&self) -> T {
        self.beta.to_owned()
    }

    /// Derivative frequency cutoff.
    #[inline]
    pub fn dcutoff(&self) -> T {
        self.dcutoff.to_owned()
    }

    /// Minimum value for frequency cutoff.
    #[inline]
    pub fn mincutoff(&self) -> T {
        self.mincutoff.to_owned()
    }

    /// Derivative smoothing factor.
    #[inline]
    pub fn alpha(&self) -> &SVector<T, D> {
        &self.alpha
    }

    /// Set sampling frequency.
    #[inline]
    pub fn set_rate(&mut self, value: T) {
        assert_positive!(value);
        self.rate = value;
        self.alpha = Self::get_alpha(self.rate(), self.mincutoff());
    }

    /// Set derivate frequency cutoff.
    #[inline]
    pub fn set_dcutoff(&mut self, value: T) {
        assert_positive!(value);
        self.dcutoff = value;
        self.alpha = Self::get_alpha(self.rate(), self.dcutoff());
    }

    /// Set minimum value for frequency cutoff.
    #[inline]
    pub fn set_mincutoff(&mut self, value: T) {
        assert_positive!(value);
        self.mincutoff = value.to_owned();
    }

    /// Set slope for frequency cutoff.
    #[inline]
    pub fn set_beta(&mut self, value: T) {
        assert!(value >= T::zero(), "beta should be zero or positive.");
        self.beta = value.to_owned();
    }

    /// Filter state using current parameters.
    #[inline]
    pub fn filter(&self, state: &mut OneEuroState<T, D>, raw: &SVector<T, D>) {
        unsafe {
            state.update_unchecked(
                raw,
                self.alpha(),
                self.rate(),
                self.mincutoff(),
                self.beta(),
            )
        };
    }

    #[inline]
    fn get_alpha(rate: T, cutoff: T) -> SVector<T, D> {
        SVector::<T, D>::repeat(cutoff).map(|v| unsafe { get_alpha_unchecked(rate.to_owned(), v) })
    }
}

impl<T: RealField, const D: usize> Default for OneEuroFilter<T, D> {
    /// Each setable parameter is 1.
    #[inline]
    fn default() -> Self {
        Self {
            rate: T::one(),
            dcutoff: T::one(),
            mincutoff: T::one(),
            beta: T::one(),
            alpha: Self::get_alpha(T::one(), T::one()),
        }
    }
}