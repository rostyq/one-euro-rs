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
    pub fn rate(&self) -> T {
        self.rate.to_owned()
    }

    pub fn beta(&self) -> T {
        self.beta.to_owned()
    }

    pub fn dcutoff(&self) -> T {
        self.dcutoff.to_owned()
    }

    pub fn mincutoff(&self) -> T {
        self.mincutoff.to_owned()
    }

    pub fn alpha(&self) -> &SVector<T, D> {
        &self.alpha
    }

    pub fn set_rate(&mut self, value: T) {
        assert!(
            value > T::zero(),
            "Sampling rate should be greater than zero."
        );
        self.rate = value;
        self.alpha = Self::get_alpha(self.rate(), self.mincutoff());
    }

    pub fn set_cutoff(&mut self, value: T) {
        assert!(
            value >= T::zero(),
            "Derivative cutoff should be positive or zero."
        );
        self.dcutoff = value.to_owned();
        self.alpha = Self::get_alpha(self.rate(), self.dcutoff());
    }

    pub fn set_mincutoff(&mut self, value: T) {
        assert!(
            value >= T::zero(),
            "Minimal frequency cutoff should be positive or zero."
        );
        self.mincutoff = value.to_owned();
    }

    #[inline]
    pub fn filter(&self, state: &mut OneEuroState<T, D>, sample: &SVector<T, D>) {
        unsafe {
            state.update_unchecked(
                sample,
                self.alpha(),
                self.rate(),
                self.mincutoff(),
                self.beta(),
            )
        };
    }

    pub fn set_beta(&mut self, value: T) {
        assert!(value >= T::zero(), "beta should be positive or zero.");
        self.beta = value.to_owned();
    }

    fn get_alpha(rate: T, cutoff: T) -> SVector<T, D> {
        SVector::<T, D>::repeat(cutoff).map(|v| unsafe { get_alpha_unchecked(rate.to_owned(), v) })
    }
}

impl<T: RealField, const D: usize> Default for OneEuroFilter<T, D> {
    fn default() -> Self {
        Self {
            rate: T::one(),
            dcutoff: T::one(),
            mincutoff: T::one(),
            beta: T::zero(),
            alpha: Self::get_alpha(T::one(), T::one()),
        }
    }
}