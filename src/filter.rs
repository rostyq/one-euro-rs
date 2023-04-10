use nalgebra::{RealField, SVector};

use crate::alpha::get_alpha_unchecked;
use crate::state::OneEuroState;

/// 1€ Filter parameters.
#[derive(Debug, Clone, Copy)]
pub struct OneEuroFilter<T: RealField> {
    beta: T,
    dcutoff: T,
    mincutoff: T,
}

impl<T: RealField> OneEuroFilter<T> {
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

    /// Set derivate frequency cutoff.
    #[inline]
    pub fn set_dcutoff(&mut self, value: T) {
        assert_positive!(value);
        self.dcutoff = value;
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
    pub fn filter<const D: usize>(
        &self,
        state: &mut OneEuroState<T, D>,
        raw: &SVector<T, D>,
        rate: T,
    ) {
        unsafe {
            state.update_unchecked(
                raw,
                &self.get_alpha(rate.to_owned()),
                rate,
                self.mincutoff(),
                self.beta(),
            )
        };
    }

    /// Filter multiple states using current parameters.
    #[inline]
    pub fn filter_slice<const D: usize>(
        &self,
        states: &mut [OneEuroState<T, D>],
        raws: &[SVector<T, D>],
        rate: T,
    ) {
        let alpha = self.get_alpha::<D>(rate.to_owned());

        for (state, raw) in states.iter_mut().zip(raws) {
            unsafe {
                state.update_unchecked(raw, &alpha, rate.to_owned(), self.mincutoff(), self.beta())
            };
        }
    }

    #[inline]
    pub fn get_alpha<const D: usize>(&self, rate: T) -> SVector<T, D> {
        SVector::<T, D>::repeat(unsafe { get_alpha_unchecked(rate, self.dcutoff()) })
    }
}

impl<T: RealField> Default for OneEuroFilter<T> {
    /// Each setable parameter is 1.
    #[inline]
    fn default() -> Self {
        Self {
            dcutoff: T::one(),
            mincutoff: T::one(),
            beta: T::one(),
        }
    }
}
