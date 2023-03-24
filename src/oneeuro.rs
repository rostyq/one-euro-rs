use nalgebra::{RealField, SVector};

use crate::lowpass::LowPassState;

/// 1€ Filter state.
#[derive(Clone, Copy, Debug)]
pub struct OneEuroState<T: RealField, const D: usize> {
    raw: SVector<T, D>,
    sample: LowPassState<T, D>,
    derivate: LowPassState<T, D>,
}

impl<T: RealField, const D: usize> OneEuroState<T, D> {
    /// Initializes 1€ Filter state.
    pub fn new(state: SVector<T, D>) -> Self {
        Self {
            raw: state.to_owned(),
            sample: state.to_owned().into(),
            derivate: SVector::<T, D>::zeros().into()
        }
    }

    /// Current derivate.
    #[inline]
    fn derivate(&self) -> &SVector<T, D> {
        &self.derivate.data()
    }

    /// Current state.
    #[inline]
    pub fn data(&self) -> &SVector<T, D> {
        &self.sample.data()
    }

    /// Current raw (not filtered) state.
    #[inline]
    pub fn raw(&self) -> &SVector<T, D> {
        &self.raw
    }

    /// Calculate frequency cutoff:
    /// 
    ///     intercept + slope * derivate
    /// 
    /// where `derivate` is value from [`derivate`].
    /// 
    /// # Arguments
    ///     
    /// * `intercept` - minimal frequency cutoff
    /// * `slope` - coefficient (`slope`)
    #[inline]
    fn get_frequency_cutoff(&self, intercept: T, slope: T) -> SVector<T, D> {
        self.derivate().abs().scale(slope).add_scalar(intercept)
    }

    /// Update state.
    /// 
    /// # Arguments
    /// 
    /// * `sample` - new signal
    /// * `alpha` - smoothing factor for derivate
    /// * `rate` - sampling rate
    /// * `mincutoff` - minimal value for frequency cutoff
    /// * `beta` - slope for frequency cutoff
    /// 
    /// # Panics
    /// 
    /// This function panics if:
    /// 
    /// * any value in `alpha` is not in \[0, 1\] range
    /// * `rate` is negative or zero
    /// * any value of `mincutoff` or `beta` is negative
    #[inline]
    pub fn update(
        &mut self,
        sample: &SVector<T, D>,
        alpha: &SVector<T, D>,
        rate: T,
        mincutoff: T,
        beta: T,
    ) {
        self.derivate.update(&(sample - &self.raw).scale(rate.to_owned()), alpha);

        let alpha = self
            .get_frequency_cutoff(mincutoff, beta)
            .map(|v| get_alpha(rate.to_owned(), v));

        // get_alpha is checked
        unsafe { self.sample.update_unchecked(sample, &alpha) }

        self.raw = sample.to_owned();
    }

    /// Same as [`OneEuroState::update`] but without safety checks.
    /// 
    /// # Safety
    /// 
    /// Calculation is valid if:
    /// 
    /// * each value in `alpha` is in \[0, 1\] range
    /// * `rate` is greater than zero
    /// * each value of `mincutoff` or `beta` is positive or zero
    #[inline]
    pub unsafe fn update_unchecked(
        &mut self,
        sample: &SVector<T, D>,
        alpha: &SVector<T, D>,
        rate: T,
        mincutoff: T,
        beta: T,
    ) {
        self.derivate.update_unchecked(&(sample - &self.raw).scale(rate.to_owned()), alpha);

        let alpha = self
            .get_frequency_cutoff(mincutoff, beta)
            .map(|v| get_alpha_unchecked(rate.to_owned(), v));

        self.sample.update_unchecked(sample, &alpha);

        self.raw = sample.to_owned();
    }

}

impl<T: RealField, const D: usize> AsRef<SVector<T, D>> for OneEuroState<T, D> {
    fn as_ref(&self) -> &SVector<T, D> {
        self.data()
    }
}

impl<T: RealField, const D: usize> From<SVector<T, D>> for OneEuroState<T, D> {
    fn from(value: SVector<T, D>) -> Self {
        Self::new(value)
    }
}

#[derive(Debug)]
pub struct OneEuroFilter<T: RealField, const D: usize> {
    rate: T,
    beta: T,
    mincutoff: T,
    alpha: SVector<T, D>,
}

impl<T: RealField, const D: usize> OneEuroFilter<T, D> {
    pub fn new(rate: T, mincutoff: T, beta: T) -> Self {
        Self {
            beta,
            rate: rate.to_owned(),
            mincutoff: mincutoff.to_owned(),
            alpha: Self::get_alpha(rate.to_owned(), mincutoff.to_owned()),
        }
    }

    pub fn rate(&self) -> T {
        self.rate.to_owned()
    }

    pub fn beta(&self) -> T {
        self.beta.to_owned()
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

    pub fn set_mincutoff(&mut self, value: T) {
        assert!(
            value >= T::zero(),
            "Minimal frequency cutoff should be positive or zero."
        );
        self.mincutoff = value.to_owned();
        self.alpha = Self::get_alpha(self.rate(), self.mincutoff());
    }

    pub fn filter(&self, state: &mut OneEuroState<T, D>, sample: &SVector<T, D>) {
        unsafe { state.update_unchecked(sample, self.alpha(), self.rate(), self.mincutoff(), self.beta()) };
    }

    pub fn set_beta(&mut self, value: T) {
        assert!(
            value >= T::zero(),
            "beta should be positive or zero."
        );
        self.beta = value.to_owned();
    }

    fn get_alpha(rate: T, cutoff: T) -> SVector<T, D> {
        SVector::<T, D>::repeat(cutoff)
            .map(|v| get_alpha(rate.to_owned(), v))
    }
}

/// Calculate smoothing factor as follows
/// 
///     (1 + rate / (2 * π * cutoff)) ^ (-1)
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
fn get_alpha<T: RealField>(rate: T, cutoff: T) -> T {
    assert!(rate > T::zero(), "`rate` should be greater than zero." );
    assert!(cutoff >= T::zero(), "`cutoff` should be zero or positive value." );

    unsafe { get_alpha_unchecked(rate, cutoff) }
}

#[inline]
unsafe fn get_alpha_unchecked<T: RealField>(rate: T, cutoff: T) -> T {
    T::one() / (T::one() + rate / (T::two_pi() * cutoff))
}