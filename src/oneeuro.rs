use nalgebra::{RealField, SVector};

use crate::lowpass::LowPassFilter;

#[derive(Clone, Copy, Debug)]
pub struct OneEuroFilter<T: RealField, const D: usize> {
    raw_state: SVector<T, D>,
    sample_filter: LowPassFilter<T, D>,
    derivate_filter: LowPassFilter<T, D>,
}

impl<T: RealField, const D: usize> OneEuroFilter<T, D> {
    pub fn new(state: SVector<T, D>) -> Self {
        Self {
            raw_state: state.to_owned(),
            sample_filter: LowPassFilter::new(state.to_owned()),
            derivate_filter: LowPassFilter::new(SVector::zeros()),
        }
    }

    #[inline]
    fn update_derivate(&mut self, sample: &SVector<T, D>, alpha: &SVector<T, D>, scale: T) {
        self.derivate_filter
            .update(&(sample - &self.raw_state).scale(scale), alpha);
    }

    #[inline]
    fn update_sample(&mut self, sample: &SVector<T, D>, alpha: &SVector<T, D>) {
        self.sample_filter.update(sample, alpha);
    }

    #[inline]
    fn derivate(&self) -> &SVector<T, D> {
        &self.derivate_filter.state
    }

    #[inline]
    fn get_frequency_cutoff(&self, intercept: T, slope: T) -> SVector<T, D> {
        self.derivate().abs().scale(slope).add_scalar(intercept)
    }

    #[inline]
    pub fn update(
        &mut self,
        sample: &SVector<T, D>,
        alpha: &SVector<T, D>,
        rate: T,
        mincutoff: T,
        beta: T,
    ) {
        self.update_derivate(sample, alpha, rate.to_owned());
        let alpha = self
            .get_frequency_cutoff(mincutoff, beta)
            .map(|v| Self::get_alpha(rate.to_owned(), v));
        self.update_sample(sample, &alpha);
        self.raw_state = sample.to_owned();
    }

    #[inline]
    pub fn update_with_config(&mut self, sample: &SVector<T, D>, config: &OneEuroConfig<T, D>) {
        self.update(
            sample,
            &config.alpha,
            config.rate(),
            config.mincutoff(),
            config.beta(),
        );
    }

    #[inline]
    fn get_alpha(rate: T, cutoff: T) -> T {
        T::one() / (T::one() + rate / (T::two_pi() * cutoff))
    }

    #[inline]
    pub fn state(&self) -> &SVector<T, D> {
        &self.sample_filter.state
    }

    #[inline]
    pub fn raw_state(&self) -> &SVector<T, D> {
        &self.raw_state
    }

    #[inline]
    pub fn derivative_state(&self) -> &SVector<T, D> {
        &self.derivate_filter.state
    }
}

#[derive(Debug)]
pub struct OneEuroConfig<T: RealField, const D: usize> {
    rate: T,
    beta: T,
    mincutoff: T,
    alpha: SVector<T, D>,
}

impl<T: RealField, const D: usize> OneEuroConfig<T, D> {
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
    }

    pub fn set_mincutoff(&mut self, value: T) {
        assert!(
            value > T::zero(),
            "Minimum frequency cutoff should be greater than zero."
        );
        self.mincutoff = value.to_owned();
        self.alpha = Self::get_alpha(self.rate(), value.to_owned());
    }

    fn get_alpha(rate: T, cutoff: T) -> SVector<T, D> {
        SVector::<T, D>::repeat(cutoff)
            .map(|v| OneEuroFilter::<T, D>::get_alpha(rate.to_owned(), v))
    }
}
