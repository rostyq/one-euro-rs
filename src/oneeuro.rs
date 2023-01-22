use nalgebra::{SVector, RealField};

use crate::LowPassFilter;

#[derive(Debug)]
pub struct OneEuroFilter<T: RealField, const D: usize> {
    rate: T,
    cutoff_slope: T,
    min_cutoff: SVector<T, D>,
    derivate_cutoff: SVector<T, D>,
    sample_filter: LowPassFilter<T, D>,
    derivate_filter: LowPassFilter<T, D>,
    state: Option<SVector<T, D>>,
}

impl<T: RealField, const D: usize> OneEuroFilter<T, D> {
    pub fn new(rate: T, cutoff_slope: T, min_cutoff: SVector<T, D>, derivate_cutoff: SVector<T, D>) -> Self {
        assert!(rate > T::zero());

        for value in &min_cutoff {
            assert!(*value > T::zero());
        }

        for value in &derivate_cutoff {
            assert!(*value > T::zero());
        }

        Self {
            rate,
            cutoff_slope,
            min_cutoff,
            derivate_cutoff,
            sample_filter: LowPassFilter::<T, D>::default(),
            derivate_filter: LowPassFilter::<T, D>::default(),
            state: None,
        }
    }

    pub fn builder() -> OneEuroFilterBuilder<T, D> {
        OneEuroFilterBuilder::<T, D>::default()
    }

    pub fn get_rate(&self) -> T {
        self.rate.to_owned()
    }

    pub fn set_rate(&mut self, value: T) {
        assert!(value > T::zero());
        self.rate = value;
    }

    pub fn set_cutoff_slope(&mut self, value: T) {
        assert!(value > T::zero());
        self.cutoff_slope = value;
    }

    pub fn get_cutoff_slope(&self) -> T {
        self.cutoff_slope.to_owned()
    }

    pub fn filter(&mut self, sample: &SVector<T, D>) -> SVector<T, D> {
        let mut output = sample.clone();
        self.filter_mut(&mut output);
        output
    }

    pub fn filter_mut(&mut self, sample: &mut SVector<T, D>) {
        let mut derivate = match self.state.as_ref() {
            Some(value) => (sample.to_owned() - value).scale(self.rate.to_owned()),
            None => SVector::<T, D>::zeros()
        };

        self.derivate_filter.filter_mut(&mut derivate, self.alpha(&self.derivate_cutoff));

        let cutoff = self.min_cutoff.to_owned() + derivate.abs().scale(self.cutoff_slope.to_owned());

        self.state = Some(sample.clone());

        self.sample_filter.filter_mut(sample, self.alpha(&cutoff));
    }

    fn alpha(&self, cutoff: &SVector<T, D>) -> SVector<T, D> {
        let tau = cutoff.scale(T::two_pi());

        let k = SVector::<T, D>::repeat(self.rate.to_owned())
            .component_div(&tau);

        SVector::<T, D>::repeat(T::one()).component_div(&k.add_scalar(T::one()))
    }

    pub fn get_state(&self) -> Option<&SVector<T, D>> {
        self.state.as_ref()
    }

    pub fn reset(&mut self) {
        self.state = None;
        self.sample_filter.reset();
        self.derivate_filter.reset();
    }
}

#[derive(Debug)]
pub struct OneEuroFilterBuilder<T: RealField, const D: usize> {
    rate: Option<T>,
    cutoff_slope: Option<T>,
    min_cutoff: Option<SVector<T, D>>,
    derivate_cutoff: Option<SVector<T, D>>,
}

impl<T: RealField, const D: usize> OneEuroFilterBuilder<T, D> {
    pub fn build(self) -> OneEuroFilter<T, D> {
        OneEuroFilter::new(
            self.rate.unwrap_or(T::one()),
            self.cutoff_slope.unwrap_or(T::zero()),
            self.min_cutoff.unwrap_or(SVector::<T, D>::repeat(T::one())),
            self.derivate_cutoff.unwrap_or(SVector::<T, D>::repeat(T::one())),
        )
    }

    pub fn with_rate(mut self, value: T) -> Self {
        self.rate = Some(value);
        self
    }

    pub fn with_cutoff_slope(mut self, value: T) -> Self {
        self.cutoff_slope = Some(value);
        self
    }

    pub fn with_min_cutoff(mut self, value: T) -> Self {
        self.min_cutoff = Some(SVector::<T, D>::repeat(value));
        self
    }

    pub fn with_derivate_cutoff(mut self, value: T) -> Self {
        self.derivate_cutoff = Some(SVector::<T, D>::repeat(value));
        self
    }
}

impl<T: RealField, const D: usize> Default for OneEuroFilterBuilder<T, D> {
    fn default() -> Self {
        Self {
            rate: None,
            cutoff_slope: None,
            min_cutoff: None,
            derivate_cutoff: None,
        }
    }
}