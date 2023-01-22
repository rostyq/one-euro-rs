use std::ops::{AddAssign, Neg};

use nalgebra::{SVector, RealField};

#[derive(Clone, Copy, Debug)]
pub struct LowPassFilter<T: RealField, const D: usize>
{
    state: Option<SVector<T, D>>,
}

impl<T: RealField, const D: usize> LowPassFilter<T, D> {
    pub fn new(sample: SVector<T, D>) -> Self {
        Self { state: Some(sample) }
    }

    pub fn filter(&mut self, sample: &SVector<T, D>, alpha: SVector<T, D>) -> SVector<T, D> {
        let output = match &self.state {
            Some(value) => {
                // current_sample * alpha + (1 - alpha) * previous_sample
                sample.component_mul(&alpha) + value.component_mul(&alpha.neg().add_scalar(T::one()))
            },
            None => sample.to_owned(),
        };

        self.state = Some(output.to_owned());
        output
    }

    pub fn filter_mut(&mut self, sample: &mut SVector<T, D>, mut alpha: SVector<T, D>) {
        if let Some(value) = &self.state {
            // current_sample * alpha + (1 - alpha) * previous_sample
            sample.component_mul_assign(&alpha);

            alpha.neg_mut();
            alpha.add_scalar_mut(T::one());

            sample.add_assign(value.component_mul(&alpha));
        };

        self.state = Some(sample.to_owned());
    }

    pub fn get_state(&self) -> Option<&SVector<T, D>> {
        self.state.as_ref()
    }

    pub fn reset(&mut self) {
        self.state = None;
    }
}

impl<T: RealField, const D: usize> Default for LowPassFilter<T, D> {
    fn default() -> Self {
        Self { state: None }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector1;

    use super::*;

    #[test]
    fn update_state_and_reset_on_default() {
        let mut filter = LowPassFilter::<f64, 1>::default();
        assert_eq!(filter.state, None);
        assert_eq!(filter.get_state(), None);

        let sample = Vector1::new(1.0);
        filter.filter(&sample, Vector1::new(0.0));

        assert_eq!(filter.state.as_ref().unwrap(), &sample);
        assert_eq!(filter.get_state().unwrap(), &sample);

        filter.reset();

        assert_eq!(filter.state, None);
        assert_eq!(filter.get_state(), None);
    }
}
