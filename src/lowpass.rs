use std::ops::AddAssign;

use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, OVector, RealField};

#[derive(Debug)]
pub struct LowPassFilter<T, D>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: Allocator<T, D>
{
    state: Option<OVector<T, D>>,
}

impl<T, D> LowPassFilter<T, D>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: Allocator<T, D>
{
    pub fn new(sample: OVector<T, D>) -> Self {
        Self { state: Some(sample) }
    }

    pub fn filter(&mut self, sample: &OVector<T, D>, alpha: OVector<T, D>) -> OVector<T, D> {
        let mut output = sample.clone();
        self.filter_mut(&mut output, alpha);
        output
    }

    pub fn filter_mut(&mut self, sample: &mut OVector<T, D>, mut alpha: OVector<T, D>) {
        if let Some(value) = &self.state {
            // current_sample * alpha + (1 - alpha) * previous_sample
            sample.component_mul_assign(&alpha);

            alpha.neg_mut();
            alpha.add_scalar_mut(T::one());

            sample.add_assign(value.component_mul(&alpha));
        };

        self.state = Some(sample.clone());
    }

    pub fn get_state(&self) -> Option<&OVector<T, D>> {
        self.state.as_ref()
    }

    pub fn reset(&mut self) {
        self.state = None;
    }
}

impl<T, D> Default for LowPassFilter<T, D>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: Allocator<T, D>
{
    fn default() -> Self {
        Self { state: None }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::U1;
    use nalgebra::Vector1;

    use super::*;

    #[test]
    fn update_state_and_reset_on_default() {
        let mut filter = LowPassFilter::<f64, U1>::default();
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
