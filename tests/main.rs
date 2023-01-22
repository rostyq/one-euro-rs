#[macro_use]
extern crate approx;

#[cfg(test)]
mod tests {

    use std::{fs::File, time::Duration, fmt::Debug};

    use nalgebra::Point2;
    use one_euro::OneEuroFilter;
    use serde::Deserialize;

    #[derive(Debug)]
    struct Entry {
        pub timestamp: Duration,
        #[allow(dead_code)]
        pub origin: Point2<f64>,
        pub noisy: Point2<f64>,
        pub filtered: Point2<f64>,
    }

    #[derive(Debug, Deserialize)]
    struct Record {
        pub timestamp: f64,

        pub origin_x: f64,
        pub origin_y: f64,

        pub noisy_x: f64,
        pub noisy_y: f64,

        pub filtered_x: f64,
        pub filtered_y: f64,
    }

    impl From<Record> for Entry {
        fn from(value: Record) -> Self {
            Self {
                timestamp: Duration::from_secs_f64(value.timestamp),
                origin: Point2::new(value.origin_x, value.origin_y),
                noisy: Point2::new(value.noisy_x, value.noisy_y),
                filtered: Point2::new(value.filtered_x, value.filtered_y),
            }
        }
    }

    #[test]
    fn signal_2d() {
        let file = File::open("./assets/signal.csv")
            .expect("Cannot open file for signal data.");

        let mut reader = csv::Reader::from_reader(file);
        let mut filter = OneEuroFilter::<f64, 2>::builder()
            .with_rate(60.0)
            .with_cutoff_slope(0.007)
            .with_min_cutoff(1.0)
            .with_derivate_cutoff(1.0)
            .build();

        let mut timestamp: Option<Duration> = None;

        for result in reader.deserialize() {
            let record: Record = result.expect("Error parsing test entry.");
            let entry = Entry::from(record);

            if let Some(value) = timestamp {
                filter.set_rate((entry.timestamp - value).as_secs_f64().recip());
            }

            let filtered = filter.filter(&entry.noisy.coords);

            timestamp = Some(entry.timestamp);

            assert_abs_diff_eq!(entry.filtered.coords, filtered, epsilon = 1e-6);
        }
    }
}
