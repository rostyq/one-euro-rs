#[macro_use]
extern crate approx;

#[cfg(test)]
mod tests {

    use std::{fs::File, time::Duration, fmt::Debug};

    use nalgebra::Point2;
    use one_euro::{OneEuroState, OneEuroFilter};
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
        let mut filter = OneEuroFilter::<f64>::default();

        filter.set_beta(0.007);

        let mut records = reader.deserialize::<Record>();

        let (mut timestamp, mut state) = records.next().map(|r| {
            let record: Record = r.expect("Error parsing test entry.");
            let entry = Entry::from(record);
            let state: OneEuroState<f64, 2> = entry.noisy.coords.into();
            let timestamp = entry.timestamp;
            (timestamp, state)
        }).unwrap();

        for result in records {
            let record: Record = result.expect("Error parsing test entry.");
            let entry = Entry::from(record);

            let rate = (entry.timestamp - timestamp).as_secs_f64().recip();

            filter.filter(&mut state, &entry.noisy.coords, rate);

            timestamp = entry.timestamp;

            assert_abs_diff_eq!(entry.filtered.coords, state.data(), epsilon = 1e-6);
        }
    }
}
