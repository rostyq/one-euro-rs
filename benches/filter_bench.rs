use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use nalgebra::SVector;
use one_euro::{OneEuroFilter, OneEuroState};

macro_rules! bench_filter_n {
    ($n:expr, $group:ident) => {
        let id = BenchmarkId::from_parameter($n as usize);
        let filter = OneEuroFilter::<f64>::default();

        $group.bench_function(id, |b| {
            let rate = 1.0;
            let raw = SVector::<f64, $n>::repeat(1.0);
            let mut state: OneEuroState<f64, $n> = SVector::<f64, $n>::zeros().into();

            b.iter(|| {
                black_box(filter).filter(black_box(&mut state), black_box(&raw), black_box(rate.to_owned()));
            })
        });
    };
}

fn bench_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("OneEuroFilter::filter");
    group.sample_size(1000);

    bench_filter_n!(1, group);
    bench_filter_n!(2, group);
    bench_filter_n!(3, group);
}

criterion_group!(benches, bench_filter);
criterion_main!(benches);