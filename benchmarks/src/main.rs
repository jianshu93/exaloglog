use ahash::RandomState;
use exaloglog::ExaLogLog;
use std::hash::{BuildHasher, BuildHasherDefault, Hasher};
use std::hint::black_box;
use std::time::{Duration, Instant};
use tab_hash::Tab64Simple;
use ultraloglog::UltraLogLog;

const EXA_T: u32 = 2;
const EXA_D: u32 = 24;

fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn elapsed_per<F: FnMut()>(iterations: usize, mut f: F) -> Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    start.elapsed() / iterations as u32
}

#[derive(Clone, Copy)]
struct ErrorStats {
    relative_bias: f64,
    relative_rmse: f64,
}

fn accuracy_exaloglog(p: u32, cardinality: usize, trials: usize) -> ErrorStats {
    let mut error_sum = 0.0;
    let mut error_square_sum = 0.0;
    for trial in 0..trials {
        let mut seed = 0xa076_1d64_78bd_642f ^ ((p as u64) << 48) ^ trial as u64;
        let mut sketch = ExaLogLog::new(EXA_T, EXA_D, p).unwrap();
        for _ in 0..cardinality {
            sketch.add(splitmix64(&mut seed));
        }
        let error = sketch.estimate() - cardinality as f64;
        error_sum += error;
        error_square_sum += error * error;
    }
    ErrorStats {
        relative_bias: error_sum / (trials as f64 * cardinality as f64),
        relative_rmse: (error_square_sum / trials as f64).sqrt() / cardinality as f64,
    }
}

fn accuracy_ultraloglog(p: u32, cardinality: usize, trials: usize) -> ErrorStats {
    let mut error_sum = 0.0;
    let mut error_square_sum = 0.0;
    for trial in 0..trials {
        let mut seed = 0xa076_1d64_78bd_642f ^ (((p - 2) as u64) << 48) ^ trial as u64;
        let mut sketch = UltraLogLog::new(p).unwrap();
        for _ in 0..cardinality {
            sketch.add(splitmix64(&mut seed));
        }
        let error = sketch.get_distinct_count_estimate() - cardinality as f64;
        error_sum += error;
        error_square_sum += error * error;
    }
    ErrorStats {
        relative_bias: error_sum / (trials as f64 * cardinality as f64),
        relative_rmse: (error_square_sum / trials as f64).sqrt() / cardinality as f64,
    }
}

fn speed_benchmark() {
    let n = 200_000usize;
    let mut seed = 0x243f_6a88_85a3_08d3;
    let hashes: Vec<u64> = (0..n).map(|_| splitmix64(&mut seed)).collect();

    println!(
        "implementation,pair,bytes,insert_ns_per_item,estimate_ns,merge_ns,estimate_after_{n}"
    );

    for exa_p in [8u32, 10, 12] {
        let ull_p = exa_p + 2;
        let exa_bytes = ExaLogLog::new(EXA_T, EXA_D, exa_p).unwrap().state().len();
        let ull_bytes = UltraLogLog::new(ull_p).unwrap().get_state().len();

        let exa_insert = elapsed_per(5, || {
            let mut sketch = ExaLogLog::new(EXA_T, EXA_D, exa_p).unwrap();
            for &hash in &hashes {
                sketch.add(black_box(hash));
            }
            black_box(sketch);
        });
        let mut exa = ExaLogLog::new(EXA_T, EXA_D, exa_p).unwrap();
        let mut exa_a = ExaLogLog::new(EXA_T, EXA_D, exa_p).unwrap();
        let mut exa_b = ExaLogLog::new(EXA_T, EXA_D, exa_p).unwrap();
        for (i, &hash) in hashes.iter().enumerate() {
            exa.add(hash);
            if i & 1 == 0 {
                exa_a.add(hash);
            } else {
                exa_b.add(hash);
            }
        }
        let exa_estimate = elapsed_per(1_000, || {
            black_box(exa.estimate());
        });
        let exa_merge = elapsed_per(1_000, || {
            let merged = ExaLogLog::merge(black_box(&exa_a), black_box(&exa_b)).unwrap();
            black_box(merged);
        });
        println!(
            "ExaLogLog,t2d24-p{exa_p},{exa_bytes},{:.2},{:.2},{:.2},{:.2}",
            exa_insert.as_nanos() as f64 / n as f64,
            exa_estimate.as_nanos() as f64,
            exa_merge.as_nanos() as f64,
            exa.estimate()
        );

        let ull_insert = elapsed_per(5, || {
            let mut sketch = UltraLogLog::new(ull_p).unwrap();
            for &hash in &hashes {
                sketch.add(black_box(hash));
            }
            black_box(sketch);
        });
        let mut ull = UltraLogLog::new(ull_p).unwrap();
        let mut ull_a = UltraLogLog::new(ull_p).unwrap();
        let mut ull_b = UltraLogLog::new(ull_p).unwrap();
        for (i, &hash) in hashes.iter().enumerate() {
            ull.add(hash);
            if i & 1 == 0 {
                ull_a.add(hash);
            } else {
                ull_b.add(hash);
            }
        }
        let ull_estimate = elapsed_per(1_000, || {
            black_box(ull.get_distinct_count_estimate());
        });
        let ull_merge = elapsed_per(1_000, || {
            let merged = UltraLogLog::merge(black_box(&ull_a), black_box(&ull_b)).unwrap();
            black_box(merged);
        });
        println!(
            "UltraLogLog,p{ull_p},{ull_bytes},{:.2},{:.2},{:.2},{:.2}",
            ull_insert.as_nanos() as f64 / n as f64,
            ull_estimate.as_nanos() as f64,
            ull_merge.as_nanos() as f64,
            ull.get_distinct_count_estimate()
        );
    }
}

fn accuracy_benchmark() {
    let trials = 50usize;
    let cardinalities = [1_000usize, 10_000, 100_000, 1_000_000];

    println!();
    println!("implementation,pair,bytes,true_count,trials,relative_bias,relative_rmse");
    for exa_p in [8u32, 10, 12] {
        let ull_p = exa_p + 2;
        let exa_bytes = ExaLogLog::new(EXA_T, EXA_D, exa_p).unwrap().state().len();
        let ull_bytes = UltraLogLog::new(ull_p).unwrap().get_state().len();

        for &cardinality in &cardinalities {
            let exa_stats = accuracy_exaloglog(exa_p, cardinality, trials);
            println!(
                "ExaLogLog,t{EXA_T}d{EXA_D}-p{exa_p},{exa_bytes},{cardinality},{trials},{:.6},{:.6}",
                exa_stats.relative_bias, exa_stats.relative_rmse
            );

            let ull_stats = accuracy_ultraloglog(ull_p, cardinality, trials);
            println!(
                "UltraLogLog,p{ull_p},{ull_bytes},{cardinality},{trials},{:.6},{:.6}",
                ull_stats.relative_bias, ull_stats.relative_rmse
            );
        }
    }
}

fn time_hash_function<F>(name: &str, p: u32, values: &[u64], iterations: usize, mut run: F)
where
    F: FnMut(&mut ExaLogLog, &[u64]),
{
    let bytes = ExaLogLog::new(EXA_T, EXA_D, p).unwrap().state().len();
    let elapsed = elapsed_per(iterations, || {
        let mut sketch = ExaLogLog::new(EXA_T, EXA_D, p).unwrap();
        run(&mut sketch, black_box(values));
        black_box(sketch);
    });

    let mut sketch = ExaLogLog::new(EXA_T, EXA_D, p).unwrap();
    run(&mut sketch, values);
    let estimate = sketch.estimate();
    let true_count = values.len() as f64;
    let relative_error = (estimate - true_count) / true_count;

    println!(
        "{name},t{EXA_T}d{EXA_D}-p{p},{bytes},{},{:.2},{:.2},{:.6}",
        values.len(),
        elapsed.as_nanos() as f64 / values.len() as f64,
        estimate,
        relative_error
    );
}

fn hash_function_benchmark() {
    const P: u32 = 12;
    const N: usize = 200_000;
    const ITERATIONS: usize = 5;

    let values: Vec<u64> = (0..N as u64).collect();
    println!();
    println!("hash_function,pair,bytes,true_count,hash_and_insert_ns_per_item,estimate,relative_error");

    time_hash_function("xxh3", P, &values, ITERATIONS, |sketch, values| {
        for &value in values {
            sketch.add_raw(black_box(value));
        }
    });

    let ahash = RandomState::with_seeds(1, 2, 3, 4);
    time_hash_function("ahash", P, &values, ITERATIONS, |sketch, values| {
        for &value in values {
            sketch.add_raw_with_build_hasher(black_box(value), &ahash);
        }
    });

    use komihash::v5::KomiHasher;
    #[derive(Clone)]
    struct KomiBuildHasher {
        seed: u64,
    }
    impl BuildHasher for KomiBuildHasher {
        type Hasher = KomiHasher;

        #[inline]
        fn build_hasher(&self) -> Self::Hasher {
            KomiHasher::new(self.seed)
        }
    }
    let komihash = KomiBuildHasher {
        seed: 0x1234_5678_9abc_def0,
    };
    time_hash_function("komihash", P, &values, ITERATIONS, |sketch, values| {
        for &value in values {
            sketch.add_raw_with_build_hasher(black_box(value), &komihash);
        }
    });

    type PolymurBuildHasher = BuildHasherDefault<polymur_hash::PolymurHasher>;
    let polymur = PolymurBuildHasher::default();
    time_hash_function("polymurhash", P, &values, ITERATIONS, |sketch, values| {
        for &value in values {
            sketch.add_raw_with_build_hasher(black_box(value), &polymur);
        }
    });

    let wyhash = wyhash::WyHasherBuilder::default();
    time_hash_function("wyhash", P, &values, ITERATIONS, |sketch, values| {
        for &value in values {
            sketch.add_raw_with_build_hasher(black_box(value), &wyhash);
        }
    });

    #[derive(Default)]
    struct T1ha2AtonceHasher {
        seed: u64,
        buf: Vec<u8>,
    }
    impl Hasher for T1ha2AtonceHasher {
        fn write(&mut self, bytes: &[u8]) {
            self.buf.extend_from_slice(bytes);
        }

        fn finish(&self) -> u64 {
            t1ha::t1ha2_atonce(&self.buf, self.seed)
        }
    }
    #[derive(Clone, Default)]
    struct T1ha2AtonceBuildHasher;
    impl BuildHasher for T1ha2AtonceBuildHasher {
        type Hasher = T1ha2AtonceHasher;

        fn build_hasher(&self) -> Self::Hasher {
            T1ha2AtonceHasher::default()
        }
    }
    let t1ha = T1ha2AtonceBuildHasher;
    time_hash_function("t1ha2_atonce", P, &values, ITERATIONS, |sketch, values| {
        for &value in values {
            sketch.add_raw_with_build_hasher(black_box(value), &t1ha);
        }
    });

    let tab_hash = Tab64Simple::new();
    time_hash_function("tab-hash", P, &values, ITERATIONS, |sketch, values| {
        for &value in values {
            sketch.add_raw_with_hash_fn(black_box(value), |value| tab_hash.hash(*value));
        }
    });
}

fn main() {
    speed_benchmark();
    accuracy_benchmark();
    hash_function_benchmark();
}
