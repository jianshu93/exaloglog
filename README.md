# exaloglog

Standalone Rust implementation of ExaLogLog, based on the algorithm described in the EDBT 2025 paper and the Java reference implementation.

The core crate exposes:

- `ExaLogLog::new(t, d, p)` for sketch creation
- prehashed 64-bit insertion with `add_hash`/`add`
- raw value insertion with default `xxh3-64` hashing via `add_raw`
- caller-provided hashers via `add_raw_with_build_hasher` and `add_raw_with_hash_fn`
- 32-bit token support with `compute_token` and `add_token`
- ML cardinality estimation with bias correction constants
- merge and downsize operations
- versioned byte serialization with `to_bytes` and `from_bytes`

## Input Modes

```rust
use ahash::RandomState;
use exaloglog::ExaLogLog;

let mut sketch = ExaLogLog::new(2, 24, 12).unwrap();

// 1. Already-hashed 64-bit values.
sketch.add_hash(0x0123_4567_89ab_cdef);
sketch.add_hashes([0xfedc_ba98_7654_3210, 0x9e37_79b9_7f4a_7c15]);

// 2. Raw values using the default xxh3-64 hash function.
sketch.add_raw("apple");
sketch.add_raw_values(["banana", "cherry"]);
let default_hash = ExaLogLog::hash_value("dragonfruit");
sketch.add_hash(default_hash);

// 3. Raw values using a caller-provided BuildHasher.
let ahash = RandomState::with_seeds(1, 2, 3, 4);
sketch.add_raw_with_build_hasher("elderberry", &ahash);

// 4. One-shot hash functions, useful for tabulation hashing or domain-specific prehashers.
sketch.add_raw_with_hash_fn(42u64, |value| ExaLogLog::hash_value(value));
```

The older `add(u64)` method remains available as a compatibility alias for `add_hash`.

## Serialization

```rust
use exaloglog::ExaLogLog;

let mut sketch = ExaLogLog::new(2, 24, 12).unwrap();
sketch.add_raw_values(["apple", "banana", "cherry"]);

let bytes = sketch.to_bytes();
let restored = ExaLogLog::from_bytes(&bytes).unwrap();

assert_eq!(restored.t(), sketch.t());
assert_eq!(restored.d(), sketch.d());
assert_eq!(restored.p(), sketch.p());
assert_eq!(restored.state(), sketch.state());
```

Important licensing note: the paper repository's Java code is published under a restrictive illustrative license, not a normal reusable open-source license. This crate is useful as an implementation prototype and benchmark target, but the licensing should be reviewed before publishing or using it beyond local evaluation.

## Quick check

```bash
cargo test
```

## UltraLogLog comparison

The benchmark harness is kept in `benchmarks/` so the core crate stays standalone. It references the local UltraLogLog checkout at `/Users/jianshuzhao/Github/ultraloglog`.

```bash
cargo run --release --manifest-path benchmarks/Cargo.toml
```

The runner prints three CSV tables:

- speed: insert time, estimate time, merge time
- accuracy: relative bias and relative RMSE over repeated random trials
- hash-function comparison: hash+insert time and relative error for xxh3, ahash, komihash,
  polymurhash, wyhash, t1ha2, and tab-hash

## References
Ertl, O., 2024. ExaLogLog: Space-Efficient and Practical Approximate Distinct Counting up to the Exa-Scale. arXiv preprint arXiv:2402.13726.
