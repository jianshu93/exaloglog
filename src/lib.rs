mod ml_bias;

use std::error::Error;
use std::fmt;
use std::hash::{BuildHasher, Hash, Hasher};
use xxhash_rust::xxh3::Xxh3;

pub const V: u32 = 26;
pub const MIN_P: u32 = 2;
pub const MAX_T: u32 = V - MIN_P;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExaLogLogError {
    InvalidT,
    InvalidD,
    InvalidP,
    InvalidStateLength,
    IncompatibleT,
    OtherHasSmallerD,
    OtherHasSmallerPrecision,
}

impl fmt::Display for ExaLogLogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidT => f.write_str("invalid t parameter"),
            Self::InvalidD => f.write_str("invalid d parameter"),
            Self::InvalidP => f.write_str("invalid precision parameter"),
            Self::InvalidStateLength => f.write_str("unexpected state length"),
            Self::IncompatibleT => f.write_str("t parameters differ"),
            Self::OtherHasSmallerD => f.write_str("other sketch has smaller d parameter"),
            Self::OtherHasSmallerPrecision => f.write_str("other sketch has smaller precision"),
        }
    }
}

impl Error for ExaLogLogError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExaLogLog {
    t: u32,
    d: u32,
    p: u32,
    state: Vec<u8>,
}

impl ExaLogLog {
    pub fn new(t: u32, d: u32, p: u32) -> Result<Self, ExaLogLogError> {
        check_t(t)?;
        check_d(d, t)?;
        check_p(p, t)?;
        Ok(Self {
            t,
            d,
            p,
            state: vec![0; state_len_bytes(t, d, p)],
        })
    }

    pub fn wrap(t: u32, d: u32, state: Vec<u8>) -> Result<Self, ExaLogLogError> {
        check_t(t)?;
        check_d(d, t)?;
        let width = register_bit_size(t, d) as usize;
        if state.is_empty() {
            return Err(ExaLogLogError::InvalidStateLength);
        }
        let m = (state.len() * 8) / width;
        if m == 0 {
            return Err(ExaLogLogError::InvalidStateLength);
        }
        let p = usize::BITS - 1 - m.leading_zeros();
        if p < MIN_P || p > max_p(t) || state_len_bytes(t, d, p) != state.len() {
            return Err(ExaLogLogError::InvalidStateLength);
        }
        Ok(Self { t, d, p, state })
    }

    pub fn t(&self) -> u32 {
        self.t
    }

    pub fn d(&self) -> u32 {
        self.d
    }

    pub fn p(&self) -> u32 {
        self.p
    }

    pub fn state(&self) -> &[u8] {
        &self.state
    }

    pub fn into_state(self) -> Vec<u8> {
        self.state
    }

    pub fn reset(&mut self) -> &mut Self {
        self.state.fill(0);
        self
    }

    /// Hashes a raw value with the crate's default hash function, xxh3-64.
    pub fn hash_value<T: Hash>(value: T) -> u64 {
        let mut hasher = Xxh3::default();
        value.hash(&mut hasher);
        hasher.finish()
    }

    /// Hashes a raw value with a caller-provided [`BuildHasher`].
    #[allow(clippy::manual_hash_one)]
    pub fn hash_value_with_build_hasher<T, S>(value: T, build_hasher: &S) -> u64
    where
        T: Hash,
        S: BuildHasher + ?Sized,
    {
        let mut hasher = build_hasher.build_hasher();
        value.hash(&mut hasher);
        hasher.finish()
    }

    /// Adds an already hashed 64-bit value to this sketch.
    pub fn add_hash(&mut self, hash_value: u64) -> &mut Self {
        let mask = ((1u64 << self.t) << self.p) - 1;
        let idx = ((hash_value & mask) >> self.t) as usize;
        let nlz = (hash_value | mask).leading_zeros() as u64;
        let low_mask = (1u64 << self.t) - 1;
        let k = (nlz << self.t) + (hash_value & low_mask) + 1;

        let r_old = self.get_register(idx);
        let u = r_old >> self.d;
        if k > u {
            let delta = k - u;
            let mut r_new = k << self.d;
            if delta <= self.d as u64 {
                r_new |= ((1u64 << self.d) | (r_old & ((1u64 << self.d) - 1))) >> delta;
            }
            self.set_register(idx, r_new);
        } else if k < u {
            let delta = u - k;
            if delta <= self.d as u64 {
                let r_new = r_old | (1u64 << (self.d as u64 - delta));
                if r_new != r_old {
                    self.set_register(idx, r_new);
                }
            }
        }
        self
    }

    /// Adds already hashed 64-bit values to this sketch.
    pub fn add_hashes<I>(&mut self, hash_values: I) -> &mut Self
    where
        I: IntoIterator<Item = u64>,
    {
        for hash_value in hash_values {
            self.add_hash(hash_value);
        }
        self
    }

    /// Adds a raw value after hashing it with the crate's default hash function, xxh3-64.
    pub fn add_raw<T: Hash>(&mut self, value: T) -> &mut Self {
        self.add_hash(Self::hash_value(value))
    }

    /// Adds raw values after hashing them with the crate's default hash function, xxh3-64.
    pub fn add_raw_values<I, T>(&mut self, values: I) -> &mut Self
    where
        I: IntoIterator<Item = T>,
        T: Hash,
    {
        for value in values {
            self.add_raw(value);
        }
        self
    }

    /// Adds a raw value after hashing it with a caller-provided [`BuildHasher`].
    pub fn add_raw_with_build_hasher<T, S>(&mut self, value: T, build_hasher: &S) -> &mut Self
    where
        T: Hash,
        S: BuildHasher + ?Sized,
    {
        self.add_hash(Self::hash_value_with_build_hasher(value, build_hasher))
    }

    /// Adds raw values after hashing them with a caller-provided [`BuildHasher`].
    pub fn add_raw_values_with_build_hasher<I, T, S>(
        &mut self,
        values: I,
        build_hasher: &S,
    ) -> &mut Self
    where
        I: IntoIterator<Item = T>,
        T: Hash,
        S: BuildHasher + ?Sized,
    {
        for value in values {
            self.add_raw_with_build_hasher(value, build_hasher);
        }
        self
    }

    /// Adds a raw value using an arbitrary one-shot hash function.
    ///
    /// This path is useful for hashers that are naturally exposed as functions, such as tabulation
    /// hashing over integer IDs or other domain-specific prehashers.
    pub fn add_raw_with_hash_fn<T, F>(&mut self, value: T, hash_fn: F) -> &mut Self
    where
        F: FnOnce(&T) -> u64,
    {
        self.add_hash(hash_fn(&value))
    }

    /// Adds raw values using an arbitrary one-shot hash function.
    pub fn add_raw_values_with_hash_fn<I, T, F>(&mut self, values: I, mut hash_fn: F) -> &mut Self
    where
        I: IntoIterator<Item = T>,
        F: FnMut(&T) -> u64,
    {
        for value in values {
            self.add_hash(hash_fn(&value));
        }
        self
    }

    pub fn add(&mut self, hash_value: u64) -> &mut Self {
        self.add_hash(hash_value)
    }

    pub fn compute_token(hash_value: u64) -> u32 {
        compute_token(hash_value, V)
    }

    pub fn add_token(&mut self, token: u32) -> &mut Self {
        self.add(reconstruct_hash(token, V))
    }

    pub fn estimate(&self) -> f64 {
        self.estimate_with_solver_stats(None)
    }

    pub fn get_distinct_count_estimate(&self) -> f64 {
        self.estimate()
    }

    pub fn downsize(&self, d: u32, p: u32) -> Result<Self, ExaLogLogError> {
        check_d(d, self.t)?;
        check_p(p, self.t)?;
        if p >= self.p && d >= self.d {
            Ok(self.clone())
        } else {
            let mut downsized = Self::new(self.t, d, p)?;
            downsized.add_sketch(self)?;
            Ok(downsized)
        }
    }

    pub fn add_sketch(&mut self, other: &Self) -> Result<&mut Self, ExaLogLogError> {
        if other.t != self.t {
            return Err(ExaLogLogError::IncompatibleT);
        }
        if other.d < self.d {
            return Err(ExaLogLogError::OtherHasSmallerD);
        }
        if other.p < self.p {
            return Err(ExaLogLogError::OtherHasSmallerPrecision);
        }

        let m = self.num_registers();
        if other.d == self.d && other.p == self.p {
            for idx in 0..m {
                let this_r = self.get_register(idx);
                let other_r = other.get_register(idx);
                let merged_r = merge_register(this_r, other_r, self.d);
                if this_r != merged_r {
                    self.set_register(idx, merged_r);
                }
            }
        } else {
            let max_sub_index = 1usize << (other.p - self.p);
            let downsize = DownsizeParams {
                t: self.t,
                from_d: other.d,
                to_d: self.d,
                from_p: other.p,
                to_p: self.p,
                threshold_u: compute_downsize_threshold_u(self.t, other.p),
            };
            for register_index in 0..m {
                let mut merged_r =
                    downsize_register(other.get_register(register_index), 0, downsize);
                for sub_index in 1..max_sub_index {
                    let other_idx = register_index + (sub_index << self.p);
                    let other_r = downsize_register(
                        other.get_register(other_idx),
                        sub_index as u32,
                        downsize,
                    );
                    merged_r = merge_register(merged_r, other_r, self.d);
                }
                if merged_r != 0 {
                    let this_r = self.get_register(register_index);
                    merged_r = merge_register(merged_r, this_r, self.d);
                    if this_r != merged_r {
                        self.set_register(register_index, merged_r);
                    }
                }
            }
        }
        Ok(self)
    }

    pub fn merge(sketch1: &Self, sketch2: &Self) -> Result<Self, ExaLogLogError> {
        if sketch1.t != sketch2.t {
            return Err(ExaLogLogError::IncompatibleT);
        }
        if sketch1.p <= sketch2.p {
            if sketch1.d <= sketch2.d {
                let mut result = sketch1.clone();
                result.add_sketch(sketch2)?;
                Ok(result)
            } else {
                let mut result = sketch1.downsize(sketch2.d, sketch1.p)?;
                result.add_sketch(sketch2)?;
                Ok(result)
            }
        } else if sketch1.d >= sketch2.d {
            let mut result = sketch2.clone();
            result.add_sketch(sketch1)?;
            Ok(result)
        } else {
            let mut result = sketch2.downsize(sketch1.d, sketch2.p)?;
            result.add_sketch(sketch1)?;
            Ok(result)
        }
    }

    fn estimate_with_solver_stats(&self, solver_stats: Option<&mut SolverStatistics>) -> f64 {
        let mut agg = 0u64;
        let mut b = [0i32; 64];
        for idx in 0..self.num_registers() {
            agg = agg.wrapping_add(contribute(
                self.get_register(idx),
                Some(&mut b),
                self.t,
                self.d,
                self.p,
            ));
        }

        let q = (63 - self.t - self.p) as usize;
        if agg == 0 {
            return if b[q] == 0 { 0.0 } else { f64::INFINITY };
        }

        let factor = ((1u64 << self.p) << (self.t + 1)) as f64;
        let a = unsigned_long_to_double(agg) * TWO_POW_MINUS_64 * factor;
        factor * solve_maximum_likelihood_equation(a, &b, q, 0.0, solver_stats)
            / (1.0 + ml_bias::bias_correction(self.t, self.d) / (1u64 << self.p) as f64)
    }

    fn num_registers(&self) -> usize {
        1usize << self.p
    }

    fn register_width(&self) -> u32 {
        register_bit_size(self.t, self.d)
    }

    fn get_register(&self, idx: usize) -> u64 {
        if self.register_width() == 32 {
            let offset = idx * 4;
            let bytes = [
                self.state[offset],
                self.state[offset + 1],
                self.state[offset + 2],
                self.state[offset + 3],
            ];
            u32::from_le_bytes(bytes) as u64
        } else {
            packed_get(&self.state, idx, self.register_width())
        }
    }

    fn set_register(&mut self, idx: usize, value: u64) {
        if self.register_width() == 32 {
            let offset = idx * 4;
            self.state[offset..offset + 4].copy_from_slice(&(value as u32).to_le_bytes());
        } else {
            let width = self.register_width();
            packed_set(&mut self.state, idx, width, value);
        }
    }
}

pub fn max_p(t: u32) -> u32 {
    V - t
}

pub fn max_d(t: u32) -> u32 {
    64 - 6 - t
}

pub fn register_bit_size(t: u32, d: u32) -> u32 {
    6 + t + d
}

pub fn compute_token(hash_value: u64, v: u32) -> u32 {
    let mask = u64::MAX >> (64 - v);
    let idx = (hash_value & mask) as u32;
    let nlz = (hash_value | mask).leading_zeros();
    (idx << 6) | nlz
}

pub fn reconstruct_hash(token: u32, v: u32) -> u64 {
    let idx = (token >> 6) as u64;
    let nlz = token & 0x3f;
    ((u64::MAX >> v) >> nlz) << v | idx
}

#[derive(Default)]
struct SolverStatistics {
    iteration_counter: usize,
}

const TWO_POW_MINUS_64: f64 = 1.0 / 18446744073709551616.0;

fn check_t(t: u32) -> Result<(), ExaLogLogError> {
    if t <= MAX_T {
        Ok(())
    } else {
        Err(ExaLogLogError::InvalidT)
    }
}

fn check_d(d: u32, t: u32) -> Result<(), ExaLogLogError> {
    if d <= max_d(t) {
        Ok(())
    } else {
        Err(ExaLogLogError::InvalidD)
    }
}

fn check_p(p: u32, t: u32) -> Result<(), ExaLogLogError> {
    if (MIN_P..=max_p(t)).contains(&p) {
        Ok(())
    } else {
        Err(ExaLogLogError::InvalidP)
    }
}

fn state_len_bytes(t: u32, d: u32, p: u32) -> usize {
    (((register_bit_size(t, d) as usize) * (1usize << p)) + 7) >> 3
}

fn packed_mask(width: u32) -> u128 {
    if width == 64 {
        u64::MAX as u128
    } else {
        (1u128 << width) - 1
    }
}

fn packed_window_len(width: u32, shift: usize) -> usize {
    (shift + width as usize + 7) >> 3
}

fn packed_get(data: &[u8], idx: usize, width: u32) -> u64 {
    let bit_offset = idx * width as usize;
    let byte_offset = bit_offset >> 3;
    let shift = bit_offset & 7;
    if shift == 0 && width.is_multiple_of(8) {
        let byte_width = (width / 8) as usize;
        let mut bytes = [0u8; 8];
        bytes[..byte_width].copy_from_slice(&data[byte_offset..byte_offset + byte_width]);
        return u64::from_le_bytes(bytes) & packed_mask(width) as u64;
    }
    let len = packed_window_len(width, shift);
    let mut window = 0u128;
    for i in 0..len {
        if let Some(&byte) = data.get(byte_offset + i) {
            window |= (byte as u128) << (i * 8);
        }
    }
    ((window >> shift) & packed_mask(width)) as u64
}

fn packed_set(data: &mut [u8], idx: usize, width: u32, value: u64) {
    let bit_offset = idx * width as usize;
    let byte_offset = bit_offset >> 3;
    let shift = bit_offset & 7;
    if shift == 0 && width.is_multiple_of(8) {
        let byte_width = (width / 8) as usize;
        let bytes = (value & packed_mask(width) as u64).to_le_bytes();
        data[byte_offset..byte_offset + byte_width].copy_from_slice(&bytes[..byte_width]);
        return;
    }
    let len = packed_window_len(width, shift);
    let mask = packed_mask(width) << shift;
    let mut window = 0u128;
    for i in 0..len {
        if let Some(&byte) = data.get(byte_offset + i) {
            window |= (byte as u128) << (i * 8);
        }
    }
    window = (window & !mask) | (((value as u128) << shift) & mask);
    for i in 0..len {
        if let Some(byte) = data.get_mut(byte_offset + i) {
            *byte = (window >> (i * 8)) as u8;
        }
    }
}

fn shift_right(value: u64, delta: u64) -> u64 {
    if delta < 64 { value >> delta } else { 0 }
}

fn compute_downsize_threshold_u(t: u32, from_p: u32) -> u64 {
    ((64 - t - from_p) as u64) << t | 1
}

#[derive(Clone, Copy)]
struct DownsizeParams {
    t: u32,
    from_d: u32,
    to_d: u32,
    from_p: u32,
    to_p: u32,
    threshold_u: u64,
}

fn downsize_register(mut r: u64, sub_idx: u32, params: DownsizeParams) -> u64 {
    let DownsizeParams {
        t,
        from_d,
        to_d,
        from_p,
        to_p,
        threshold_u,
    } = params;
    debug_assert!(from_d >= to_d);
    let u = r >> from_d;
    r >>= from_d - to_d;
    if u >= threshold_u {
        let sub_idx_width = u32::BITS - sub_idx.leading_zeros();
        let shift = ((from_p - to_p) - sub_idx_width) << t;
        if shift > 0 {
            let num_bits_to_shift = to_d as i64 + threshold_u as i64 - u as i64;
            if num_bits_to_shift > 0 {
                let mask = u64::MAX.wrapping_shl((num_bits_to_shift as u32) & 63);
                r = (mask & r) | shift_right(r & !mask, shift as u64);
            }
            r += (shift as u64) << to_d;
        }
    }
    r
}

fn merge_register(r1: u64, r2: u64, d: u32) -> u64 {
    let u1 = r1 >> d;
    let u2 = r2 >> d;
    if u1 > u2 && u2 > 0 {
        let x = 1u64 << d;
        r1 | shift_right(x | (r2 & (x - 1)), u1 - u2)
    } else if u2 > u1 && u1 > 0 {
        let x = 1u64 << d;
        r2 | shift_right(x | (r1 & (x - 1)), u2 - u1)
    } else {
        r1 | r2
    }
}

fn arithmetic_shr_u64(value: u64, shift: u32) -> u64 {
    ((value as i64) >> shift) as u64
}

fn contribute(r: u64, mut b: Option<&mut [i32; 64]>, t: u32, d: u32, p: u32) -> u64 {
    let u = (r >> d) as i32;
    if u == 0 {
        return 1u64 << (64 - p);
    }

    let q = 63_i32 - t as i32 - p as i32;
    let j = (u - 1) >> t;
    let mut i = q.min(j);
    let r_inv = !r;
    let num_bits = (u - 1) - (i << t);
    let mut mask = u64::MAX << (d as i32 - num_bits).max(0);
    let low_mask = (1u64 << d) - 1;
    let mask2 = mask & low_mask;

    let mut a =
        ((((i + 2) as u64) << t) - u as u64 + (r_inv & mask2).count_ones() as u64) << (q - i);
    if let Some(b) = b.as_mut() {
        b[i as usize] += 1 + (r & mask2).count_ones() as i32;
    }

    if t <= 5 {
        let shift = 1u32 << t;
        mask ^= arithmetic_shr_u64(mask, shift);
        while i > 0 && mask != 0 {
            i -= 1;
            a += ((mask & r_inv).count_ones() as u64) << (q - i);
            if let Some(b) = b.as_mut() {
                b[i as usize] += (mask & r).count_ones() as i32;
            }
            mask >>= shift;
        }
    } else if i > 0 {
        mask = !mask;
        i -= 1;
        a += ((mask & r_inv).count_ones() as u64) << (q - i);
        if let Some(b) = b.as_mut() {
            b[i as usize] += (mask & r).count_ones() as i32;
        }
    }
    a
}

fn solve_maximum_likelihood_equation(
    a: f64,
    b: &[i32],
    n: usize,
    relative_error_limit: f64,
    mut solver_statistics: Option<&mut SolverStatistics>,
) -> f64 {
    let mut sigma0 = 0i64;
    let mut sigma1 = 0.0;
    let mut u_min: Option<usize> = None;
    let mut u_max = 0usize;

    for (j, &bj) in b.iter().take(n + 1).enumerate() {
        if bj > 0 {
            if u_min.is_none() {
                u_min = Some(j);
            }
            u_max = j;
            sigma0 += bj as i64;
            sigma1 += (bj as f64) * pow2(-(j as i32));
        }
    }

    let Some(u_min) = u_min else {
        return 0.0;
    };

    let pow_u_max = pow2(u_max as i32);
    sigma1 *= pow_u_max;
    let a_pow_u_max = a * pow_u_max;
    let mut x = sigma1 / a_pow_u_max;

    if u_min < u_max {
        x = ((sigma0 as f64 / sigma1) * x.ln_1p()).exp_m1();

        loop {
            if let Some(stats) = solver_statistics.as_deref_mut() {
                stats.iteration_counter += 1;
            }
            let mut lambda = 1.0;
            let mut eta = 0.0;
            let mut y = x;
            let mut u = u_max;
            let mut phi = b[u_max] as f64;
            let mut psi = 0.0;
            loop {
                u -= 1;
                let y_plus_2 = 2.0 + y;
                let z = 2.0 / y_plus_2;
                lambda *= z;
                eta = eta * (2.0 - z) + (1.0 - z);
                let b_lambda = b[u] as f64 * lambda;
                phi += b_lambda;
                psi += b_lambda * eta;
                if u <= u_min {
                    break;
                }
                y *= y_plus_2;
            }

            let x_prime = a_pow_u_max * x;
            if phi <= x_prime {
                break;
            }
            let old_x = x;
            let eps = (phi - x_prime) / (psi + x_prime);
            x += x * eps;
            if eps <= relative_error_limit || x <= old_x {
                break;
            }
        }
    }
    x.ln_1p() * pow_u_max
}

fn unsigned_long_to_double(value: u64) -> f64 {
    let mut d = (value & 0x7fff_ffff_ffff_ffff) as f64;
    if value & (1u64 << 63) != 0 {
        d += 9_223_372_036_854_775_808.0;
    }
    d
}

fn pow2(x: i32) -> f64 {
    f64::from_bits(((x as i64 + 1023) as u64) << 52)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ahash::RandomState;
    use std::hash::{BuildHasher, Hasher};

    fn splitmix64(x: &mut u64) -> u64 {
        *x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *x;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn phi(k: u64, p: u32, t: u32) -> u32 {
        if k == 0 {
            t
        } else {
            (t + 1 + (((k - 1) >> t) as u32)).min(64 - p)
        }
    }

    fn omega_scaled(u: u64, p: u32, t: u32) -> u64 {
        let phi_eval = phi(u, p, t);
        ((((1_i64 - t as i64 + phi_eval as i64) as u64) << t) - u) << (64 - p - phi_eval)
    }

    fn contribute_reference(r: u64, b: &mut [i32; 64], p: u32, t: u32, d: u32) -> u64 {
        let u = r >> d;
        let mut a_prime = omega_scaled(u, p, t);
        if u >= 1 {
            let mut j = phi(u, p, t);
            b[(j - t - 1) as usize] += 1;
            if u >= 2 {
                for k in 1.max(u.saturating_sub(d as u64))..u {
                    j = phi(k, p, t);
                    let bit = k + d as u64 - u;
                    if (r & (1u64 << bit)) == 0 {
                        a_prime += 1u64 << (64 - p - j);
                    } else {
                        b[(j - t - 1) as usize] += 1;
                    }
                }
            }
        }
        a_prime
    }

    #[test]
    fn validates_parameters() {
        assert_eq!(max_p(0), 26);
        assert_eq!(max_d(0), 58);
        assert!(ExaLogLog::new(0, 0, MIN_P).is_ok());
        assert!(ExaLogLog::new(MAX_T, max_d(MAX_T), MIN_P).is_ok());
        assert_eq!(
            ExaLogLog::new(MAX_T + 1, 0, MIN_P),
            Err(ExaLogLogError::InvalidT)
        );
        assert_eq!(
            ExaLogLog::new(0, max_d(0) + 1, MIN_P),
            Err(ExaLogLogError::InvalidD)
        );
        assert_eq!(
            ExaLogLog::new(0, 0, MIN_P - 1),
            Err(ExaLogLogError::InvalidP)
        );
    }

    #[test]
    fn packed_registers_round_trip() {
        for width in 1u32..=64 {
            let len = (width * 129).div_ceil(8);
            let mut data = vec![0u8; len as usize];
            let mask = if width == 64 {
                u64::MAX
            } else {
                (1u64 << width) - 1
            };
            let mut seed = width as u64;
            for idx in 0..129 {
                let value = splitmix64(&mut seed);
                packed_set(&mut data, idx, width, value);
                assert_eq!(packed_get(&data, idx, width), value & mask);
            }
        }
    }

    #[test]
    fn token_addition_matches_hash_addition() {
        let mut seed = 0x1234_5678_9abc_def0;
        for p in 2..=10 {
            let mut by_hash = ExaLogLog::new(2, 20, p).unwrap();
            let mut by_token = ExaLogLog::new(2, 20, p).unwrap();
            for _ in 0..10_000 {
                let hash = splitmix64(&mut seed);
                by_hash.add(hash);
                by_token.add_token(ExaLogLog::compute_token(hash));
            }
            assert_eq!(by_hash.state(), by_token.state());
        }
    }

    #[test]
    fn merge_matches_direct_insert() {
        let mut seed = 0x11a7_3f21_bb8a_d8f6;
        for p1 in 2..=8 {
            for p2 in 2..=8 {
                let mut sketch1 = ExaLogLog::new(2, 20, p1).unwrap();
                let mut sketch2 = ExaLogLog::new(2, 20, p2).unwrap();
                let mut total = ExaLogLog::new(2, 20, p1.min(p2)).unwrap();

                for _ in 0..2048 {
                    let hash = splitmix64(&mut seed);
                    sketch1.add(hash);
                    total.add(hash);
                }
                for _ in 0..3072 {
                    let hash = splitmix64(&mut seed);
                    sketch2.add(hash);
                    total.add(hash);
                }

                let merged = ExaLogLog::merge(&sketch1, &sketch2).unwrap();
                assert_eq!(merged.state(), total.state());
            }
        }
    }

    #[test]
    fn downsize_matches_direct_insert() {
        let mut seed = 0x2378_46c7_b27d_f6b4;
        for original_p in 2..=10 {
            for downsized_p in 2..=original_p {
                let mut original = ExaLogLog::new(2, 24, original_p).unwrap();
                let mut direct = ExaLogLog::new(2, 24, downsized_p).unwrap();
                for _ in 0..4096 {
                    let hash = splitmix64(&mut seed);
                    original.add(hash);
                    direct.add(hash);
                }
                let downsized = original.downsize(24, downsized_p).unwrap();
                assert_eq!(downsized.state(), direct.state());
            }
        }
    }

    #[test]
    fn contribute_matches_reference() {
        let mut seed = 0;
        for t in 0..=8 {
            for d in 0..=16.min(max_d(t)) {
                for p in MIN_P..=8.min(max_p(t)) {
                    let max_u = ((65 - p - t) as u64) << t;
                    for u in 0..=max_u.min(400) {
                        for _ in 0..3 {
                            let low = if d == 0 {
                                0
                            } else {
                                splitmix64(&mut seed) & ((1u64 << d) - 1)
                            };
                            let r = (u << d) | low;
                            let mut b = [0i32; 64];
                            let mut b_ref = [0i32; 64];
                            let a = contribute(r, Some(&mut b), t, d, p);
                            let a_ref = contribute_reference(r, &mut b_ref, p, t, d);
                            assert_eq!((a, b), (a_ref, b_ref), "t={t} d={d} p={p} r={r}");
                            assert_eq!(contribute(r, None, t, d, p), a_ref);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn estimates_empty_and_nonempty() {
        let mut sketch = ExaLogLog::new(2, 24, 8).unwrap();
        assert_eq!(sketch.estimate(), 0.0);
        let mut seed = 0xfeed_face_cafe_beef;
        for _ in 0..50_000 {
            sketch.add(splitmix64(&mut seed));
        }
        let estimate = sketch.estimate();
        assert!(estimate > 40_000.0 && estimate < 60_000.0, "{estimate}");
    }

    #[test]
    fn hash_input_api_prehashed_u64_matches_add_alias() {
        let hashes = [
            0x0123_4567_89ab_cdef,
            0xfedc_ba98_7654_3210,
            0x9e37_79b9_7f4a_7c15,
            0xbf58_476d_1ce4_e5b9,
            0x94d0_49bb_1331_11eb,
        ];

        let mut via_add = ExaLogLog::new(2, 24, 8).unwrap();
        for hash in hashes {
            via_add.add(hash);
        }

        let mut via_hash_api = ExaLogLog::new(2, 24, 8).unwrap();
        via_hash_api.add_hashes(hashes);

        assert_eq!(via_add.state(), via_hash_api.state());
    }

    #[test]
    fn test_xxhash3() {
        let mut sketch = ExaLogLog::new(2, 24, 8).unwrap();

        sketch
            .add_raw("apple")
            .add_raw("banana")
            .add_raw("cherry")
            .add_raw("033");

        let est = sketch.estimate();
        assert!(
            (est - 4.0).abs() < 0.25,
            "estimate {:.3} deviates from true count 4",
            est
        );
    }

    #[test]
    fn test_custom_ahash_hasher() {
        let ahash = RandomState::with_seeds(1, 2, 3, 4);
        let mut sketch = ExaLogLog::new(2, 24, 8).unwrap();

        sketch
            .add_raw_with_build_hasher("apple", &ahash)
            .add_raw_with_build_hasher("banana", &ahash)
            .add_raw_with_build_hasher("cherry", &ahash);

        let est = sketch.estimate();
        assert!(
            (est - 3.0).abs() < 0.25,
            "estimate {:.3} deviates from true count 3",
            est
        );
    }

    #[test]
    fn test_custom_komihash_hasher() {
        use komihash::v5::KomiHasher;

        /// Simple BuildHasher that spawns a fresh Komihash with a fixed seed.
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

        let build_hasher = KomiBuildHasher {
            seed: 0x1234_5678_9abc_def0,
        };
        let mut sketch = ExaLogLog::new(2, 24, 8).unwrap();

        sketch
            .add_raw_with_build_hasher("apple", &build_hasher)
            .add_raw_with_build_hasher("banana", &build_hasher)
            .add_raw_with_build_hasher("cherry", &build_hasher)
            .add_raw_with_build_hasher("dragonfruit", &build_hasher);

        let est = sketch.estimate();
        assert!(
            (est - 4.0).abs() < 0.25,
            "estimate {:.3} deviates from true count 4",
            est
        );
    }

    #[test]
    fn test_custom_hasher_polymurhash() {
        use polymur_hash::PolymurHasher;
        use std::hash::BuildHasherDefault;

        type PolymurBuildHasher = BuildHasherDefault<PolymurHasher>;
        let build_hasher = PolymurBuildHasher::default();
        let mut sketch = ExaLogLog::new(2, 24, 8).unwrap();

        sketch
            .add_raw_with_build_hasher("apple", &build_hasher)
            .add_raw_with_build_hasher("banana", &build_hasher)
            .add_raw_with_build_hasher("cherry", &build_hasher)
            .add_raw_with_build_hasher("dragonfruit", &build_hasher);

        let est = sketch.estimate();
        assert!(
            (est - 4.0).abs() < 0.25,
            "estimate {:.3} deviates from true count 4",
            est
        );
    }

    #[test]
    fn test_custom_hasher_wyhash() {
        use wyhash::WyHasherBuilder;

        let build_hasher = WyHasherBuilder::default();
        let mut sketch = ExaLogLog::new(2, 24, 8).unwrap();

        sketch
            .add_raw_with_build_hasher("apple", &build_hasher)
            .add_raw_with_build_hasher("banana", &build_hasher)
            .add_raw_with_build_hasher("cherry", &build_hasher)
            .add_raw_with_build_hasher("dragonfruit", &build_hasher);

        let est = sketch.estimate();
        assert!(
            (est - 4.0).abs() < 0.25,
            "wyhash estimate {:.3} deviates too much from true count 4",
            est
        );
    }

    #[test]
    fn test_custom_hasher_t1ha2_atonce() {
        use t1ha::t1ha2_atonce;

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
                t1ha2_atonce(&self.buf, self.seed)
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

        let builder = T1ha2AtonceBuildHasher;
        let mut sketch = ExaLogLog::new(2, 24, 8).unwrap();

        sketch
            .add_raw_with_build_hasher("alpha", &builder)
            .add_raw_with_build_hasher("beta", &builder)
            .add_raw_with_build_hasher("gamma", &builder)
            .add_raw_with_build_hasher("delta", &builder);

        let est = sketch.estimate();
        assert!(
            (est - 4.0).abs() < 0.25,
            "t1ha2-atonce estimate {:.3} deviates too much from 4",
            est
        );
    }

    #[test]
    fn test_custom_hasher_tab_hash() {
        use tab_hash::Tab64Simple;

        let tabulation = Tab64Simple::new();
        let mut sketch = ExaLogLog::new(2, 24, 8).unwrap();

        sketch
            .add_raw_with_hash_fn(1u64, |value| tabulation.hash(*value))
            .add_raw_with_hash_fn(2u64, |value| tabulation.hash(*value))
            .add_raw_with_hash_fn(3u64, |value| tabulation.hash(*value))
            .add_raw_with_hash_fn(5u64, |value| tabulation.hash(*value));

        let est = sketch.estimate();
        assert!(
            (est - 4.0).abs() < 0.25,
            "tab-hash estimate {:.3} deviates too much from 4",
            est
        );
    }
}
