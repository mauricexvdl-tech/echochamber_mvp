//! Minimal complex number implementation using only std.

use std::ops::{Add, AddAssign, Mul, MulAssign};

#[cfg(test)]
use std::f64::consts::PI;

/// A complex number with real and imaginary parts.
#[derive(Clone, Copy, Debug)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const ZERO: Complex = Complex { re: 0.0, im: 0.0 };

    pub fn new(re: f64, im: f64) -> Self {
        Complex { re, im }
    }

    /// Create from polar coordinates: r * e^(i*theta)
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Complex {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    /// Magnitude (norm/amplitude) of the complex number
    pub fn norm(&self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    /// Power (squared magnitude) - the physically meaningful metric
    pub fn power(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Phase angle (argument) in radians [-PI, PI]
    pub fn arg(&self) -> f64 {
        self.im.atan2(self.re)
    }

    /// Scale by a real factor
    pub fn scale(&self, factor: f64) -> Self {
        Complex {
            re: self.re * factor,
            im: self.im * factor,
        }
    }

    /// Complex conjugate: (a + bi)* = (a - bi)
    pub fn conj(&self) -> Self {
        Complex {
            re: self.re,
            im: -self.im,
        }
    }
}

impl Add for Complex {
    type Output = Complex;
    fn add(self, other: Complex) -> Complex {
        Complex {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

impl AddAssign for Complex {
    fn add_assign(&mut self, other: Complex) {
        self.re += other.re;
        self.im += other.im;
    }
}

impl Mul for Complex {
    type Output = Complex;
    fn mul(self, other: Complex) -> Complex {
        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        Complex {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

impl MulAssign<f64> for Complex {
    fn mul_assign(&mut self, factor: f64) {
        self.re *= factor;
        self.im *= factor;
    }
}

impl std::fmt::Display for Complex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.im >= 0.0 {
            write!(f, "{:.6}+{:.6}i", self.re, self.im)
        } else {
            write!(f, "{:.6}{:.6}i", self.re, self.im)
        }
    }
}

/// Kuramoto order parameter: measures phase synchronization across oscillators.
/// Returns R ∈ [0, 1] where:
///   - R ≈ 0: completely desynchronized (random phases)
///   - R ≈ 1: fully synchronized (all phases aligned)
///
/// Formula: R = |1/N * Σ exp(i * φ_j)|
///
/// Reference: Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence.
pub fn kuramoto_order_parameter(phases: &[f64]) -> f64 {
    if phases.is_empty() {
        return 0.0;
    }

    let n = phases.len() as f64;

    // Sum of unit phasors: Σ exp(i * φ_j) = Σ (cos(φ_j) + i*sin(φ_j))
    let sum_re: f64 = phases.iter().map(|&phi| phi.cos()).sum();
    let sum_im: f64 = phases.iter().map(|&phi| phi.sin()).sum();

    // Order parameter R = |sum| / N
    let magnitude = (sum_re * sum_re + sum_im * sum_im).sqrt();
    magnitude / n
}

/// Weighted Kuramoto order parameter: accounts for signal amplitudes.
/// Stronger signals contribute more to the coherence measure.
///
/// Formula: R_w = |Σ A_j * exp(i * φ_j)| / Σ A_j
///
/// This is more physically meaningful for EchoChamber where
/// low-amplitude nodes shouldn't dominate the coherence metric.
pub fn kuramoto_weighted(amplitudes: &[f64], phases: &[f64]) -> f64 {
    if amplitudes.len() != phases.len() || amplitudes.is_empty() {
        return 0.0;
    }

    let total_amp: f64 = amplitudes.iter().sum();
    if total_amp < 1e-12 {
        return 0.0;
    }

    // Weighted sum of phasors: Σ A_j * exp(i * φ_j)
    let sum_re: f64 = amplitudes
        .iter()
        .zip(phases.iter())
        .map(|(&a, &phi)| a * phi.cos())
        .sum();
    let sum_im: f64 = amplitudes
        .iter()
        .zip(phases.iter())
        .map(|(&a, &phi)| a * phi.sin())
        .sum();

    let magnitude = (sum_re * sum_re + sum_im * sum_im).sqrt();
    magnitude / total_amp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_polar() {
        let c = Complex::from_polar(1.0, PI);
        assert!((c.re + 1.0).abs() < 1e-10);
        assert!(c.im.abs() < 1e-10);
    }

    #[test]
    fn test_cancellation() {
        let a = Complex::new(1.0, 0.0);
        let b = Complex::new(-1.0, 0.0);
        let sum = a + b;
        assert!(sum.norm() < 1e-10);
    }

    #[test]
    fn test_power() {
        let c = Complex::new(3.0, 4.0);
        assert!((c.power() - 25.0).abs() < 1e-10);
        assert!((c.norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_conj() {
        let c = Complex::new(3.0, 4.0);
        let cc = c.conj();
        assert!((cc.re - 3.0).abs() < 1e-10);
        assert!((cc.im + 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_kuramoto_synchronized() {
        // All phases aligned at 0 -> R = 1.0
        let phases = vec![0.0, 0.0, 0.0, 0.0];
        let r = kuramoto_order_parameter(&phases);
        assert!((r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kuramoto_desynchronized() {
        // Phases evenly spread around the circle -> R ≈ 0
        let phases = vec![0.0, PI / 2.0, PI, 3.0 * PI / 2.0];
        let r = kuramoto_order_parameter(&phases);
        assert!(r < 0.01, "Expected R ≈ 0 for evenly distributed phases, got {}", r);
    }

    #[test]
    fn test_kuramoto_partial_sync() {
        // Two clusters: 2 at 0, 2 at π -> R = 0
        let phases = vec![0.0, 0.0, PI, PI];
        let r = kuramoto_order_parameter(&phases);
        assert!(r < 0.01, "Expected R ≈ 0 for anti-phase clusters, got {}", r);

        // Three at 0, one at π -> R = 0.5
        let phases2 = vec![0.0, 0.0, 0.0, PI];
        let r2 = kuramoto_order_parameter(&phases2);
        assert!((r2 - 0.5).abs() < 0.01, "Expected R ≈ 0.5, got {}", r2);
    }

    #[test]
    fn test_kuramoto_weighted() {
        // All same amplitude, same phase -> R = 1.0
        let amps = vec![1.0, 1.0, 1.0, 1.0];
        let phases = vec![0.0, 0.0, 0.0, 0.0];
        let r = kuramoto_weighted(&amps, &phases);
        assert!((r - 1.0).abs() < 1e-10);

        // One strong signal dominates
        let amps2 = vec![10.0, 0.1, 0.1, 0.1];
        let phases2 = vec![0.0, PI, PI, PI]; // Strong at 0, weak at π
        let r2 = kuramoto_weighted(&amps2, &phases2);
        // Weighted average dominated by the strong signal at phase 0
        assert!(r2 > 0.9, "Expected strong coherence due to dominant signal, got {}", r2);
    }
}
