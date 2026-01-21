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
}
