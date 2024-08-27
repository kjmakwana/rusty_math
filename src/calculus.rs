//! # Calculus
//! This module contains functions related to calculus.  
//! The functions include finding the value of a definite integral using the trapezoidal rule, finding the value of the derivative of a function at a point in Rust.
//! # Examples
//! ```
//! use rusty_math::calculus;
//! let result = calculus::integrate(&|x| x*x, 0.0, 3.0);
//! ```

/// Function to find the value of a definite integral using the trapezoidal rule
/// # Paramaters:
/// f: `&dyn Fn(f64) -> f64` - The function whose integral is to be found  
/// a: `f64` - The lower limit of the integral  
/// b: `f64` - The upper limit of the integral
/// # Returns:
/// `f64` - The value of the definite integral
/// # Examples
/// ```
/// use rusty_math::calculus::integrate;
/// let result = integrate(&|x| x*x, 0.0, 3.0);
/// ```
pub fn integrate(f: &dyn Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    let w = 0.001;
    let n = ((b - a) / w).ceil() as usize;
    println!("{}", n);
    let mut sum: f64 = 0.0;
    for i in 0..n {
        let x1 = a + i as f64 * w;
        let x2 = a + (i + 1) as f64 * w;
        sum += (f(x1) + f(x2)) * w / 2.0;
    }
    sum
}

/// Function to find the value of the derivative of a function at a point
/// # Paramaters:
/// f: `&dyn Fn(f64) -> f64` - The function whose derivative is to be found  
/// x: `f64` - The point at which the derivative is to be found
/// # Returns:
/// `f64` - The value of the derivative at the given point
/// # Examples
/// ```
/// use rusty_math::calculus::differentiate;
/// let result = differentiate(&|x| x*x, 3.0);
/// ```
pub fn differentiate(f: &dyn Fn(f64) -> f64, x: f64) -> f64 {
    let h = 0.0001;
    (f(x + h) - f(x - h)) / (2.0 * h)
}
