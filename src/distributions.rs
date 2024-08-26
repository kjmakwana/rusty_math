//! # Distributions
//! This module contains functions related to gamma and beta distributions in Rust.  
//! The functions include finding the gamma function, beta function, permutation and combination in Rust.
//! # Examples
//! ```
//! use rusty_math::distributions::{permutation, combination, gamma, beta};
//! let result = permutation(5, 2);
//! ```

use crate::{calculus, numbers};
/// Function to find the number of ways to choose and arrange r items from n items
/// # Paramaters:
/// n: usize - The total number of items  
/// r: usize - The number of items to be arranged
/// # Returns:
/// usize - The number of ways to choose and arrange r items from n items
/// # Examples
/// ```
/// use rusty_math::distributions::permutation;
/// let result = permutation(5, 2);
/// ```
pub fn permutation(n: usize, r: usize) -> usize {
    if r > n {
        return 0;
    }
    let mut res = 1;
    for i in n - r + 1..=n {
        res *= i;
    }
    res
}

/// Function to find the number of ways to choose r items from n items
/// # Paramaters:
/// n: usize - The total number of items  
/// r: usize - The number of items to be chosen
/// # Returns:
/// usize - The number of ways to choose r items from n items
/// # Examples
/// ```
/// use rusty_math::distributions::combination;
/// let result = combination(5, 2);
/// ```
pub fn combination(n: usize, r: usize) -> usize {
    if r > n {
        return 0;
    }
    permutation(n, r) / numbers::factorial(r)
}

/// Function to find the gamma function of a number
/// # Paramaters:
/// n: f64 - The number whose gamma function is to be found
/// # Returns:
/// f64 - The gamma function of the given number
/// # Examples
/// ```
/// use rusty_math::distributions::gamma;
/// let result = gamma(5.8);
/// ```
pub fn gamma(n: f64) -> f64 {
    let gamma_function = |x: f64| (std::f64::consts::E).powf(-x) * x.powf(n - 1.0);

    calculus::integrate(&gamma_function, 0.0, 1000 as f64)
}

/// Function to find the beta function of two numbers
/// # Paramaters:
/// x: f64 - The first number
/// y: f64 - The second number
/// # Returns:
/// f64 - The beta function of the two numbers
/// # Examples
/// ```
/// use rusty_math::distributions::beta;
/// let result = beta(5.5, 3.52);
/// ```
pub fn beta(x: f64, y: f64) -> f64 {
    gamma(x) * gamma(y) / gamma(x + y)
}
