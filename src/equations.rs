//! # Equations
//! This module contains functions to solve equations and find roots of equations in Rust.
//! The functions include solving a system of linear equations, finding the root of a polynomial equation using the Regula Falsi method and the Newton-Raphson method.
//! # Examples
//! ```
//! use rusty_math::equations::solve_linear_eq;
//! let coeff = vec![vec![2.0, 1.0, -1.0], vec![-3.0, -1.0, 2.0], vec![-2.0, 1.0, 2.0]];
//! let rhs = vec![8.0, -11.0, -3.0];
//! let result = solve_linear_eq(coeff, rhs);
//! ```

use crate::calculus::differentiate;

/// Function to solve a system of linear equations
/// # Paramaters:
/// coeff: Vec<Vec<f64>> - A vector of vectors containing the coefficients of the equations  
/// rhs: Vec<f64> - A vector containing the right hand side of the equations  
/// # Returns:
/// Vec<f64> - A vector containing the solutions to the system of linear equations rounded upto 2 decimal places  
/// # Panics:
/// If the number of equations and the number of right hand side values do not match  
/// # Examples
/// ```
/// use rusty_math::equations::solve_linear_eq;
/// let coeff = vec![vec![2.0, 1.0, -1.0], vec![-3.0, -1.0, 2.0], vec![-2.0, 1.0, 2.0]];
/// let rhs = vec![8.0, -11.0, -3.0];
/// let result = solve_linear_eq(coeff, rhs);
/// assert_eq!(result, vec![2.0, 3.0, -1.0]);
/// ```
pub fn solve_linear_eq(coeff: Vec<Vec<f64>>, rhs: Vec<f64>) -> Vec<f64> {
    let n = coeff.len();
    if coeff.len() != rhs.len() {
        panic!("Number of equations and number of right hand side values do not match");
    }
    let mut coeff = coeff;
    let mut rhs = rhs;
    for i in 0..n {
        let mut max = i;
        for j in i + 1..n {
            if coeff[j][i].abs() > coeff[max][i].abs() {
                max = j;
            }
        }
        coeff.swap(i, max);
        rhs.swap(i, max);
        for j in i + 1..n {
            let ratio = coeff[j][i] / coeff[i][i];
            for k in i..n {
                coeff[j][k] -= ratio * coeff[i][k];
            }
            rhs[j] -= ratio * rhs[i];
        }
    }
    let mut res = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in i + 1..n {
            sum += coeff[i][j] * res[j];
        }
        res[i] = (((rhs[i] - sum) / coeff[i][i]) * 100.0).round() / 100.0;
    }
    res
}

fn f(coeff: &Vec<f64>, x: f64) -> f64 {
    let mut sum = 0.0;
    for i in 0..coeff.len() {
        sum += coeff[i] * x.powi((coeff.len() - 1 - i) as i32);
    }
    sum
}

/// Function to find the root of a polynomial equation using the Regula Falsi (False Position) method. The function assumes that the equation has at least one real root. Usually converges faster than the bisection method.  
/// Equation must be of the form a<sub>1</sub>x<sup>n</sup> + a<sub>2</sub>x<sup>n-1</sup> + ... + a<sub>n</sub>x + a<sub>n+1</sub> = 0
/// # Paramaters:
/// coeff: Vec<f64> - A vector containing the coefficients of the polynomial equation  
/// tol: f64 - The tolerance value  
/// # Returns:
/// f64 - A root of the polynomial equation  
/// # Examples
/// ```
/// use rusty_math::equations::regula_falsi;
/// let coeff = vec![1.0, -2.0, 1.0];
/// let result = regula_falsi(coeff, 0.0001);
/// ```
pub fn regula_falsi(coeff: Vec<f64>, tol: f64) -> f64 {
    let mut a = 0.0;
    let mut b = 0.0;
    let mut fa = f(&coeff, a);

    if fa == 0.0 {
        return a;
    }

    if fa > 0.0 {
        if f(&coeff, b + 0.1) < fa {
            while f(&coeff, b) > 0.0 {
                b += 0.1;
            }
        } else {
            while f(&coeff, b) > 0.0 {
                b -= 0.1;
            }
        }
    } else {
        if f(&coeff, b + 0.1) > fa {
            while f(&coeff, b) < 0.0 {
                b += 0.1;
            }
        } else {
            while f(&coeff, b) < 0.0 {
                b -= 0.1;
            }
        }
    }

    let mut fb = f(&coeff, b);
    let mut c = a - fa * (b - a) / (fb - fa);
    let mut fc = f(&coeff, c);
    while fc.abs() > tol {
        if fc > 0.0 {
            if fb > 0.0 {
                b = c;
                fb = fc;
            } else {
                a = c;
                fa = fc;
            }
        } else {
            if fb < 0.0 {
                b = c;
                fb = fc;
            } else {
                a = c;
                fa = fc;
            }
        }
        c = a - fa * (b - a) / (fb - fa);
        fc = f(&coeff, c);
    }
    c
}

/// Function to find the root of a polynomial equation using the Newton-Raphson method. The function assumes that the equation has at least one real root.  
/// Equation must be of the form a<sub>1</sub>x<sup>n</sup> + a<sub>2</sub>x<sup>n-1</sup> + ... + a<sub>n</sub>x + a<sub>n+1</sub> = 0  
/// Warning: The function may not converge if function is not well defined near root or derivative is zero or undefined at root.
/// # Paramaters:
/// coeff: &Vec<f64> - A reference to a vector containing the coefficients of the polynomial equation  
/// tol: f64 - The tolerance value for convergence before stopping the iteration  
/// # Returns:
/// f64 - A root of the polynomial equation  
/// # Examples
/// ```
/// use rusty_math::equations::newton_raphson;
/// let coeff = vec![1.0, 2.0, 1.0];
/// let result = newton_raphson(&coeff, 0.0001);
/// ```
pub fn newton_raphson(coeff: &Vec<f64>, tol: f64) -> f64 {
    let mut x = 0.0;
    let mut fx = f(&coeff, x);
    let mut dfx = differentiate(&|x| f(&coeff, x), x);
    if dfx == 0.0 {
        x += 0.01;
        dfx = differentiate(&|x| f(&coeff, x), x);
    }
    let mut x1 = x - fx / dfx;
    let mut fx1 = f(&coeff, x1);
    while (fx1 - fx).abs() > tol {
        x = x1;
        fx = fx1;
        dfx = differentiate(&|x| f(&coeff, x), x);
        if dfx == 0.0 {
            x += 0.01;
            dfx = differentiate(&|x| f(&coeff, x), x);
        }
        x1 = x - fx / dfx;
        fx1 = f(&coeff, x1);
    }
    x1
}
