# rusty_math

This is a Rust library for mathematical operations.

## Features

- GCD
- LCM
- Factorial
- Check if a number is prime
- Find all primes numbers less than a number
- Permutation
- Combination
- Integration
- Differentiation
- Solve a linear equation
- Find roots of a polynomial using Falsi Reguli and Newton-Raphson methods
- Linear Regression
- R<pow>2</pow> score
- Mean squared error
- Accuracy
- Precision
- Confusion Matrix

## Installation

To use `rusty_math` in your project, add the following to your `Cargo.toml` file:

```toml
[dependencies]
rusty_math = "0.2.1"
```

## Modules

1. numbers: Functions to find the GCD, LCM, factorial, to check for prime, to find all prime numbers less than n.
2. calculus: Find the definite integral and slope of a function at a point.
3. Linear: Fit and predict a linear regression function.
4. Equations: Solve a system of linear equations and find a root of polynomials using Falsi Reguli and Newton-Raphson methods
5. Metrics: Score machine learning models.


## Usage

```rust
use rusty_math::gcd;

fn main() {
    let result = gcd(12, 15);
    println!("GCD of 12 and 15 is {}",result) ;
}
```

## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
