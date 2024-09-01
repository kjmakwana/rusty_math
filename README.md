# rusty_math

This is a Rust library for mathematical, statistical and machine learning operations.

## Version 0.7.0
New version now supports K-Means clustering. See clustering module for more details.

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
- Logistic Regression
- Lasso and Ridge Regression
- Logistic Regression
- Naive Bayes Classifier
- K-Nearest Neighbors
- K-Means Clustering
- R<pow>2</pow> score
- Mean squared error
- Accuracy
- Precision
- Confusion Matrix
- Recall
- F1 score

## Installation

To use `rusty_math` in your project, add the following to your `Cargo.toml` file:

```toml
[dependencies]
rusty_math = "0.7.0"
```

## Modules

1. numbers: Functions to find the GCD, LCM, factorial, to check for prime, to find all prime numbers less than n.
2. calculus: Find the definite integral and slope of a function at a point.
3. Linear: Fit and predict a several types linear functions.
4. Equations: Solve a system of linear equations and find a root of polynomials using Falsi Reguli and Newton-Raphson methods
5. naive_bayes: Fit a Gaussian Naive Bayes classifier and predict classes.
6. knn: Fit and predict target values using K-nearest neighbors classification and regression models.
7. clustering: Clustering Algorithms like KMeans.
8. Metrics: Score machine learning models.  
  
See detailed documation for list of functionalities in each module.  

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
  
In case of any questions or feedback feel free to contact me at kjmakwana00@gmail.com  

## License

This project is licensed under the [MIT License](LICENSE).
