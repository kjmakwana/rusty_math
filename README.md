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

## Installation

To use `rusty_math` in your project, add the following to your `Cargo.toml` file:

```toml
[dependencies]
rusty_math = "0.1.0"
```

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
