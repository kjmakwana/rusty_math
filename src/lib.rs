
//! rusty_math is a collection of mathematical functions implemented in Rust
//! Exampe usage:
//! ```
//! use rusty_math::{gcd, lcm, factorial, isprime, primes, permutation, combination};
//! let result = gcd(12, 15);
//! ```
//! 






/// Function to find all the prime numbers less than a given number  
/// Uses the Sieve of Eratosthenes algorithm
/// # Examples
/// ```
/// use rusty_math::primes;
/// let result = primes(10);
/// ```
pub fn primes(num: usize) -> Vec<usize> {

    let mut nums = vec![true; num + 1];
    let mut i = 2;
    loop {
        if i*i >= num {
            break;
        }
        if nums[i] == true {
            for j in (i*i..num + 1).step_by(i) {
                nums[j] = false;
            }
        }
        i+=1;
    }
    let mut primes = vec![];
    for i in 2..num {
        if nums[i] == true {
            primes.push(i);
        }
    }
    primes


}


/// Function to check if a number is prime
/// # Examples
/// ```
/// use rusty_math::isprime;
/// let result = isprime(7);
/// ```
pub fn isprime(num: usize) -> bool {
    if num <= 1 {
        return false;
    }
    for i in 2..num {
        if num % i == 0 {
            return false;
        }
    }
    true
}


/// Function to find the factorial of a number
/// # Examples
/// ```
/// use rusty_math::factorial;
/// let result = factorial(5);
/// ```
/// Panics if the input is negative
pub fn factorial(num: usize) -> usize {
    if num == 0 {
        return 1;
    }
    num * factorial(num - 1)
}

/// Function to find the greatest common divisor of two numbers
/// # Examples
/// ```
/// use rusty_math::gcd;
/// let result = gcd(12, 15);
/// ```
/// Panics if the input is negative
pub fn gcd(mut a: i32, mut b: i32) -> i32 {
    if a < 0 || b < 0 {
        panic!("Negative numbers are not allowed");
    }
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}


/// Function to find the least common multiple of two numbers
/// # Examples
/// ```
/// use rusty_math::lcm;
/// let result = lcm(12, 15);
/// ```
/// Panics if the input is negative
pub fn lcm(a: i32, b: i32) -> i32 {
    if a < 0 || b < 0 {
        panic!("Negative numbers are not allowed");
    }
    a * b / gcd(a, b)
}


/// Function to find the number of ways to choose and arrange r items from n items
/// # Examples
/// ```
/// use rusty_math::permutation;
/// let result = permutation(5, 2);
/// ```
pub fn permutation(n: usize, r: usize) -> usize {
    let mut res = 1;
    for i in n-r+1..=n {
        res*=i;
    }
    res
}

/// Function to find the number of ways to choose r items from n items
/// # Examples
/// ```
/// use rusty_math::combination;
/// let result = combination(5, 2);
/// ```
pub fn combination(n: usize, r: usize) -> usize {
    permutation(n, r) / factorial(r)
}


