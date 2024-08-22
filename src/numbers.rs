/// Function to find all the prime numbers less than a given number  
/// Uses the Sieve of Eratosthenes algorithm
/// # Paramaters:
/// num: usize - The positive number upto which the prime numbers are to be found
/// # Returns:
/// Vec<usize> - A vector containing all the prime numbers less than the given number
/// # Examples
/// ```
/// use rusty_math::numbers::primes;
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
/// # Paramaters:
/// num: usize - The positive number to be checked
/// # Returns:
/// bool - True if the number is prime, False otherwise
/// # Examples
/// ```
/// use rusty_math::numbers::isprime;
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
/// # Paramaters:
/// num: usize - The positive number whose factorial is to be found
/// # Returns:
/// usize - The factorial of the given number
/// # Examples
/// ```
/// use rusty_math::numbers::factorial;
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
/// # Paramaters:
/// a: i32 - The first number
/// b: i32 - The second number
/// # Returns:
/// i32 - The greatest common divisor of the two numbers
/// # Panics:
/// If the input is negative
/// # Examples
/// ```
/// use rusty_math::numbers::gcd;
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
/// # Paramaters:
/// a: i32 - The first number
/// b: i32 - The second number
/// # Returns:
/// i32 - The least common multiple of the two numbers
/// # Panics:
/// If the input is negative
/// # Examples
/// ```
/// use rusty_math::numbers::lcm;
/// let result = lcm(12, 15);
/// ```
/// Panics if the input is negative
pub fn lcm(a: i32, b: i32) -> i32 {
    if a < 0 || b < 0 {
        panic!("Negative numbers are not allowed");
    }
    a * b / gcd(a, b)
}