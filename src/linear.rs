//! # Regression
//! The `linear` module contains the `LinearRegression` struct.   
//! You can fit a linear regression model in Rust to the training data and use the model to predict the target values for test data.
//! # Examples
//! ```
//! use rusty_math::linear::LinearRegression;
//! let model = LinearRegression::new();
//! ```
//!

/// # Linear Regression
/// Fits a linear regression model to the training data. The model is of the form y = w1*x1 + w2*x2 + ... + wn*xn + b.  
/// The model is fit using gradient descent. The model can be used to predict the target values for test data.
pub struct LinearRegression {
    pub weights: Vec<f64>,
    pub intercept: f64,
}

impl LinearRegression {
    /// Create a new LinearRegression object
    /// # Returns
    /// LinearRegression - A new LinearRegression object
    /// # Examples
    /// ```
    /// use rusty_math::linear::LinearRegression;
    /// let model = LinearRegression::new();
    /// ```
    pub fn new() -> LinearRegression {
        LinearRegression {
            weights: Vec::new(),
            intercept: 0.0,
        }
    }

    /// Fit the Linear Regression model
    /// # Parameters
    /// x_train: &Vec<Vec<f64>> - A reference to a vector of vectors containing the training data  
    /// y_train: &Vec<f64> - A reference to a vector containing the target values  
    /// lr: f64 - The learning rate  
    /// n_iter: i32 - The number of iterations  
    /// # Examples
    /// ```
    /// use rusty_math::linear::LinearRegression;
    /// let mut model = LinearRegression::new();
    /// let x_train = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
    /// let y_train = vec![3.0, 4.0, 5.0];
    /// model.fit(&x_train, &y_train, 0.01, 1000);
    /// ```
    /// # Panics
    /// If the number of samples in the training data does not match the number of samples in the target values
    pub fn fit(&mut self, x_train: &Vec<Vec<f64>>, y_train: &Vec<f64>, lr: f64, n_iter: i32) {
        let n_samples = x_train.len();
        let n_features = x_train[0].len();
        self.weights = vec![1.0; n_features];
        if n_samples != y_train.len() {
            panic!("Number of samples in training data does not match the number of samples in target values");
        }

        for _ in 0..n_iter {
            let mut y_pred = vec![0.0; n_samples];
            for i in 0..n_samples {
                for j in 0..n_features {
                    y_pred[i] += self.weights[j] * x_train[i][j];
                }
                y_pred[i] += self.intercept;
            }
            let mut dw = vec![0.0; n_features];
            let mut di = 0.0;

            for i in 0..n_samples {
                for j in 0..n_features {
                    dw[j] += (y_pred[i] - y_train[i]) * x_train[i][j];
                }
                di += y_pred[i] - y_train[i];
            }

            self.intercept -= lr * di / n_samples as f64;

            for i in 0..n_features {
                self.weights[i] -= lr * dw[i] / n_samples as f64;
            }
        }
    }

    /// Predict the target values
    /// # Parameters
    /// x_test: &Vec<Vec<f64> - A reference to a vector of vectors containing the test data  
    /// # Returns
    /// Vec<f64> - A vector containing the predicted target values
    /// # Examples
    /// ```
    /// use rusty_math::linear::LinearRegression;
    /// let mut model = LinearRegression::new();
    /// let x_train = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
    /// let y_train = vec![3.0, 4.0, 5.0];
    /// model.fit(&x_train, &y_train, 0.01, 1000);
    /// let x_test = vec![vec![4.0, 5.0], vec![5.0, 6.0]];
    /// let y_pred = model.predict(&x_test);
    /// ```
    /// # Panics
    /// If the number of features in the test data does not match the number of features in the training data
    pub fn predict(&self, x_test: &Vec<Vec<f64>>) -> Vec<f64> {
        let n_samples = x_test.len();
        let n_features = x_test[0].len();
        if n_features != self.weights.len() {
            panic!("Number of features in test data does not match the number of features in training data");
        }
        let mut y_pred = vec![0.0; n_samples];

        for i in 0..n_samples {
            for j in 0..n_features {
                y_pred[i] += self.weights[j] * x_test[i][j];
            }
            y_pred[i] += self.intercept;
        }
        y_pred
    }

    /// Get the weights of the model
    /// # Returns
    /// Vec<f64> - A vector containing the weights of the model
    /// # Examples
    /// ```
    /// use rusty_math::linear::LinearRegression;
    /// let mut model = LinearRegression::new();
    /// let x_train = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
    /// let y_train = vec![3.0, 4.0, 5.0];
    /// model.fit(&x_train, &y_train, 0.01, 1000);
    /// let weights = model.get_weights();
    /// ```
    pub fn get_weights(&self) -> Vec<f64> {
        self.weights.clone()
    }

    /// Get the intercept of the model
    /// # Returns
    /// f64 - The intercept of the model
    /// # Examples
    /// ```
    /// use rusty_math::linear::LinearRegression;
    /// let mut model = LinearRegression::new();
    /// let x_train = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
    /// let y_train = vec![3.0, 4.0, 5.0];
    /// model.fit(&x_train, &y_train, 0.01, 1000);
    /// let intercept = model.get_intercept();
    /// ```
    pub fn get_intercept(&self) -> f64 {
        self.intercept
    }
}
