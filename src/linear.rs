//! # Regression
//! The `linear` module structs for fitting linear models.   
//! You can fit a linear regression or polynomial model in Rust to the training data and use the model to predict the target values for test data.
//! You can also fit a ridge or lasso regression model to the training data and use the model to predict the target values for test data.
//! Logistic regression is also supported. You can fit a logistic regression model to the training data and use the model to predict the target values and their probabilities for test data.
//! # Examples
//! ```
//! use rusty_math::linear::LinearRegression;
//! let model = LinearRegression::new();
//! ```

use std::collections::HashMap;

use crate::metrics::{r2_score,accuracy};

/// # Linear Regression
/// Fits a linear regression model to the training data. The model is of the form y = b + a<sub>1</sub>x<sub>1</sub> + a<sub>2</sub>x<sub>2</sub> + ... + a<sub>n</sub>x<sub>n</sub>.  
/// The model is fit using gradient descent. The model can be used to predict the target values for test data.
pub struct LinearRegression {
    pub weights: Vec<f64>,
    pub intercept: f64,
}

impl LinearRegression {
    /// Create a new LinearRegression object
    /// # Returns
    /// `LinearRegression` - A new LinearRegression object
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
    /// x_train: `&Vec<Vec<f64>>` - A reference to a vector of vectors containing the training data  
    /// y_train: `&Vec<f64>` - A reference to a vector containing the target values  
    /// lr: `f64` - The learning rate  
    /// n_iter: `i32` - The number of iterations  
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
    /// x_test:`&Vec<Vec<f64>` - A reference to a vector of vectors containing the test data  
    /// # Returns
    /// `Vec<f64>` - A vector containing the predicted target values
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
    /// `Vec<f64>` - A vector containing the weights of the model
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
    /// `f64` - The intercept of the model
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

    /// Get the R<sup>2</sup> score of the model on the test data
    /// # Parameters
    /// x_test: `&Vec<Vec<f64>>` - A reference to a vector of vectors containing the test data
    /// y_test: `&Vec<f64>` - A reference to a vector containing the target values
    /// # Returns
    /// `f64` - The R<sup>2</sup> score of the model on the test data
    /// # Examples
    /// ```
    /// use rusty_math::linear::LinearRegression;
    /// let mut model = LinearRegression::new();
    /// model.fit(&x_train, &y_train, 0.01, 1000);
    /// let score = model.score(&x_test, &y_test);
    /// ```
    pub fn score(&self, x_test: &Vec<Vec<f64>>, y_test: &Vec<f64>) -> f64 {
        let y_pred = self.predict(x_test);
        r2_score(y_test, &y_pred)
    }
}

/// # Polynomial Regression
/// Fits a polynomial regression model to the training data. The model is of the form y = b + a<sub>1</sub>x + a<sub>2</sub>x<sup>2</sup> + ... + a<sub>n</sub>x<sup>n</sup>.  
/// The model is fit using gradient descent. The model can be used to predict the target values for test data.   
/// The degree of the polynomial can be set by the user.
/// # Examples
/// ```
/// use rusty_math::linear::PolynomialRegression;
/// let model = PolynomialRegression::new(2); // set the degree of the polynomial to 2
/// ```
pub struct PolynomialRegression {
    pub weights: Vec<f64>,
    pub intercept: f64,
    pub degree: usize,
}

impl PolynomialRegression {
    /// Create a new PolynomialRegression object
    /// # Parameters
    /// degree: `usize` - The degree of the polynomial
    /// # Returns
    /// `PolynomialRegression` - A new PolynomialRegression object
    pub fn new(degree: usize) -> PolynomialRegression {
        PolynomialRegression {
            weights: Vec::new(),
            intercept: 0.0,
            degree: degree,
        }
    }

    /// Expand the features of the training data to include polynomial features. Called automatically by the fit and predict method.
    /// ## Warning
    /// Do not pass the expanded features to the fit method. The fit method will automatically expand the features.
    /// # Parameters
    /// x: `&Vec<Vec<f64>` - A reference to a vector of vectors containing the training data  
    /// # Returns
    /// `Vec<Vec<f64>` - A vector of vectors containing the expanded features  
    /// # Examples
    /// ```
    /// use rusty_math::linear::PolynomialRegression;
    /// let model = PolynomialRegression::new(2);
    /// let x_train = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
    /// let x_poly = model.expand_features(&x_train);
    /// ```
    pub fn expand_features(&self, x: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        use itertools::Itertools;
        let mut x_poly = Vec::new();
        for row in x {
            let mut row_poly = vec![1.0];
            row_poly.extend(row.clone());

            for d in 2..=self.degree {
                let mut combis = Vec::new();

                for combo in (0..row.len()).combinations_with_replacement(d) {
                    let mut product = 1.0;
                    for &i in &combo {
                        product *= row[i];
                    }
                    combis.push(product);
                }
                row_poly.extend(combis);
            }
            x_poly.push(row_poly);
        }
        x_poly
    }

    /// Fit the Polynomial Regression model. The model is fit using gradient descent.  
    /// # Parameters
    /// x_train: `&Vec<Vec<f64>>` - A reference to a vector of vectors containing the training data  
    /// y_train: `&Vec<f64>` - A reference to a vector containing the target values  
    /// lr: `f64 - The learning rate  
    /// n_iter: `i32` - The number of iterations
    /// # Examples
    /// ```
    /// use rusty_math::linear::PolynomialRegression;
    /// let mut model = PolynomialRegression::new(2);
    /// let x_train = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
    /// let y_train = vec![3.0, 4.0, 5.0];
    /// model.fit(&x_train, &y_train, 0.01, 1000);
    /// ```
    /// # Panics
    /// If the number of samples in the training data does not match the number of samples in the target values
    pub fn fit(&mut self, x_train: &Vec<Vec<f64>>, y_train: &Vec<f64>, lr: f64, n_iter: i32) {
        let n_samples = x_train.len();
        if n_samples != y_train.len() {
            panic!("Number of samples in training data does not match the number of samples in target values");
        }
        let x_poly = self.expand_features(x_train);
        let n_features_poly = x_poly[0].len();
        self.weights = vec![1.0; n_features_poly];
        for _ in 0..n_iter {
            let mut y_pred = vec![0.0; n_samples];
            for i in 0..n_samples {
                for j in 0..n_features_poly {
                    y_pred[i] += self.weights[j] * x_poly[i][j];
                }
            }
            let mut dw = vec![0.0; n_features_poly];
            let mut di = 0.0;
            for i in 0..n_samples {
                for j in 0..n_features_poly {
                    dw[j] += (y_pred[i] - y_train[i]) * x_poly[i][j];
                }
                di += y_pred[i] - y_train[i];
            }
            self.intercept -= lr * di / n_samples as f64;
            for i in 0..n_features_poly {
                self.weights[i] -= lr * dw[i] / n_samples as f64;
            }
        }
    }

    /// Predict the target values.
    /// # Parameters
    /// x_test: `Vec<Vec<f64>` - A reference to a vector of vectors containing the test data
    /// # Returns
    /// `Vec<f64>` - A vector containing the predicted target values
    /// # Examples
    /// ```
    /// use rusty_math::linear::PolynomialRegression;
    /// let mut model = PolynomialRegression::new(2);
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
        let x_poly = self.expand_features(&x_test);
        if x_poly[0].len() != self.weights.len() {
            panic!("Number of features in test data does not match the number of features in training data");
        }
        let mut y_pred = vec![0.0; n_samples];
        for i in 0..n_samples {
            for j in 0..x_poly[0].len() {
                y_pred[i] += self.weights[j] * x_poly[i][j];
            }
            y_pred[i] += self.intercept;
        }
        y_pred
    }

    /// Get the weights of the model
    /// # Returns
    /// `Vec<f64>` - A vector containing the weights of the model
    /// # Examples
    /// ```
    /// use rusty_math::linear::PolynomialRegression;
    /// let mut model = PolynomialRegression::new(2);
    /// model.fit(&x_train, &y_train, 0.01, 1000);
    /// let weights = model.get_weights();
    /// ```
    pub fn get_weights(&self) -> Vec<f64> {
        self.weights.clone()
    }

    /// Get the intercept of the model
    /// # Returns
    /// `f64` - The intercept of the model
    /// # Examples
    /// ```
    /// use rusty_math::linear::PolynomialRegression;
    /// let mut model = PolynomialRegression::new(2);
    /// model.fit(&x_train, &y_train, 0.01, 1000);
    /// let intercept = model.get_intercept();
    /// ```
    pub fn get_intercept(&self) -> f64 {
        self.intercept
    }

    /// Get the R<sup>2</sup> score of the model on the test data
    /// # Parameters
    /// x_test: `&Vec<Vec<f64>>` - A reference to a vector of vectors containing the test data
    /// y_test: `&Vec<f64>` - A reference to a vector containing the target values
    /// # Returns
    /// `f64` - The R<sup>2</sup> score of the model on the test data
    /// # Examples
    /// ```
    /// use rusty_math::linear::PolynomialRegression;
    /// let mut model = PolynomialRegression::new(2);
    /// model.fit(&x_train, &y_train, 0.01, 1000);
    /// let score = model.score(&x_test, &y_test);
    /// ```
    pub fn score(&self, x_test: &Vec<Vec<f64>>, y_test: &Vec<f64>) -> f64 {
        let y_pred = self.predict(&x_test);
        r2_score(&y_test, &y_pred)
    }
}

/// # Ridge Regression
/// Fits a ridge regression model to the training data. The model is of the form y = b + a<sub>1</sub>x<sub>1</sub> + a<sub>2</sub>x<sub>2</sub> + ... + a<sub>n</sub>x<sub>n</sub>.
/// The model is fit using gradient descent with L2 regularization. The model can be used to predict the target values for test data.
/// # Examples
/// ```
/// use rusty_math::linear::RidgeRegression;
/// let model = RidgeRegression::new(0.01);
/// ```
pub struct RidgeRegression {
    pub weights: Vec<f64>,
    pub intercept: f64,
    pub alpha: f64,
}

impl RidgeRegression {
    /// Create a new RidgeRegression object
    /// # Parameters
    /// alpha: `f64` - The regularization parameter
    /// # Returns
    /// `RidgeRegression` - A new RidgeRegression object
    /// # Examples
    /// ```
    /// use rusty_math::linear::RidgeRegression;
    /// let model = RidgeRegression::new(0.01);
    /// ```
    pub fn new(alpha: f64) -> RidgeRegression {
        RidgeRegression {
            weights: Vec::new(),
            intercept: 0.0,
            alpha: alpha,
        }
    }

    /// Fit the Ridge Regression model. The model is fit using gradient descent with L2 regularization.
    /// # Parameters
    /// x_train: `&Vec<Vec<f64>>` - A reference to a vector of vectors containing the training data  
    /// y_train: `&Vec<f64>` - A reference to a vector containing the target values  
    /// lr: `f64` - The learning rate  
    /// n_iter: `i32` - The number of iterations  
    /// # Examples
    /// ```
    /// use rusty_math::linear::RidgeRegression;
    /// let mut model = RidgeRegression::new(0.01);
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
                self.weights[i] -=
                    lr * (dw[i] / n_samples as f64 + self.alpha * self.weights[i].powi(2));
            }
        }
    }

    /// Predict the target values.
    /// # Parameters
    /// x_test: `Vec<Vec<f64>` - A reference to a vector of vectors containing the test data  
    /// # Returns
    /// `Vec<f64>` - A vector containing the predicted target values  
    /// # Examples
    /// ```
    /// use rusty_math::linear::RidgeRegression;
    /// let mut model = RidgeRegression::new(0.01);
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
    /// `Vec<f64>` - A vector containing the weights of the model  
    /// # Examples
    /// ```
    /// use rusty_math::linear::RidgeRegression;
    /// let mut model = RidgeRegression::new(0.01);
    /// model.fit(&x_train, &y_train, 0.01, 1000);
    /// let weights = model.get_weights();
    /// ```
    pub fn get_weights(&self) -> Vec<f64> {
        self.weights.clone()
    }

    /// Get the intercept of the model
    /// # Returns
    /// `f64` - The intercept of the model
    /// # Examples
    /// ```
    /// use rusty_math::linear::RidgeRegression;
    /// let mut model = RidgeRegression::new(0.01);
    /// model.fit(&x_train, &y_train, 0.01, 1000);
    /// let intercept = model.get_intercept();
    /// ```
    pub fn get_intercept(&self) -> f64 {
        self.intercept
    }

    /// Get the R<sup>2</sup> score of the model on the test data
    /// # Parameters
    /// x_test: `&Vec<Vec<f64>>` - A reference to a vector of vectors containing the test data  
    /// y_test: `&Vec<f64>` - A reference to a vector containing the target values
    /// # Returns
    /// `f64` - The R<sup>2</sup> score of the model on the test data
    /// # Examples
    /// ```
    /// use rusty_math::linear::RidgeRegression;
    /// let mut model = RidgeRegression::new(0.01);
    /// model.fit(&x_train, &y_train, 0.01, 1000);
    /// let score = model.score(&x_test, &y_test);
    /// ```
    pub fn score(&self, x_test: &Vec<Vec<f64>>, y_test: &Vec<f64>) -> f64 {
        let y_pred = self.predict(x_test);
        r2_score(y_test, &y_pred)
    }
}

/// # Lasso Regression
/// Fits a lasso regression model to the training data. The model is of the form y = b + a<sub>1</sub>x<sub>1</sub> + a<sub>2</sub>x<sub>2</sub> + ... + a<sub>n</sub>x<sub>n</sub>.
/// The model is fit using gradient descent with L1 regularization. The model can be used to predict the target values for test data.
/// # Examples
/// ```
/// use rusty_math::linear::LassoRegression;
/// let model = LassoRegression::new(0.01);
/// ```
pub struct LassoRegression {
    pub weights: Vec<f64>,
    pub intercept: f64,
    pub alpha: f64,
}

impl LassoRegression {
    /// Create a new LassoRegression object.
    /// # Parameters
    /// alpha: `f64` - The regularization parameter
    /// # Returns
    /// `LassoRegression` - A new LassoRegression object
    /// # Examples
    /// ```
    /// use rusty_math::linear::LassoRegression;
    /// let model = LassoRegression::new(0.01);
    /// ```
    pub fn new(alpha: f64) -> LassoRegression {
        LassoRegression {
            weights: Vec::new(),
            intercept: 0.0,
            alpha: alpha,
        }
    }

    /// Fit the Lasso Regression model. The model is fit using gradient descent with L1 regularization.
    /// # Parameters
    /// x_train: `&Vec<Vec<f64>>` - A reference to a vector of vectors containing the training data  
    /// y_train: `&Vec<f64>` - A reference to a vector containing the target values  
    /// lr: `f64` - The learning rate  
    /// n_iter: `i32` - The number of iterations  
    /// # Examples
    /// ```
    /// use rusty_math::linear::LassoRegression;
    /// let mut model = LassoRegression::new(0.01);
    /// let x_train = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
    /// let y_train = vec![3.0, 4.0, 5.0];
    /// model.fit(&x_train, &y_train, 0.01, 1000);
    /// ```
    /// # Panics
    /// If the number of samples in the training data does not match the number of samples in the target values.  
    pub fn fit(&mut self, x_train: &Vec<Vec<f64>>, y_train: Vec<f64>, lr: f64, n_iters: i32) {
        let n_samples = x_train.len();
        let n_features = x_train[0].len();
        self.weights = vec![1.0; n_features];
        if n_samples != y_train.len() {
            panic!("Number of samples in training data does not match the number of samples in target values");
        }

        for _ in 0..n_iters {
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
                self.weights[i] -=
                    lr * (dw[i] / n_samples as f64 + self.alpha * self.weights[i].signum());
            }
        }
    }

    /// Predict the target values from the test data.  
    /// # Parameters
    /// x_test: `Vec<Vec<f64>` - A reference to a vector of vectors containing the test data
    /// # Returns
    /// `Vec<f64>` - A vector containing the predicted target values
    /// # Examples
    /// ```
    /// use rusty_math::linear::LassoRegression;
    /// let mut model = LassoRegression::new(0.01);
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
    /// `Vec<f64>` - A vector containing the weights of the model
    /// # Examples
    /// ```
    /// use rusty_math::linear::LassoRegression;
    /// let mut model = LassoRegression::new(0.01);
    /// model.fit(&x_train, &y_train, 0.01, 1000);
    /// let weights = model.weights();
    /// ```
    pub fn weights(&self) -> Vec<f64> {
        self.weights.clone()
    }

    /// Get the intercept of the model
    /// # Returns
    /// `f64` - The intercept of the model
    /// # Examples
    /// ```
    /// use rusty_math::linear::LassoRegression;
    /// let mut model = LassoRegression::new(0.01);
    /// model.fit(&x_train, &y_train, 0.01, 1000);
    /// let intercept = model.intercept();
    /// ```
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    /// Get the R<sup>2</sup> score of the model on the test data.
    /// # Parameters
    /// x_test: `&Vec<Vec<f64>>` - A reference to a vector of vectors containing the test data  
    /// y_test: `&Vec<f64>` - A reference to a vector containing the target values      
    /// # Returns  
    /// `f64` - The R<sup>2</sup> score of the model on the test data
    /// # Examples
    /// ```
    /// use rusty_math::linear::LassoRegression;
    /// let mut model = LassoRegression::new(0.01);
    /// model.fit(&x_train, &y_train, 0.01, 1000);
    /// let score = model.score(&x_test, &y_test);
    /// ```
    pub fn score(&self, x_test: &Vec<Vec<f64>>, y_test: &Vec<f64>) -> f64 {
        let y_pred = self.predict(x_test);
        r2_score(y_test, &y_pred)
    }
}



/// # Logistic Regression
/// Fits a logistic regression model to the training data. The model is of the form y = 1 / (1 + e<sup>-(b + a<sub>1</sub>x<sub>1</sub> + a<sub>2</sub>x<sub>2</sub> + ... + a<sub>n</sub>x<sub>n</sub>)).
/// The model is fit using gradient descent of likelihood. The model can be used to predict the target values for test data.
/// # Examples
/// ```
/// use rusty_math::linear::LogisticRegression;
/// let model = LogisticRegression::new();
/// ```
/// Fields:
/// weights: `Vec<f64>` - The weights of the model
/// intercept: `f64` - The intercept of the model
pub struct LogisticRegression {
    pub weights: Vec<f64>,
    pub intercept: f64,
}

impl LogisticRegression{

    /// Create a new LogisticRegression object
    /// # Returns
    /// `LogisticRegression` - A new LogisticRegression object
    /// # Examples
    /// ```
    /// use rusty_math::linear::LogisticRegression;
    /// let model = LogisticRegression::new();
    /// ```
    pub fn new() -> LogisticRegression{
        LogisticRegression{
            weights: Vec::new(),
            intercept: 0.0,
        }
    }


    fn sigmoid(x: f64) -> f64{
        1.0 / (1.0 + (-x).exp())
    }


    /// Fit the Logistic Regression model. The model is fit using gradient descent of likelihood.
    /// # Parameters
    /// x_train: `&Vec<Vec<f64>>` - A reference to a vector of vectors containing the training data  
    /// y_train: `&Vec<f64>` - A reference to a vector containing the target values  
    /// lr: `f64` - The learning rate  
    /// n_iter: `i32` - The number of iterations  
    /// # Examples
    /// ```
    /// use rusty_math::linear::LogisticRegression;
    /// let mut model = LogisticRegression::new();
    /// let x_train = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
    /// let y_train = vec![0.0, 1.0, 0.0];
    /// model.fit(&x_train, &y_train, 0.01, 1000);
    /// ```
    /// # Panics
    /// If the number of samples in the training data does not match the number of samples in the target values  
    pub fn fit(&mut self, x_train: &Vec<Vec<f64>>, y_train: &Vec<f64>, lr: f64, n_iter: i32){
        let n_samples = x_train.len();
        let n_features = x_train[0].len();
        self.weights = vec![1.0; n_features];
        if n_samples != y_train.len() {
            panic!("Number of samples in training data does not match the number of samples in target values");
        }
        for _ in 0..n_iter{
            let mut y_pred = vec![0.0; n_samples];
            for i in 0..n_samples{
                for j in 0..n_features{
                    y_pred[i] += self.weights[j] * x_train[i][j];
                }
                y_pred[i] += self.intercept;
                y_pred[i] = LogisticRegression::sigmoid(y_pred[i]);
            }

            let mut dw = vec![0.0; n_features];
            let mut di = 0.0;

            for i in 0..n_samples{
                for j in 0..n_features{
                    dw[j] += (y_pred[i] - y_train[i]) * x_train[i][j];
                }
                di += y_pred[i] - y_train[i];
            }

            self.intercept -= lr * di / n_samples as f64;
            for i in 0..n_features{
                self.weights[i] -= lr * dw[i] / n_samples as f64;
            }


            
        }

    }

    /// Predict the target probabilities for the test data.
    /// # Parameters
    /// x_test: `Vec<Vec<f64>` - A reference to a vector of vectors containing the test data
    /// # Returns
    /// `Vec<Vec<f64>` - A vector of vectors containing the predicted probabilities
    /// # Examples
    /// ```
    /// use rusty_math::linear::LogisticRegression;
    /// let mut model = LogisticRegression::new();
    /// model.fit(&x_train, &y_train, 0.01, 1000);
    /// let x_test = vec![vec![4.0, 5.0], vec![5.0, 6.0]];
    /// let y_pred = model.predict_proba(&x_test);
    /// ```
    /// # Panics
    /// If the number of features in the test data does not match the number of features in the training data
    /// # Remarks
    /// This function returns the probabilities of the positive class. The probability of the negative class is 1 - probability of the positive class.  
    /// If you want to predict the target values, you can threshold the probabilities at 0.5.
    pub fn predict_proba(&self, x_test: &Vec<Vec<f64>>) -> Vec<Vec<f64>>{
        let n_samples = x_test.len();
        let n_features = x_test[0].len();
        if n_features != self.weights.len() {
            panic!("Number of features in test data does not match the number of features in training data");
        }
        let mut y_pred = vec![vec![0.0; 2];n_samples];

        for i in 0..n_samples {
            for j in 0..n_features {
                y_pred[i][1] += self.weights[j] * x_test[i][j];
            }
            y_pred[i][1] += self.intercept;
            y_pred[i][1] = LogisticRegression::sigmoid(y_pred[i][1]);
            y_pred[i][0] = 1.0 - y_pred[i][1];
        }
        y_pred
    }


    /// Predict the target values for the test data.
    /// # Parameters
    /// x_test: `Vec<Vec<f64>` - A reference to a vector of vectors containing the test data
    /// # Returns
    /// `Vec<i32>` - A vector containing the predicted target values
    /// # Examples
    /// ```
    /// use rusty_math::linear::LogisticRegression;
    /// let mut model = LogisticRegression::new();
    /// model.fit(&x_train, &y_train, 0.01, 1000);
    /// let x_test = vec![vec![4.0, 5.0], vec![5.0, 6.0]];
    /// let y_pred = model.predict(&x_test);
    /// ```
    /// # Panics
    /// If the number of features in the test data does not match the number of features in the training data
    /// # Remarks
    /// This function uses the predict_proba function to predict the target values. If the probability of the positive class is greater than 0.5, the target value is 1, otherwise it is 0.
    /// This function is equivalent to calling predict_proba and then thresholding the probabilities at 0.5.
    pub fn predict(&self, x_test: &Vec<Vec<f64>>) -> Vec<i32>{
        let probs = self.predict_proba(x_test);
        let mut y_pred = vec![0; probs.len()];
        for i in 0..probs.len(){
            if probs[i][1] > probs[i][0]{
                y_pred[i] = 1;
            }
        }
        y_pred
    }

    /// Get the weights of the model
    /// # Returns
    /// `Vec<f64>` - A vector containing the weights of the model
    pub fn get_weights(&self) -> Vec<f64>{
        self.weights.clone()
    }


    /// Get the intercept of the model
    /// # Returns
    /// `f64` - The intercept of the model
    pub fn get_intercept(&self) -> f64{
        self.intercept
    }

    /// Get the accuracy of the model on the test data
    /// # Parameters
    /// x_test: `&Vec<Vec<f64>>` - A reference to a vector of vectors containing the test data  
    /// y_test: `&Vec<f64>` - A reference to a vector containing the target values  
    /// # Returns
    /// `HashMap<String,f64>` - A hashmap containing the accuracy of the model. See the accuracy function in metrics for more details
    /// # Examples
    /// ```
    /// use rusty_math::linear::LogisticRegression;
    /// let mut model = LogisticRegression::new();
    /// model.fit(&x_train, &y_train, 0.01, 1000);
    /// let score = model.score(&x_test, &y_test);
    /// ```
    pub fn score(&self, x_test: &Vec<Vec<f64>>, y_test: &Vec<f64>) -> HashMap<String,f64>{
        let y_pred = self.predict(x_test);
        let y_test = y_test.iter().map(|&x| x as i32).collect();
        let y_pred = y_pred.iter().map(|&x| x as i32).collect();
        accuracy(&y_pred, &y_test)
    }

}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_regression() {
        let mut model = LinearRegression::new();
        let x_train = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
        let y_train = vec![3.0, 4.0, 5.0];
        model.fit(&x_train, &y_train, 0.01, 1000);
        let x_test = vec![vec![4.0, 5.0], vec![5.0, 6.0]];
        let y_pred = model.predict(&x_test);
        let score = model.score(&x_test, &vec![6.0, 7.0]);
        let coefficients = model.get_weights();
        let intercept = model.get_intercept();
    }

    #[test]
    fn test_polynomial_regression() {
        let mut model = PolynomialRegression::new(2);
        let x_train = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y_train = vec![1.0, 4.0, 9.0];
        model.fit(&x_train, &y_train, 0.1, 1000);
        let x_test = vec![vec![4.0], vec![5.0]];
        let y_pred = model.predict(&x_test);
        let score = model.score(&x_test, &vec![16.0, 25.0]);
        let coefficients = model.get_weights();
        let intercept = model.get_intercept();
    }

    #[test]
    fn test_ridge_regression() {
        let mut model = RidgeRegression::new(0.01);
        let x_train = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
        let y_train = vec![3.0, 4.0, 5.0];
        model.fit(&x_train, &y_train, 0.01, 1000);
        let x_test = vec![vec![4.0, 5.0], vec![5.0, 6.0]];
        let y_pred = model.predict(&x_test);
        let score = model.score(&x_test, &vec![6.0, 7.0]);
        let coefficients = model.get_weights();
        let intercept = model.get_intercept();
    }

    #[test]
    fn test_lasso_regression() {
        let mut model = LassoRegression::new(0.01);
        let x_train = vec![vec![1.0, 2.0], vec![1.0, 3.0], vec![1.0, 4.0]];
        let y_train = vec![3.0, 4.0, 5.0];
        model.fit(&x_train, y_train, 0.1, 1000);
        let x_test = vec![vec![1.0, 5.0], vec![1.0, 6.0]];
        let y_pred = model.predict(&x_test);
        let score = model.score(&x_test, &vec![6.0, 7.0]);
        let coefficients = model.weights();
        let intercept = model.intercept();
    }

    #[test]
    fn test_logistic_regression(){
        let x_train = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
            vec![4.0, 5.0],
            vec![5.0, 6.0],
            vec![6.0, 7.0],
        ];
        let y_train = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let x_test = vec![vec![-1.0, 0.0], vec![4.0, 5.0], vec![7.0, 7.0], vec![1.0, 9.0]];
        let mut model = LogisticRegression::new();
        model.fit(&x_train, &y_train, 0.01, 1000);
        let preds = model.predict(&x_test);
        let score = model.score(&x_test, &vec![0.0, 1.0,1.0,0.0]);
        let weights = model.get_weights();
        let intercept = model.get_intercept();
    }
}
