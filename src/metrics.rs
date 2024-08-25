

/// Calculate the R<sup>2</sup> score as the goodness of fit of the model
/// The R<sup>2</sup> score is defined as 1 - RSS/TSS, where RSS is the residual sum of squares and TSS is the total sum of squares
/// # Parameters
/// y_pred: &Vec<f64> - A reference to a vector containing the predicted target values
/// y_true: &Vec<f64> - A reference to a vector containing the true target values
/// # Returns
/// f64 - The R<sup>2</sup> score
/// # Examples
/// ```
/// use rusty_math::linear::LinearRegression;
/// use rusty_math::metrics::r2_score;
/// let mut model = LinearRegression::new();
/// let x_train = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
/// let y_train = vec![3.0, 4.0, 5.0];
/// model.fit(&x_train, &y_train, 0.01, 1000);
/// let x_test = vec![vec![4.0, 5.0], vec![5.0, 6.0]];
/// let y_pred = model.predict(&x_test);
/// let y_true = vec![6.0, 7.0];
/// let score = r2_score(&y_pred, &y_true);
/// ```
/// # Panics
/// If the number of samples in the predicted values does not match the number of samples in the true values
pub fn r2_score(y_pred: &Vec<f64>, y_true: &Vec<f64>) -> f64 {
    if y_pred.len() != y_true.len() {
        panic!("Number of samples in predicted values does not match the number of samples in true values");
    }
    let n_samples = y_pred.len();
    let mut rss = 0.0;
    let mut tss = 0.0;
    let y_mean = y_true.iter().sum::<f64>() / n_samples as f64;
    for i in 0..n_samples {
        rss += (y_pred[i] - y_true[i]).powi(2);
        tss += (y_true[i] - y_mean).powi(2);

    }

    1.0 - rss / tss as f64
}