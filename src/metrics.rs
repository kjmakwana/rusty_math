//! # Evaluation Metrics
//! Module for evaluating the performance of machine learning models
//! ## Functions
//! - mean_squared_error
//! - r2_score
//! - accuracy
//! - confusion_matrix
//! - precision
//! - recall

use std::collections::{HashMap, HashSet};

/// Calculate the mean squared error (MSE) of the model
/// The mean squared error is defined as the average of the squared differences between the predicted and true target values
/// # Parameters
/// y_pred: &Vec<f64> - A reference to a vector containing the predicted target values  
/// y_true: &Vec<f64> - A reference to a vector containing the true target values  
/// # Returns
/// f64 - The mean squared error
/// # Examples
/// ```
/// use rusty_math::metrics::mse;
/// let y_pred = vec![6.9, 7.2];
/// let y_true = vec![6.0, 7.0];
/// let mse = mse(&y_pred, &y_true);
/// ```
/// # Panics
/// If the number of samples in the predicted values does not match the number of samples in the true values
pub fn mse(y_pred: &Vec<f64>, y_true: &Vec<f64>) -> f64 {
    if y_pred.len() != y_true.len() {
        panic!("Number of samples in predicted values does not match the number of samples in true values");
    }
    let n_samples = y_pred.len();
    let mut mse = 0.0;
    for i in 0..n_samples {
        mse += (y_pred[i] - y_true[i]).powi(2);
    }

    mse / n_samples as f64
}

/// Calculate the R<sup>2</sup> score as the goodness of fit of the model
/// The R<sup>2</sup> score is defined as 1 - RSS/TSS, where RSS is the residual sum of squares and TSS is the total sum of squares
/// # Parameters
/// y_pred: &Vec<f64> - A reference to a vector containing the predicted target values  
/// y_true: &Vec<f64> - A reference to a vector containing the true target values  
/// # Returns
/// f64 - The R<sup>2</sup> score
/// # Examples
/// ```
/// use rusty_math::metrics::mse;
/// let y_pred = vec![6.9, 7.2];
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

/// Calculate the accuracy of a classification model. The accuracy is defined as the number of correct predictions divided by the total number of predictions.    
/// Function calculates the overall accuracy and the accuracy for each class.
/// # Parameters
/// y_pred: &Vec<u8> - A reference to a vector containing the predicted target values  
/// y_true: &Vec<u8> - A reference to a vector containing the true target values  
/// # Returns
/// HashMap<String,f64> - A hashmap containing the overall accuracy and the accuracy for each class
/// # Examples
/// ```
/// use rusty_math::metrics::accuracy;
/// let y_pred = vec![0, 1, 1, 0, 1, 0, 1, 1, 1, 0];
/// let y_true = vec![0, 1, 1, 0, 1, 0, 1, 0, 1, 0];
/// let scores = accuracy(&y_pred, &y_true);
/// ```
/// # Panics
/// If the number of samples in the predicted values does not match the number of samples in the true values
pub fn accuracy(y_pred: &Vec<u8>, y_true: &Vec<u8>) -> HashMap<String, f64> {
    if y_pred.len() != y_true.len() {
        panic!("Number of samples in predicted values does not match the number of samples in true values");
    }
    let n_samples = y_pred.len();
    let mut correct = 0;
    let nunique = y_true.iter().collect::<HashSet<_>>().len();
    let mut class_correct = vec![0; nunique];
    let mut class_total = vec![0; nunique];
    for i in 0..n_samples {
        if y_pred[i] == y_true[i] {
            correct += 1;
            class_correct[y_true[i] as usize] += 1;
        }
        class_total[y_true[i] as usize] += 1;
    }

    let mut scores = HashMap::new();
    scores.insert("Overall".to_string(), correct as f64 / n_samples as f64);
    for i in 0..nunique {
        scores.insert(
            i.to_string(),
            class_correct[i] as f64 / class_total[i] as f64,
        );
    }
    scores
}

/// Calculate the confusion matrix of a classification model. The confusion matrix is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known.
/// # Parameters
/// y_pred: &Vec<u8> - A reference to a vector containing the predicted target values  
/// y_true: &Vec<u8> - A reference to a vector containing the true target values  
/// # Returns
/// Vec<Vec<u32>> - A 2D vector containing the confusion matrix
/// # Examples
/// ```
/// use rusty_math::metrics::confusion_matrix;
/// let y_pred = vec![0, 1, 1, 0, 1, 0, 1, 1, 1, 0];
/// let y_true = vec![0, 1, 1, 0, 1, 0, 1, 0, 1, 0];
/// let matrix = confusion_matrix(&y_pred, &y_true);
/// ```
/// # Panics
/// If the number of samples in the predicted values does not match the number of samples in the true values
pub fn confusion_matrix(y_pred: &Vec<u8>, y_true: &Vec<u8>) -> Vec<Vec<u32>> {
    if y_pred.len() != y_true.len() {
        panic!("Number of samples in predicted values does not match the number of samples in true values");
    }
    let n_samples = y_pred.len();
    let nunique = y_true.iter().collect::<HashSet<_>>().len();
    let mut matrix = vec![vec![0; nunique]; nunique];
    for i in 0..n_samples {
        matrix[y_true[i] as usize][y_pred[i] as usize] += 1;
    }
    matrix
}

/// Calculate the precision of a classification model. The precision is defined as the number of true positive predictions divided by the total number of positive predictions.
/// Function calculates the overall precision and the precision for each class (in case of multiclass classification). The function supports two methods for calculating precision: "macro" and "micro".
/// # Parameters
/// y_pred: &Vec<u8> - A reference to a vector containing the predicted target values  
/// y_true: &Vec<u8> - A reference to a vector containing the true target values  
/// method: &str - The method to use for calculating precision. Options are "macro" and "micro"  
/// # Returns
/// HashMap<String,f64> - A hashmap containing the overall precision and the precision for each class  
/// # Examples
/// ```
/// use rusty_math::metrics::precision;
/// let y_pred = vec![0, 1, 1, 0, 1, 0, 1, 1, 1, 0];
/// let y_true = vec![0, 1, 1, 0, 1, 0, 1, 0, 1, 0];
/// let precision = precision(&y_pred, &y_true, "macro");
/// ```
/// # Panics
/// If the number of samples in the predicted values does not match the number of samples in the true values  
/// If the method is not "macro" or "micro"  
pub fn precision(y_pred: &Vec<u8>, y_true: &Vec<u8>, method: &str) -> HashMap<String, f64> {
    if y_pred.len() != y_true.len() {
        panic!("Number of samples in predicted values does not match the number of samples in true values");
    }
    let matrix = confusion_matrix(&y_pred, &y_true);
    let nunique = y_true.iter().collect::<HashSet<_>>().len();
    let mut p = 0.0;
    let mut precision = HashMap::new();
    if nunique == 1 {
        precision.insert(
            "overall".to_string(),
            matrix[0][0] as f64 / (matrix[0][0] + matrix[0][1]) as f64,
        );
    } else {
        match method {
            "macro" => {
                let mut p_sum = 0.0;
                for i in 0..nunique {
                    for j in 0..nunique {
                        p += matrix[i][j] as f64;
                    }
                    p_sum += matrix[i][i] as f64;
                    precision.insert(format!("precision_{}", i), matrix[i][i] as f64 / p as f64);
                }
                precision.insert("overall".to_string(), p_sum as f64 / nunique as f64);
            }
            "micro" => {
                let mut tp = 0;
                let mut fp = 0;
                for i in 0..nunique {
                    tp += matrix[i][i];
                    for j in 0..nunique {
                        fp += matrix[j][i];
                    }
                    precision.insert(format!("precision_{}", i), tp as f64 / (tp + fp) as f64);
                }
                precision.insert("overall".to_string(), tp as f64 / (tp + fp) as f64);
            }
            _ => panic!("Invalid method"),
        }
    }
    precision
}

/// Calculate the recall of a classification model. The recall is defined as the number of true positive predictions divided by the total number of actual positive values.
/// Function calculates the overall recall and the recall for each class (in case of multiclass classification). The function supports two methods for calculating recall: "macro" and "micro".
/// # Parameters
/// y_pred: &Vec<u8> - A reference to a vector containing the predicted target values  
/// y_true: &Vec<u8> - A reference to a vector containing the true target values  
/// method: &str - The method to use for calculating recall. Options are "macro" and "micro"  
/// # Returns
/// HashMap<String,f64> - A hashmap containing the overall recall and the recall for each class
/// # Examples
/// ```
/// use rusty_math::metrics::recall;
/// let y_pred = vec![0, 1, 1, 0, 1, 0, 1, 1, 1, 0];
/// let y_true = vec![0, 1, 1, 0, 1, 0, 1, 0, 1, 0];
/// let recall = recall(&y_pred, &y_true, "macro");
/// ```
/// # Panics
/// If the number of samples in the predicted values does not match the number of samples in the true values  
/// If the method is not "macro" or "micro"  
pub fn recall(y_pred: &Vec<u8>, y_true: &Vec<u8>, method: &str) -> HashMap<String, f64> {
    if y_pred.len() != y_true.len() {
        panic!("Number of samples in predicted values does not match the number of samples in true values");
    }
    let matrix = confusion_matrix(&y_pred, &y_true);
    let nunique = y_true.iter().collect::<HashSet<_>>().len();
    let mut r = 0.0;
    let mut recall = HashMap::new();
    if nunique == 1 {
        recall.insert(
            "overall".to_string(),
            matrix[0][0] as f64 / (matrix[0][0] + matrix[1][0]) as f64,
        );
    } else {
        match method {
            "macro" => {
                let mut r_sum = 0.0;
                for i in 0..nunique {
                    for j in 0..nunique {
                        r += matrix[j][i] as f64;
                    }
                    r_sum += matrix[i][i] as f64;
                    recall.insert(format!("recall_{}", i), matrix[i][i] as f64 / r as f64);
                }
                recall.insert("overall".to_string(), r_sum as f64 / nunique as f64);
            }
            "micro" => {
                let mut tp = 0;
                let mut f_n = 0;
                for i in 0..nunique {
                    tp += matrix[i][i];
                    for j in 0..nunique {
                        f_n += matrix[i][j];
                    }
                    recall.insert(format!("recall_{}", i), tp as f64 / (tp + f_n) as f64);
                }
                recall.insert("overall".to_string(), tp as f64 / (tp + f_n) as f64);
            }
            _ => panic!("Invalid method"),
        }
    }
    recall
}

/// Calculate the F1 score of a classification model. The F1 score is the harmonic mean of the precision and recall.
/// Function calculates the overall F1 score and the F1 score for each class (in case of multiclass classification). The function supports two methods for calculating F1 score: "macro" and "micro".
/// # Parameters
/// y_pred: &Vec<u8> - A reference to a vector containing the predicted target values  
/// y_true: &Vec<u8> - A reference to a vector containing the true target values  
/// method: &str - The method to use for calculating F1 score. Options are "macro" and "micro"  
/// # Returns
/// HashMap<String,f64> - A hashmap containing the overall F1 score and the F1 score for each class  
/// # Examples
/// ```
/// use rusty_math::metrics::f1;
/// let y_pred = vec![0, 1, 1, 0, 1, 0, 1, 1, 1, 0];
/// let y_true = vec![0, 1, 1, 0, 1, 0, 1, 0, 1, 0];
/// let f1 = f1(&y_pred, &y_true, "macro");
/// ```
/// # Panics
/// If the number of samples in the predicted values does not match the number of samples in the true values  
/// If the method is not "macro" or "micro"  
pub fn f1(y_pred: &Vec<u8>, y_true: &Vec<u8>, method: &str) -> HashMap<String, f64> {
    let precision = precision(&y_pred, &y_true, method);
    let recall = recall(&y_pred, &y_true, method);
    let nunique = y_true.iter().collect::<HashSet<_>>().len();
    let mut f1 = HashMap::new();
    f1.insert(
        "overall".to_string(),
        2.0 * (precision.get("overall").unwrap() * recall.get("overall").unwrap())
            / (precision.get("overall").unwrap() + recall.get("overall").unwrap()),
    );
    for i in 0..nunique {
        f1.insert(
            format!("f1_{}", i),
            2.0 * (precision.get(&format!("precision_{}", i)).unwrap()
                * recall.get(&format!("recall_{}", i)).unwrap())
                / (precision.get(&format!("precision_{}", i)).unwrap()
                    + recall.get(&format!("recall_{}", i)).unwrap()),
        );
    }
    f1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse() {
        let y_pred = vec![6.9, 7.2];
        let y_true = vec![6.0, 7.0];
        let result = mse(&y_pred, &y_true);
        assert_eq!(result, 0.425);
    }

    #[test]
    fn test_r2_score() {
        let y_pred = vec![6.1, 7.2];
        let y_true = vec![6.0, 7.0];
        let result = r2_score(&y_pred, &y_true);
        assert_eq!(result, 0.9);
    }

    #[test]
    fn test_accuracy() {
        let y_pred = vec![0, 1, 1, 0, 1, 0, 1, 1, 1, 0];
        let y_true = vec![0, 1, 1, 0, 1, 0, 1, 0, 1, 0];
        let result = accuracy(&y_pred, &y_true);
        assert_eq!(result.get("Overall").unwrap(), &0.9);
        assert_eq!(result.get("0").unwrap(), &0.8);
        assert_eq!(result.get("1").unwrap(), &1.0);
    }

    #[test]
    fn test_confusion_matrix() {
        let y_pred = vec![0, 1, 1, 0, 1, 0, 1, 1, 1, 0];
        let y_true = vec![0, 1, 1, 0, 1, 0, 1, 0, 1, 0];
        let result = confusion_matrix(&y_pred, &y_true);
        assert_eq!(result[0][0], 4);
        assert_eq!(result[0][1], 1);
        assert_eq!(result[1][0], 0);
        assert_eq!(result[1][1], 5);
    }
}
