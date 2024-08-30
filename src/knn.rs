//! # K-Nearest Neighbors
//! This module contains structs and implementations for K-Nearest Neighbors algorithms.
//! The K-Nearest Neighbors (KNN) algorithm is a supervised machine learning method employed to tackle classification and regression problems. It is widely disposable in real-life scenarios since it is non-parametric, meaning it does not make any underlying assumptions about the distribution of data.
//! The K-NN algorithm works by finding the K nearest neighbors to a given data point based on a distance metric. The class or value of the data point is then determined by the majority vote or average of the K neighbors. This approach allows the algorithm to adapt to different patterns and make predictions based on the local structure of the data.
//! The K-NN algorithm is versatile and can be used for both classification and regression tasks. It is also easy to implement and understand, making it a popular choice for many machine learning applications.


use std::collections::{HashMap, HashSet};
use crate::metrics::{accuracy,r2_score};


fn euclidean_distance(x1: &Vec<f64>, x2: &Vec<f64>) -> f64 {
    let mut sum = 0.0;
    for i in 0..x1.len() {
        sum += (x1[i] - x2[i]).powi(2);
    }
    sum.sqrt()
}

fn minkowski_distance(x1: &Vec<f64>, x2: &Vec<f64>, p: i32) -> f64 {
    let mut sum = 0.0;
    for i in 0..x1.len() {
        sum += (x1[i] - x2[i]).abs().powi(p as i32);
    }
    sum.powf(1.0 / p as f64)
}

fn cosine(x1: &Vec<f64>, x2: &Vec<f64>) -> f64 {
    let mut dot = 0.0;
    let mut mag1 = 0.0;
    let mut mag2 = 0.0;
    for i in 0..x1.len() {
        dot += x1[i] * x2[i];
        mag1 += x1[i].powi(2);
        mag2 += x2[i].powi(2);
    }
    dot / (mag1.sqrt() * mag2.sqrt())
}

fn manhattan(x1: &Vec<f64>, x2: &Vec<f64>) -> f64 {
    let mut sum = 0.0;
    for i in 0..x1.len() {
        sum += (x1[i] - x2[i]).abs();
    }
    sum
}

fn get_dist(x1: &Vec<f64>, x2: &Vec<f64>, metric:String, p: Option<i32>) -> f64 {
    match metric.as_str() {
        "euclidean" => euclidean_distance(x1, x2),
        "minkowski" => minkowski_distance(x1, x2, p.unwrap()),
        "cosine" => cosine(x1, x2),
        "manhattan" => manhattan(x1, x2),
        _ => panic!("metric must be one of 'euclidean', 'minkowski', 'cosine', 'manhattan"),
    }
}



/// # K-Nearest Neighbors Classifier
/// The `KNNeighborsClassifier` struct implements the K-Nearest Neighbors algorithm for classification tasks.  
/// Fields:
/// - `n_neighbors`: The number of neighbors to consider when making predictions.
/// - `metric`: The distance metric used to calculate the distance between data points. Supported metrics include 'euclidean', 'minkowski', 'cosine', and 'manhattan'.
/// - `p`: The power parameter for the Minkowski distance metric. This parameter is required when using the 'minkowski' metric.
pub struct KNNeighborsClassifier{
    pub n_neighbors: usize,
    pub metric: String,
    pub p: Option<i32>,
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
}

impl KNNeighborsClassifier {
    
    /// Create a new K-Nearest Neighbors classifier with the specified number of neighbors and distance metric.
    /// # Parameters
    /// - `n_neighbors`: `usize` -  The number of neighbors to consider when making predictions.
    /// - `metric`: `String` -  The distance metric used to calculate the distance between data points. Supported metrics include 'euclidean', 'minkowski', 'cosine', and 'manhattan'.
    /// - `p`: `Option<i32>` -  The power parameter for the Minkowski distance metric. This parameter is required when using the 'minkowski' metric. For other metric times pass None.
    /// # Returns
    /// - `KNNeighborsClassifier`: A new instance of the K-Nearest Neighbors classifier.
    /// # Example
    /// ```
    /// use rusty_machine::learning::knn::KNNeighborsClassifier;
    /// let knn = KNNeighborsClassifier::new(3, "euclidean".to_string(), None);
    /// ```
    /// # Panics
    /// - If the metric is not one of 'euclidean', 'minkowski', 'cosine', or 'manhattan'.
    /// - If the metric is 'minkowski' and the power parameter `p` is None.
    pub fn new(n_neighbors: usize, metric: String, p: Option<i32>) -> KNNeighborsClassifier {
        
        if metric != "euclidean" && metric != "minkowski" && metric != "cosine" && metric != "manhattan" {
            panic!("metric must be one of 'euclidean', 'minkowski', 'cosine', 'manhattan'");
        }

        if metric == "minkowski" && p.is_none() {
            panic!("p must be specified for minkowski metric");
        }

        KNNeighborsClassifier {
            n_neighbors,
            metric,
            p,
            x_train: Vec::new(),
            y_train: Vec::new(),
        }
    }


    /// Fit the K-Nearest Neighbors classifier to the training data.
    /// # Parameters
    /// - `x_train`: `&Vec<Vec<f64>>` - The input features for the training data of shape (n_samples, n_features).
    /// - `y_train`: `&Vec<f64>` - The target labels for the training data of shape (n_samples).
    /// # Example
    /// ```
    /// use rusty_machine::learning::knn::KNNeighborsClassifier;
    /// let mut knn = KNNeighborsClassifier::new(3, "euclidean".to_string(), None);
    /// let x_train = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    /// let y_train = vec![0.0, 1.0, 0.0];
    /// knn.fit(&x_train, &y_train);
    /// ```
    /// # Panics   
    /// - If the length of `x_train` and `y_train` are not equal.
    pub fn fit(&mut self, x_train: &Vec<Vec<f64>>, y_train: &Vec<f64>) {
        if x_train.len() != y_train.len() {
            panic!("x_train and y_train must have the same length");
        }
        self.x_train = x_train.clone();
        self.y_train = y_train.clone();
    }

    


    /// Get the indices in training data of the K-nearest neighbors to the input data point.
    /// # Parameters
    /// - `x`: `&Vec<f64>` - The input data point for which to find the nearest neighbors of shape (n_features).
    /// # Returns
    /// - `Vec<i32>`: The target labels of the K-nearest neighbors.
    /// # Example
    /// ```
    /// use rusty_machine::learning::knn::KNNeighborsClassifier;
    /// let knn = KNNeighborsClassifier::new(3, "euclidean".to_string(), None);
    /// knn.fit(&x_train, &y_train);
    /// let x = vec![1.0, 2.0];
    /// let y = knn.knearest(&x);
    /// ```
    /// # Panics
    /// - If the length of `x` is not equal to the number of features in the training data.
    /// 
    pub fn knearest_neighbors(&self, x: &Vec<f64>) -> Vec<i32> {
        if x.len() != self.x_train[0].len() {
            panic!("x must have the same number of features as the training data");
        }
        let mut distances = Vec::new();
        for i in 0..self.x_train.len() {
            let distance = crate::knn::get_dist(x, &self.x_train[i], self.metric.clone(), self.p.clone());
            distances.push((distance, i as i32));
        }
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut y = Vec::new();
        for i in 0..self.n_neighbors {
            y.push(distances[i].1);
        }
        y
    }
    
    
    fn knearest(&self, x: &Vec<f64>) -> Vec<f64> {
        if x.len() != self.x_train[0].len() {
            panic!("x must have the same number of features as the training data");
        }
        let mut distances = Vec::new();
        for i in 0..self.x_train.len() {
            let distance = crate::knn::get_dist(x, &self.x_train[i], self.metric.clone(), self.p.clone());
            distances.push((distance, self.y_train[i]));
        }
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut y = Vec::new();
        for i in 0..self.n_neighbors {
            y.push(distances[i].1);
        }
        y
    }


    /// Make predictions on the test data using the K-Nearest Neighbors classifier. Get the class label of the majority of the K-nearest neighbors.
    /// # Parameters
    /// - `x_test`: `&Vec<Vec<f64>>` - The input features for the test data of shape (n_samples, n_features).
    /// # Returns
    /// - `Vec<f64>`: The predicted class labels for the test data of shape (n_samples).
    /// # Example
    /// ```
    /// use rusty_machine::learning::knn::KNNeighborsClassifier;
    /// let knn = KNNeighborsClassifier::new(3, "euclidean".to_string(), None);
    /// knn.fit(&x_train, &y_train);
    /// let y_pred = knn.predict(&x_test);
    /// ```
    /// # Panics
    /// - If the number of features in `x_test` is not equal to the number of features in the training data.
    pub fn predict(&self, x_test: &Vec<Vec<f64>>) -> Vec<f64> {
        let mut y_pred = Vec::new();
        for i in 0..x_test.len() {
            let y = self.knearest(&x_test[i]);
            let mut counts = HashMap::new();
            for j in 0..y.len() {
                *counts.entry(y[j] as i32).or_insert(0)+=1;
            }
            let mut max = 0;
            let mut max_key = 0;
            for (key, value) in counts.iter() {
                if *value > max {
                    max = *value;
                    max_key = *key;
                }
                else if *value == max {
                    max_key=y[0] as i32;
                }
            }
            y_pred.push(max_key as f64);
        }
        y_pred
    }

    /// Make predictions on the test data using the K-Nearest Neighbors classifier. Get the class probabilities of each class based on K-nearest neighbors.
    /// # Parameters
    /// - `x_test`: `&Vec<Vec<f64>>` - The input features for the test data of shape (n_samples, n_features).
    /// # Returns
    /// - `Vec<Vec<f64>>`: The predicted class probabilities for the test data of shape (n_samples, n_classes).
    /// # Example
    /// ```
    /// use rusty_machine::learning::knn::KNNeighborsClassifier;
    /// let knn = KNNeighborsClassifier::new(3, "euclidean".to_string(), None);
    /// knn.fit(&x_train, &y_train);
    /// let y_pred = knn.predict_proba(&x_test);
    /// ```
    /// # Panics
    /// - If the number of features in `x_test` is not equal to the number of features in the training data.
    pub fn predict_proba(&self, x_test: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut y_pred = Vec::new();
        let classes = self.y_train.iter().cloned().map(|x| x as i64).collect::<HashSet<_>>();
        let mut counts = HashMap::new();
        for class in classes.iter() {
            counts.insert(*class, 0);
        }
        for i in 0..x_test.len() {
            let y = self.knearest(&x_test[i]);
            counts.values_mut().for_each(|v| *v = 0);
            for j in 0..y.len() {
                *counts.entry(y[j] as i64).or_insert(0)+=1;
            }
            let mut probs = Vec::new();
            for class in classes.iter() {
                probs.push(*counts.get(class).unwrap() as f64 / self.n_neighbors as f64);
            }
            y_pred.push(probs);
        }
        y_pred
            
            
    }

    /// Evaluate the performance of the K-Nearest Neighbors classifier on the test data using the accuracy metric.
    /// # Parameters
    /// - `x_test`: `&Vec<Vec<f64>>` - The input features for the test data of shape (n_samples, n_features).
    /// - `y_test`: `&Vec<f64>` - The target labels for the test data of shape (n_samples).
    /// # Returns
    /// - `HashMap<String, f64>`: A HashMap containing the accuracy score of the classifier. See the `accuracy` function in metrics module for more details.
    /// # Example
    /// ```
    /// use rusty_machine::learning::knn::KNNeighborsClassifier;
    /// let knn = KNNeighborsClassifier::new(3, "euclidean".to_string(), None);
    /// knn.fit(&x_train, &y_train);
    /// let scores = knn.score(&x_test, &y_test);
    /// ```
    /// # Panics
    /// - If the number of features in `x_test` is not equal to the number of features in the training data.
    pub fn score(&self, x_test: &Vec<Vec<f64>>, y_test: &Vec<f64>) -> HashMap<String, f64> {
        let y_pred = self.predict(x_test).iter().map(|x| *x as i32).collect::<Vec<i32>>();
        let y_test_=&y_test.iter().map(|x| *x as i32).collect::<Vec<i32>>();
        accuracy(&y_pred, &y_test_)
    }

    /// Get the parameters of the K-Nearest Neighbors classifier.
    /// # Returns
    /// - `HashMap<String, String>`: A HashMap containing the parameters of the classifier.
    /// # Example
    /// ```
    /// use rusty_machine::learning::knn::KNNeighborsClassifier;
    /// let knn = KNNeighborsClassifier::new(3, "euclidean".to_string(), None);
    /// let params = knn.get_params();
    /// ```
    pub fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("n_neighbors".to_string(), self.n_neighbors.to_string());
        params.insert("metric".to_string(), self.metric.clone());
        if self.p.is_some() {
            params.insert("p".to_string(), self.p.unwrap().to_string());
        }
        params
    }
    


}


/// # K-Nearest Neighbors Regressor
/// The `KNNeighborsRegressor` struct implements the K-Nearest Neighbors algorithm for regression tasks.
/// Fields:
/// - `n_neighbors`: The number of neighbors to consider when making predictions.
/// - `metric`: The distance metric used to calculate the distance between data points. Supported metrics include 'euclidean', 'minkowski
/// - `p`: The power parameter for the Minkowski distance metric. This parameter is required when using the 'minkowski' metric.
pub struct KNNeighborsRegressor {
    pub n_neighbors: usize,
    pub metric: String,
    pub p: Option<i32>,
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
}


impl KNNeighborsRegressor {
    
    /// Create a new K-Nearest Neighbors regressor with the specified number of neighbors and distance metric.
    /// # Parameters
    /// - `n_neighbors`: `usize` -  The number of neighbors to consider when making predictions.
    /// - `metric`: `String` -  The distance metric used to calculate the distance between data points. Supported metrics include 'euclidean', 'minkowski
    /// - `p`: `Option<i32>` -  The power parameter for the Minkowski distance metric. This parameter is required when using the 'minkowski' metric. For other metric times pass None.
    /// # Returns
    /// - `KNNeighborsRegressor`: A new instance of the K-Nearest Neighbors regressor.
    /// # Example
    /// ```
    /// use rusty_machine::learning::knn::KNNeighborsRegressor;
    /// let knn = KNNeighborsRegressor::new(3, "euclidean".to_string(), None);
    /// ```
    /// # Panics
    /// - If the metric is not one of 'euclidean', 'minkowski', 'cosine', or 'manhattan'.
    /// - If the metric is 'minkowski' and the power parameter `p` is None.
    pub fn new(n_neighbors: usize, metric: String, p: Option<i32>) -> KNNeighborsRegressor {
        
        if metric != "euclidean" && metric != "minkowski" && metric != "cosine" && metric != "manhattan" {
            panic!("metric must be one of 'euclidean', 'minkowski', 'cosine', 'manhattan'");
        }

        if metric == "minkowski" && p.is_none() {
            panic!("p must be specified for minkowski metric");
        }

        KNNeighborsRegressor {
            n_neighbors,
            metric,
            p,
            x_train: Vec::new(),
            y_train: Vec::new(),
        }
    }

    /// Fit the K-Nearest Neighbors regressor to the training data.
    /// # Parameters
    /// - `x_train`: `&Vec<Vec<f64>>` - The input features for the training data of shape (n_samples, n_features).
    /// - `y_train`: `&Vec<f64>` - The target labels for the training data of shape (n_samples).
    /// # Example
    /// ```
    /// use rusty_machine::learning::knn::KNNeighborsRegressor;
    /// let mut knn = KNNeighborsRegressor::new(3, "euclidean".to_string(), None);
    /// let x_train = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    /// let y_train = vec![0.0, 1.0, 0.0];
    /// knn.fit(&x_train, &y_train);
    /// ```
    /// # Panics
    /// - If the length of `x_train` and `y_train` are not equal.
    pub fn fit(&mut self, x_train: &Vec<Vec<f64>>, y_train: &Vec<f64>) {
        if x_train.len() != y_train.len() {
            panic!("x_train and y_train must have the same length");
        }
        self.x_train = x_train.clone();
        self.y_train = y_train.clone();
    }

    /// Make predictions on the test data using the K-Nearest Neighbors regressor. Get the average of the target labels of the K-nearest neighbors.
    /// # Parameters
    /// - `x_test`: `&Vec<Vec<f64>>` - The input features for the test data of shape (n_samples, n_features).
    /// # Returns
    /// - `Vec<f64>`: The predicted target labels for the test data of shape (n_samples).
    /// # Example
    /// ```
    /// use rusty_machine::learning::knn::KNNeighborsRegressor;
    /// let knn = KNNeighborsRegressor::new(3, "euclidean".to_string(), None);
    /// knn.fit(&x_train, &y_train);
    /// let y_pred = knn.predict(&x_test);
    /// ```
    /// # Panics
    /// - If the number of features in `x_test` is not equal to the number of features in the training data.
    pub fn predict(&self, x_test: &Vec<Vec<f64>>) -> Vec<f64> {
        if x_test[0].len() != self.x_train[0].len() {
            panic!("x_test must have the same number of features as the training data");
        }
        let mut y_pred = Vec::new();
        for i in 0..x_test.len() {
            let mut distances = Vec::new();
            for j in 0..self.x_train.len() {
                let distance = crate::knn::get_dist(&x_test[i], &self.x_train[j], self.metric.clone(), self.p.clone());
                distances.push((distance, self.y_train[j]));
            }
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let mut sum = 0.0;
            for j in 0..self.n_neighbors {
                sum += distances[j].1;
            }
            y_pred.push(sum / self.n_neighbors as f64);
        }
        y_pred
    }

    /// Get the indices in training data of the K-nearest neighbors to the input data point.
    /// # Parameters
    /// - `x`: `&Vec<f64>` - The input data point for which to find the nearest neighbors of shape (n_features).
    /// # Returns
    /// - `Vec<i32>`: The target labels of the K-nearest neighbors.
    /// # Example
    /// ```
    /// use rusty_machine::learning::knn::KNNeighborsRegressor;
    /// let knn = KNNeighborsRegressor::new(3, "euclidean".to_string(), None);
    /// knn.fit(&x_train, &y_train);
    /// let x = vec![1.0, 2.0];
    /// let y = knn.knearest(&x);
    /// ```
    /// # Panics
    /// - If the length of `x` is not equal to the number of features in the training data.
    pub fn knearest_neighbors(&self, x: &Vec<f64>) -> Vec<i32> {
        if x.len() != self.x_train[0].len() {
            panic!("x must have the same number of features as the training data");
        }
        let mut distances = Vec::new();
        for i in 0..self.x_train.len() {
            let distance = crate::knn::get_dist(x, &self.x_train[i], self.metric.clone(), self.p.clone());
            distances.push((distance, i as i32));
        }
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut y = Vec::new();
        for i in 0..self.n_neighbors {
            y.push(distances[i].1);
        }
        y
    }

    /// Evaluate the performance of the K-Nearest Neighbors regressor on the test data using the R2 score metric.
    /// # Parameters
    /// - `x_test`: `&Vec<Vec<f64>>` - The input features for the test data of shape (n_samples, n_features).
    /// - `y_test`: `&Vec<f64>` - The target labels for the test data of shape (n_samples).
    /// # Returns
    /// - `f64`: The R2 score of the regressor. See the `r2_score` function in metrics module for more details.
    /// # Example
    /// ```
    /// use rusty_machine::learning::knn::KNNeighborsRegressor;
    /// let knn = KNNeighborsRegressor::new(3, "euclidean".to_string(), None);
    /// knn.fit(&x_train, &y_train);
    /// let scores = knn.score(&x_test, &y_test);
    /// ```
    /// # Panics
    /// - If the number of features in `x_test` is not equal to the number of features in the training data.
    /// - If the length of `x_test` and `y_test` are not equal.
    pub fn score(&self, x_test: &Vec<Vec<f64>>, y_test: &Vec<f64>) -> f64 {
        if x_test[0].len() != self.x_train[0].len() {
            panic!("x_test must have the same number of features as the training data");
        }
        if x_test.len() != y_test.len() {
            panic!("x_test and y_test must have the same length");
        }
        let y_pred = self.predict(x_test);
        r2_score(&y_pred, y_test)
    }


    /// Get the parameters of the K-Nearest Neighbors Regressor.
    /// # Returns
    /// - `HashMap<String, String>`: A HashMap containing the parameters of the regressor.
    /// # Example
    /// ```
    /// use rusty_machine::learning::knn::KNNeighborsRegressor;
    /// let knn = KNNeighborsClassifier::new(3, "euclidean".to_string(), None);
    /// let params = knn.get_params();
    /// ```
    pub fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("n_neighbors".to_string(), self.n_neighbors.to_string());
        params.insert("metric".to_string(), self.metric.clone());
        if self.p.is_some() {
            params.insert("p".to_string(), self.p.unwrap().to_string());
        }
        params
    }

}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knn_new() {
        let knn = KNNeighborsClassifier::new(3, "euclidean".to_string(), None);
        assert_eq!(knn.n_neighbors, 3);
        assert_eq!(knn.metric, "euclidean");
        assert_eq!(knn.p, None);
    }

    #[test]
    fn test_knn_fit() {
        let mut knn = KNNeighborsClassifier::new(3, "euclidean".to_string(), None);
        let x_train = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let y_train = vec![0.0, 1.0, 0.0];
        knn.fit(&x_train, &y_train);
        assert_eq!(knn.x_train, x_train);
        assert_eq!(knn.y_train, y_train);
    }

    #[test]
    fn test_knn_knearest() {
        let mut knn = KNNeighborsClassifier::new(2, "euclidean".to_string(), None);
        let x_train = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let y_train = vec![0.0, 1.0, 0.0];
        knn.fit(&x_train, &y_train);
        let x = vec![1.0, 2.0];
        let y = knn.knearest(&x);
        assert_eq!(y, vec![0.0, 1.0]);
    }

    #[test]
    fn test_knn_predict() {
        let mut knn = KNNeighborsClassifier::new(2, "euclidean".to_string(), None);
        let x_train = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let y_train = vec![0.0, 1.0, 0.0];
        knn.fit(&x_train, &y_train);
        let x_test = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let y_pred = knn.predict(&x_test);
        assert_eq!(y_pred, vec![0.0, 1.0]);
    }

    #[test]
    fn test_knn_predict_proba() {
        let mut knn = KNNeighborsClassifier::new(2, "euclidean".to_string(), None);
        let x_train = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let y_train = vec![0.0, 1.0, 0.0];
        knn.fit(&x_train, &y_train);
        let x_test = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let y_pred = knn.predict_proba(&x_test);
        assert_eq!(y_pred, vec![vec![0.5, 0.5], vec![0.5, 0.5]]);
    }

    #[test]
    fn test_knn_regressor(){
        let mut knn = KNNeighborsRegressor::new(2, "euclidean".to_string(), None);
        let x_train = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let y_train = vec![0.0, 9.0, 2.0];
        knn.fit(&x_train, &y_train);
        let x_test = vec![vec![0.0, 2.0], vec![8.0, 10.0]];
        let y_pred = knn.predict(&x_test);
        assert_eq!(y_pred, vec![4.5, 9.5]);
    }
}


