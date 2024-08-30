//! # Naive Bayes Classifier
//! The Naive Bayes classifier is a simple probabilistic classifier based on applying Bayes' theorem with strong (naive) independence assumptions.
//! It is based on a common principle: Naive Bayes classifiers assume that the value of a particular feature is independent of the value of any other feature, given the class variable.  
//! Even if these features depend on each other or upon the existence of the other features, all of these properties independently contribute to the probability that this particular class may be the correct class.  
//! The Naive Bayes classifier implemented here is a Gaussian Naive Bayes classifier, which assumes that the likelihood of the features is Gaussian.
//! The classifier is implemented in the NaiveBayesClassifier struct.
use std::collections::{HashMap, HashSet};
use crate::metrics::accuracy;

/// Naive Bayes Classifier
/// This struct implements a Gaussian Naive Bayes classifier. The model is trained using the fit method and predictions can be made using the predict method.
/// The model estimates the class priors, i.e., the probability of each class, the mean and variance of each feature for each class.
/// Fields -
/// 1. priors: 'HashMap<i32,f64>' - The class priors
/// 2. feature_means: 'HashMap<i32, Vec<f64>>' - The mean of each feature for each class
/// 3. feature_vars: 'HashMap<i32, Vec<f64>>' - The variance of each feature for each class
/// 4. class_counts: 'HashMap<i32, i32>' - The number of samples for each class
/// 5. n_features: 'i32' - The number of features
/// # Example
/// ```
/// use rusty_math::naive_bayes::NaiveBayesClassifier;
/// let mut model = NaiveBayesClassifier::new();
/// let n_features = model.n_features;
/// ```
pub struct NaiveBayesClassifier {
    pub priors: HashMap<i32, f64>,
    pub feature_means: HashMap<i32, Vec<f64>>,
    pub feature_vars: HashMap<i32, Vec<f64>>,
    pub class_counts: HashMap<i32, i32>,
    pub n_features: i32,
}

impl NaiveBayesClassifier {
    /// Create a new NaiveBayesClassifier.
    /// # Returns
    /// `NaiveBayesClassifier` - A new NaiveBayesClassifier object
    /// # Example
    /// ```
    /// use rusty_math::naive_bayes::NaiveBayesClassifier;
    /// let mut model = NaiveBayesClassifier::new();
    /// ```
    pub fn new() -> NaiveBayesClassifier {
        NaiveBayesClassifier {
            priors: HashMap::new(),
            feature_means: HashMap::new(),
            feature_vars: HashMap::new(),
            class_counts: HashMap::new(),
            n_features: 0,
        }
    }

    /// Fit the Naive Bayes classifier according to the given training data.
    /// # Parameters
    /// x_train: `Vec<Vec<f64>>` - The training input samples. A 2D vector of shape (n_samples, n_features)
    /// y_train: `Vec<i32>` - The target classes. A 1D vector of length n_samples
    /// # Example
    /// ```
    /// use rusty_math::naive_bayes::NaiveBayesClassifier;
    /// let mut model = NaiveBayesClassifier::new();
    /// let x_train = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
    /// let y_train = vec![0, 1, 0];
    /// model.fit(x_train, y_train);
    /// ```
    pub fn fit(&mut self, x_train: &Vec<Vec<f64>>, y_train: &Vec<i32>) {
        let n_samples = x_train.len();
        let n_features = x_train[0].len();
        let classes = y_train.iter().collect::<HashSet<_>>();
        self.n_features = n_features as i32;
        
        let mut class_counts = HashMap::new();
        for &class in classes {
            let mut count = 0;
            for &y in y_train.iter() {
                if y == class {
                    count += 1;
                }
            }
            class_counts.insert(class, count);
        }
        self.class_counts = class_counts.clone();
        for (class, count) in class_counts.iter() {
            self.priors.insert(*class, *count as f64 / n_samples as f64);
        }

        for (class, count) in class_counts.iter() {
            let mut feature_sum = vec![0.0; n_features];
            let mut feature_sq_sum = vec![0.0; n_features];
            for i in 0..n_samples {
                if y_train[i] == *class {
                    for j in 0..n_features {
                        feature_sum[j] += x_train[i][j];
                        feature_sq_sum[j] += x_train[i][j].powi(2);
                    }
                }
            }
            let class_mean = feature_sum
                .iter()
                .map(|x| x / *count as f64)
                .collect::<Vec<f64>>();
            let mut class_var = vec![0.0; n_features];
            for i in 0..n_features {
                class_var[i] = (feature_sq_sum[i] / *count as f64) - class_mean[i].powi(2);
            }
            self.feature_means.insert(*class, class_mean);
            self.feature_vars.insert(*class, class_var);
        }
    }

    /// Predict the class probabilities for the provided test data. Probability estimates are made using the Gaussian Naive Bayes formula.
    /// # Parameters
    /// x_test: `Vec<Vec<f64>>` - The test input samples. A 2D vector of shape (n_samples, n_features)
    /// # Returns
    /// `Vec<Vec<f64>>` - A 2D vector of shape (n_samples, n_classes) containing the class probabilities for each sample
    /// # Example
    /// ```
    /// use rusty_math::naive_bayes::NaiveBayesClassifier;
    /// let mut model = NaiveBayesClassifier::new();
    /// model.fit(&x_train, &y_train);
    /// let probs = model.predict_proba(&x_test);
    /// ```
    /// # Panics
    /// This method will panic if the number of features in x_test is not equal to the number of features in x_train
    pub fn predict_proba(&self, x_test: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let n_features = x_test[0].len();
        if n_features != self.feature_means[&0].len() {
            panic!("Number of features in x_test is not equal to number of features in x_train");
        }
        let mut probs = Vec::new();
        // let log_priors = self
        //     .priors
        //     .iter()
        //     .map(|(class, prior)| prior.ln())
        //     .collect::<Vec<f64>>();
        for x in x_test.iter() {
            let mut class_probs = vec![0.0; self.priors.len()];
            for (&class, &prior) in self.priors.iter() {
                let mut log_prob = prior.ln();
                for i in 0..n_features {
                    let mean = self.feature_means[&class][i];
                    let var = self.feature_vars[&class][i];
                    let exponent = (-1.0 * (x[i] - mean).powi(2)) / (2.0 * var);
                    log_prob += exponent - 0.5 * (2.0 * std::f64::consts::PI * var).ln();
                }
                class_probs[class as usize] = log_prob;
            }
            probs.push(class_probs);
        }
        probs = probs
            .iter()
            .map(|class_probs| class_probs.iter().map(|x| x.exp()).collect::<Vec<f64>>())
            .collect::<Vec<Vec<f64>>>();
        probs
    }

    /// Predict the class labels for the provided test data. The class label is the class with the highest probability.
    /// # Parameters
    /// x_test: `Vec<Vec<f64>>` - The test input samples. A 2D vector of shape (n_samples, n_features)
    /// # Returns
    /// `Vec<i32>` - A 1D vector of length n_samples containing the predicted class labels
    /// # Example
    /// ```
    /// use rusty_math::naive_bayes::NaiveBayesClassifier;
    /// let mut model = NaiveBayesClassifier::new();
    /// model.fit(&x_train, &y_train);
    /// let preds = model.predict(&x_test);
    /// ```
    /// # Panics
    /// This method will panic if the number of features in x_test is not equal to the number of features in x_train
    /// # Note
    /// The predict method is a wrapper around the predict_proba method. It calculates the class probabilities using the predict_proba method and returns the class with the highest probability.
    /// If you need the class probabilities, use the predict_proba method.
    pub fn predict(&self, x_test: &Vec<Vec<f64>>) -> Vec<i32> {
        let probs = self.predict_proba(x_test);
        let mut preds = Vec::new();
        for prob in probs.iter() {
            let mut max_prob = 0.0;
            let mut max_class = 0;
            for (i, &p) in prob.iter().enumerate() {
                if p > max_prob {
                    max_prob = p;
                    max_class = i;
                }
            }
            preds.push(max_class as i32);
        }
        preds
    }

    /// Predict the log probabilities for the provided test data. Log probabilities are calculated by taking the natural logarithm of the class probabilities.
    /// # Parameters
    /// x_test: `Vec<Vec<f64>>` - The test input samples. A 2D vector of shape (n_samples, n_features)
    /// # Returns
    /// `Vec<Vec<f64>>` - A 2D vector of shape (n_samples, n_classes) containing the log probabilities for each sample
    /// # Example
    /// ```
    /// use rusty_math::naive_bayes::NaiveBayesClassifier;
    /// let mut model = NaiveBayesClassifier::new();
    /// model.fit(&x_train, &y_train);
    /// let log_probs = model.predict_log_proba(&x_test);
    /// ```
    /// # Panics
    /// This method will panic if the number of features in x_test is not equal to the number of features in x_train
    /// # Note
    /// The predict_log_proba method is a wrapper around the predict_proba method. It calculates the class probabilities using the predict_proba method and returns the natural logarithm of the class probabilities.
    /// If you need the class probabilities, use the predict_proba method.
    pub fn predict_log_proba(&self, x_test: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let probs = self.predict_proba(x_test);
        probs
            .iter()
            .map(|class_probs| class_probs.iter().map(|x| x.ln()).collect::<Vec<f64>>())
            .collect::<Vec<Vec<f64>>>()
    }


    /// Score the model using the provided test data. The score is calculated as the accuracy of the model on the test data.
    /// See the accuracy function in the metrics module for more details.
    /// # Parameters
    /// x_test: `Vec<Vec<f64>>` - The test input samples. A 2D vector of shape (n_samples, n_features)
    /// y_test: `Vec<i32>` - The target classes. A 1D vector of length n_samples
    /// # Returns
    /// `HashMap<String, f64>` - A hashmap containing the accuracy of the model
    /// # Example
    /// ```
    /// use rusty_math::naive_bayes::NaiveBayesClassifier;
    /// let mut model = NaiveBayesClassifier::new();
    /// model.fit(&x_train, &y_train);
    /// let score = model.score(&x_test, &y_test);
    /// ```
    pub fn score(&self, x_test: &Vec<Vec<f64>>, y_test: &Vec<i32>) -> HashMap<String, f64> {
        let preds = self.predict(&x_test);
        accuracy(&preds, &y_test)
    }

}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;

    #[test]
    fn test_naive_bayes() {
        let x_train = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
            vec![4.0, 5.0],
            vec![5.0, 6.0],
            vec![6.0, 7.0],
        ];
        let y_train = vec![0, 0, 0, 1, 1, 1];
        let x_test = vec![vec![-1.0, 0.0], vec![4.0, 5.0], vec![7.0, 7.0], vec![1.0, 9.0]];
        let mut model = NaiveBayesClassifier::new();
        model.fit(&x_train, &y_train);
        let preds = model.predict(&x_test);
        assert_eq!(preds, vec![0, 1, 1, 0]);
        // assert_eq!(preds, vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 1.0]]);
        // println!("{:?}", preds);
    }
}