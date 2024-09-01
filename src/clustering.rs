//! # Clustering Algorithms
//! Clustering is a way of grouping the data points into different clusters, consisting of similar data points. The objects with the possible similarities remain in a group that has less or no similarities with another group.
//! The clustering technique is commonly used for statistical data analysis.
//! Algorithms:
//! 1. KMeans

use std::collections::HashMap;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::seq::index::sample;



fn euclidean_distance(x1: &Vec<f64>, x2: &Vec<f64>) -> f64 {
    let mut sum = 0.0;
    for i in 0..x1.len() {
        sum += (x1[i] - x2[i]).powi(2);
    }
    // sum.sqrt()
    sum
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



/// # KMeans Algorithm
/// KMeans is a clustering algorithm that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean.  
/// The algorithm works as follows:
/// 1. Randomly initialize k cluster centers
/// 2. Assign each data point to the nearest cluster center
/// 3. Update the cluster centers by taking the mean of all data points assigned to that cluster
/// 4. Repeat steps 2 and 3 until the cluster centers do not change beyond a tolerance or the maximum number of iterations is reached  
///   
/// In some cases, K is not clearly defined, and we have to think about the optimal number of K. K Means clustering performs best data is well separated. When data points overlapped this clustering is not suitable. K Means is faster as compare to other clustering technique. It provides strong coupling between the data points. K Means cluster do not provide clear information regarding the quality of clusters. Different initial assignment of cluster centroid may lead to different clusters. Also, K Means algorithm is sensitive to noise. It may have stuck in local minima.
/// The model is scored using the inertia, which is the sum of squared distances of samples to their closest cluster center. The inertia is a measure of how tightly the clusters are packed.
/// The KMeans struct has the following fields:
/// - `n_clusters`: `i32` -  Number of clusters to form
/// - `max_iter`: `i32` -  Maximum number of iterations to perform
/// - `tol`: `f64` -  Relative tolerance with regards to inertia to declare convergence
/// - `random_state`: `i32` -  Seed for random number generator
/// - `labels`: `Vec<f64>` -  Labels of each point
/// - `cluster_centers`: `Vec<Vec<f64>>` -  Coordinates of cluster centers
/// - `metric`: `String` -  Distance metric to use. Possible values are 'euclidean', 'minkowski', 'cosine', 'manhattan'
/// - `p`: `Option<i32>` -  Power parameter for the Minkowski metric. The field should be `None` if the metric is not 'minkowski'.
pub struct KMeans{
    pub n_clusters: i32,
    pub max_iter: i32,
    pub tol: f64,
    pub random_state: i32,
    pub labels: Vec<f64>,
    pub cluster_centers: Vec<Vec<f64>>,
    pub metric: String,
    pub p: Option<i32>,
    x_train: Vec<Vec<f64>>,
}

impl KMeans{


    /// Create a new KMeans instance
    /// # Parameters
    /// - `n_clusters`: `i32` -  Number of clusters to form
    /// - `max_iter`: `i32` -  Maximum number of iterations to perform
    /// - `tol`: `f64` -  Relative tolerance with regards to inertia to declare convergence
    /// - `random_state`: `i32` -  Seed for random number generator
    /// - `metric`: `String` -  Distance metric to use. Possible values are 'euclidean', 'minkowski', 'cosine', 'manhattan'
    /// - `p`: `Option<i32>` -  Power parameter for the Minkowski metric. The field should be `None` if the metric is not 'minkowski'.
    /// # Returns
    /// `KMeans` - A new instance of KMeans
    /// # Examples
    /// ```
    /// use rusty_machine::clustering::KMeans;
    /// let kmeans = KMeans::new(3, 100, 0.001, 0, "euclidean".to_string(), None);
    /// ```
    /// # Panics
    /// - If the metric is not one of 'euclidean', 'minkowski', 'cosine', 'manhattan'.
    /// - If the metric is 'minkowski' and the power parameter is None.
    pub fn new(n_clusters: i32, max_iter: i32, tol: f64, random_state: i32, metric: String, p: Option<i32>) -> KMeans{
        if metric != "euclidean" && metric != "minkowski" && metric != "cosine" && metric != "manhattan"{
            panic!("metric must be one of 'euclidean', 'minkowski', 'cosine', 'manhattan");
        }
        if metric == "minkowski" && p.is_none(){
            panic!("p must be provided for minkowski metric");
        }
        KMeans{
            n_clusters,
            max_iter,
            tol,
            random_state,
            labels: Vec::new(),
            cluster_centers: Vec::new(),
            metric,
            p,
            x_train: Vec::new(),
        }
    }


    
    /// Fit the KMeans model to the training data.
    /// # Parameters
    /// - `x`: `&Vec<Vec<f64>>` -  The training input samples. A 2D vector of shape (n_samples, n_features)
    /// # Example
    /// ```
    /// use rusty_machine::clustering::KMeans;
    /// let mut kmeans = KMeans::new(2, 100, 0.001, 0, "euclidean".to_string(), None);
    /// kmeans.fit(&vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0], vec![7.0, 8.0]]);
    /// ```
    /// # Panics
    /// - If the number of clusters is greater than the number of samples.
    pub fn fit (&mut self, x: &Vec<Vec<f64>>){
        self.labels = vec![0.0; x.len()];
        self.x_train = x.clone();
        if self.n_clusters > x.len() as i32{
            panic!("Number of clusters is greater than number of samples");
        }
        // let mut rng = thread_rng().seed([self.random_state as u64, 0]);
        // let centre_indices: Vec<usize> = sample(&mut rng, x.len(), self.n_clusters as usize).iter().map(|i| i as usize).collect();
        let mut rng = StdRng::seed_from_u64(self.random_state as u64);
        let centre_indices: Vec<usize> = sample(&mut rng, x.len(), self.n_clusters as usize).iter().map(|i| i as usize).collect();
        for i in 0..self.n_clusters{
            self.cluster_centers.push(x[centre_indices[i as usize]].clone());
        }
        let mut iter = 0;
        loop{
            for i in 0..x.len(){
                let mut min_dist = std::f64::INFINITY;
                let mut min_index = 0;
                for j in 0..self.n_clusters{
                    let dist = get_dist(&x[i], &self.cluster_centers[j as usize], self.metric.clone(), self.p);
                    if dist < min_dist{
                        min_dist = dist;
                        min_index = j;
                    }
                }
                self.labels[i] = min_index as f64;
            }
            let mut new_cluster_centers = vec![vec![0.0; x[0].len()]; self.n_clusters as usize];
            let mut cluster_sizes = vec![0.0; self.n_clusters as usize];
            for i in 0..x.len(){
                let cluster_index = self.labels[i] as usize;
                for j in 0..x[i].len(){
                    new_cluster_centers[cluster_index][j] += x[i][j];
                }
                cluster_sizes[cluster_index] += 1.0;
            }
            for i in 0..self.n_clusters{
                for j in 0..x[0].len(){
                    new_cluster_centers[i as usize][j] /= cluster_sizes[i as usize];
                }
            }
            let mut diff = 0.0;
            for i in 0..self.n_clusters{
                diff += get_dist(&new_cluster_centers[i as usize], &self.cluster_centers[i as usize], self.metric.clone(), self.p);
            }
            diff /= self.n_clusters as f64;
            self.cluster_centers = new_cluster_centers;
            iter += 1;
            if diff < self.tol || iter >= self.max_iter{
                break;
            }
        }        
        
    }


    /// Predict the closest cluster each sample in x belongs to. 
    /// # Parameters
    /// - `x`: `&Vec<Vec<f64>>` -  The input samples. A 2D vector of shape (n_samples, n_features)
    /// # Returns
    /// `Vec<f64>` -  Index of the cluster each sample belongs to
    /// # Example
    /// ```
    /// use rusty_machine::clustering::KMeans;
    /// let kmeans = KMeans::new(2, 100, 0.001, 0, "euclidean".to_string(), None);
    /// kmeans.fit(&vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0], vec![7.0, 8.0]]);
    /// let labels = kmeans.predict(&vec![vec![3.0, 2.0], vec![-1.0, 4.0]]);
    /// ```
    /// # Panics
    /// - If the number of features in input data does not match training data.
    pub fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<f64>{
        if x[0].len() != self.x_train[0].len(){
            panic!("Number of features in input data does not match training data");
        }
        let mut labels = vec![0.0; x.len()];
        for i in 0..x.len(){
            let mut min_dist = std::f64::INFINITY;
            let mut min_index = 0;
            for j in 0..self.n_clusters{
                let dist = get_dist(&x[i], &self.cluster_centers[j as usize], self.metric.clone(), self.p);
                if dist < min_dist{
                    min_dist = dist;
                    min_index = j;
                }
            }
            labels[i] = min_index as f64;
        }
        labels
    }

    /// Fit the KMeans model to the training data and predict the closest cluster each sample in x belongs to. It is equivalent to calling fit and predict separately on the training data.
    /// # Parameters
    /// - `x`: `&Vec<Vec<f64>>` -  The training input samples. A 2D vector of shape (n_samples, n_features)
    /// # Returns
    /// `Vec<f64>` -  Index of the cluster each sample belongs to
    /// # Example
    /// ```
    /// use rusty_machine::clustering::KMeans;
    /// let mut kmeans = KMeans::new(2, 100, 0.001, 0, "euclidean".to_string(), None);
    /// let labels = kmeans.fit_predict(&vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0], vec![7.0, 8.0]]);
    /// ```
    /// # Panics
    /// - If the number of clusters is greater than the number of samples.
    pub fn fit_predict(&mut self, x: &Vec<Vec<f64>>) -> Vec<f64>{
        self.fit(x);
        self.labels.clone()
    }


    /// Transform the input data to a cluster-distance space. It calculates the distance of each sample to every cluster center.
    /// # Returns
    /// `Vec<Vec<f64>>` -  Distance of each sample to every cluster center. A 2D vector of shape (n_samples, n_clusters)
    /// # Example
    /// ```
    /// use rusty_machine::clustering::KMeans;
    /// let kmeans = KMeans::new(2, 100, 0.001, 0, "euclidean".to_string(), None);
    /// kmeans.fit(&vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0], vec![7.0, 8.0]]);
    /// let distances = kmeans.transform();
    /// ```
    pub fn transform(&self) -> Vec<Vec<f64>>{
        let mut distances = vec![vec![0.0; self.n_clusters as usize]; self.x_train.len()];
        for i in 0..self.x_train.len(){
            for j in 0..self.n_clusters as usize{
                distances[i][j] = get_dist(&self.x_train[i], &self.cluster_centers[j], self.metric.clone(), self.p);
            }
        }
        distances
    }


    /// Fit the KMeans model to the training data and transform the input data to a cluster-distance space. It is equivalent to calling fit and transform separately on the training data.
    /// # Parameters
    /// - `x`: `&Vec<Vec<f64>>` -  The training input samples. A 2D vector of shape (n_samples, n_features)
    /// # Returns
    /// `Vec<Vec<f64>>` -  Distance of each sample to every cluster center. A 2D vector of shape (n_samples, n_clusters)
    /// # Example
    /// ```
    /// use rusty_machine::clustering::KMeans;
    /// let mut kmeans = KMeans::new(2, 100, 0.001, 0, "euclidean".to_string(), None);
    /// let distances = kmeans.fit_transform(&vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0], vec![7.0, 8.0]]);
    /// ```
    /// # Panics
    /// - If the number of clusters is greater than the number of samples.
    pub fn fit_transform(&mut self, x: &Vec<Vec<f64>>) -> Vec<Vec<f64>>{
        self.fit(x);
        self.transform()
    }


    /// Get the parameters of the KMeans model. The parameters include the number of clusters, maximum number of iterations, tolerance, random state, distance metric, and power parameter.
    /// # Returns
    /// `HashMap<String,String>` -  A hashmap containing the parameters of the KMeans model
    /// # Example
    /// ```
    /// use rusty_machine::clustering::KMeans;
    /// let kmeans = KMeans::new(2, 100, 0.001, 0, "euclidean".to_string(), None);
    /// let params = kmeans.get_params();
    /// ```
    pub fn get_params(&self) -> HashMap<String,String>{
        let mut params = HashMap::new();
        params.insert("n_clusters".to_string(), self.n_clusters.to_string());
        params.insert("max_iter".to_string(), self.max_iter.to_string());
        params.insert("tol".to_string(), self.tol.to_string());
        params.insert("random_state".to_string(), self.random_state.to_string());
        params.insert("metric".to_string(), self.metric.clone());
        match self.p{
            Some(p) => params.insert("p".to_string(), p.to_string()),
            None => params.insert("p".to_string(), "None".to_string()),
        };
        params
    }        


    /// Get the inertia of the KMeans model. The inertia is the sum of squared distances of samples to their closest cluster center. The inertia is a measure of how tightly the clusters are packed. Lower inertia values mean better clustering.
    /// # Returns
    /// `f64` -  The inertia of the KMeans model
    /// # Example
    /// ```
    /// use rusty_machine::clustering::KMeans;
    /// let kmeans = KMeans::new(2, 100, 0.001, 0, "euclidean".to_string(), None);
    /// kmeans.fit(&vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0], vec![7.0, 8.0]]);
    /// let inertia = kmeans.inertia();
    /// ```
    pub fn inertia(&self) -> f64{
        let mut inertia = 0.0;
        for i in 0..self.x_train.len(){
            inertia += get_dist(&self.x_train[i], &self.cluster_centers[self.labels[i] as usize], "euclidean".to_string(), self.p);
        }
        inertia
    }

    

        


}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans(){
        let mut kmeans = KMeans::new(2, 100, 0.001, 0, "euclidean".to_string(), None);
        kmeans.fit(&vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0], vec![7.0, 8.0]]);
        let _labels = kmeans.predict(&vec![vec![3.0, 2.0], vec![-1.0, 4.0]]);
        // assert_eq!(labels, vec![0.0, 1.0]);
        let _distances = kmeans.transform();
        // assert_eq!(distances, vec![vec![2.0, 8.0], vec![8.0, 2.0], vec![18.0, 2.0], vec![32.0, 8.0]]);
        let _params = kmeans.get_params();
        let _inertia = kmeans.inertia();
        assert_eq!(_inertia, 8.0);
    }
    

}