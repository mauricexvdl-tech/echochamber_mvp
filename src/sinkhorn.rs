//! Sinkhorn-Knopp algorithm for doubly-stochastic edge weight normalization.
//!
//! Based on DeepSeek's mHC (Manifold-Constrained Hyper-Connections) approach.
//! Makes the transition matrix doubly-stochastic, which:
//! - Stabilizes signal propagation (prevents exponential amplification)
//! - Projects weights onto the Birkhoff polytope
//! - Ensures both incoming and outgoing signal distributions are normalized

/// Compute Sinkhorn-Knopp normalized edge weights from an adjacency list.
///
/// The algorithm iteratively normalizes rows and columns to achieve
/// a doubly-stochastic matrix where both rows and columns sum to 1.
///
/// # Arguments
/// * `adj` - Adjacency list (row = source node, values = target node indices)
/// * `iterations` - Number of Sinkhorn iterations (10-20 typically sufficient)
/// * `eps` - Small constant for numerical stability
///
/// # Returns
/// Vec of (from, to, weight) tuples representing normalized edge weights.
/// The weights satisfy:
/// - For each source node i: sum of outgoing weights ≈ 1
/// - For each target node j: sum of incoming weights ≈ 1
pub fn compute_sinkhorn_weights(
    adj: &[Vec<usize>],
    iterations: usize,
    eps: f64,
) -> Vec<(usize, usize, f64)> {
    let n = adj.len();
    if n == 0 {
        return Vec::new();
    }

    // Build initial weight matrix from adjacency list
    // Start with uniform weights per edge
    let mut w: Vec<Vec<f64>> = vec![vec![0.0; n]; n];

    for (from, neighbors) in adj.iter().enumerate() {
        if neighbors.is_empty() {
            continue;
        }
        // Initial weight: uniform across outgoing edges
        let initial_weight = 1.0;
        for &to in neighbors {
            // Handle potential multi-edges by accumulating
            w[from][to] += initial_weight;
        }
    }

    // Sinkhorn-Knopp iterations
    for _ in 0..iterations {
        // Row normalization (outgoing edges sum to 1)
        for row in &mut w {
            let sum: f64 = row.iter().sum();
            if sum > eps {
                for val in row.iter_mut() {
                    *val /= sum;
                }
            }
        }

        // Column normalization (incoming edges sum to 1)
        for j in 0..n {
            let col_sum: f64 = (0..n).map(|i| w[i][j]).sum();
            if col_sum > eps {
                for i in 0..n {
                    w[i][j] /= col_sum;
                }
            }
        }
    }

    // Final row normalization to ensure outgoing sums to 1
    // (This is the "sending" side which directly affects propagation)
    for row in &mut w {
        let sum: f64 = row.iter().sum();
        if sum > eps {
            for val in row.iter_mut() {
                *val /= sum;
            }
        }
    }

    // Extract non-zero weights as edge list
    let mut edges = Vec::new();
    for (from, row) in w.iter().enumerate() {
        for (to, &weight) in row.iter().enumerate() {
            if weight > eps {
                edges.push((from, to, weight));
            }
        }
    }

    edges
}

/// Compute edge weights with a softer Sinkhorn approach.
///
/// Instead of strict doubly-stochastic, this balances between
/// the original degree-based normalization and full Sinkhorn.
///
/// # Arguments
/// * `adj` - Adjacency list
/// * `alpha` - Blending factor: 0.0 = pure degree-based, 1.0 = pure Sinkhorn
/// * `iterations` - Sinkhorn iterations
/// * `eps` - Numerical stability constant
pub fn compute_blended_weights(
    adj: &[Vec<usize>],
    alpha: f64,
    iterations: usize,
    eps: f64,
) -> Vec<(usize, usize, f64)> {
    let n = adj.len();
    if n == 0 {
        return Vec::new();
    }

    // Compute pure Sinkhorn weights
    let sinkhorn_edges = compute_sinkhorn_weights(adj, iterations, eps);

    // Build lookup for Sinkhorn weights
    let mut sinkhorn_map: std::collections::HashMap<(usize, usize), f64> =
        std::collections::HashMap::new();
    for (from, to, weight) in sinkhorn_edges {
        sinkhorn_map.insert((from, to), weight);
    }

    // Compute degree-based weights and blend
    let mut blended = Vec::new();
    for (from, neighbors) in adj.iter().enumerate() {
        if neighbors.is_empty() {
            continue;
        }
        let degree_weight = 1.0 / (neighbors.len() as f64).sqrt();

        for &to in neighbors {
            let sinkhorn_weight = *sinkhorn_map.get(&(from, to)).unwrap_or(&0.0);
            let final_weight = (1.0 - alpha) * degree_weight + alpha * sinkhorn_weight;
            blended.push((from, to, final_weight));
        }
    }

    blended
}

/// Measure how close a weight matrix is to doubly-stochastic.
///
/// Returns (row_error, col_error) where each is the max deviation from 1.0.
pub fn measure_stochasticity(adj: &[Vec<usize>], weights: &[(usize, usize, f64)]) -> (f64, f64) {
    let n = adj.len();
    if n == 0 {
        return (0.0, 0.0);
    }

    // Build weight matrix from edges
    let mut w: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for &(from, to, weight) in weights {
        w[from][to] = weight;
    }

    // Compute row sums (outgoing)
    let mut max_row_error = 0.0_f64;
    for row in &w {
        let sum: f64 = row.iter().sum();
        if sum > 0.0 {
            let error = (sum - 1.0).abs();
            max_row_error = max_row_error.max(error);
        }
    }

    // Compute column sums (incoming)
    let mut max_col_error = 0.0_f64;
    for j in 0..n {
        let sum: f64 = (0..n).map(|i| w[i][j]).sum();
        if sum > 0.0 {
            let error = (sum - 1.0).abs();
            max_col_error = max_col_error.max(error);
        }
    }

    (max_row_error, max_col_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sinkhorn_basic() {
        // Simple triangle: 0->1, 1->2, 2->0
        let adj = vec![vec![1], vec![2], vec![0]];
        let weights = compute_sinkhorn_weights(&adj, 20, 1e-10);

        // Should have 3 edges with weights summing to 1 per row/col
        assert_eq!(weights.len(), 3);

        let (row_err, col_err) = measure_stochasticity(&adj, &weights);
        assert!(row_err < 0.01, "Row error too high: {}", row_err);
        assert!(col_err < 0.01, "Col error too high: {}", col_err);
    }

    #[test]
    fn test_sinkhorn_fully_connected() {
        // 4 nodes, all connected to all others
        let adj = vec![
            vec![1, 2, 3],
            vec![0, 2, 3],
            vec![0, 1, 3],
            vec![0, 1, 2],
        ];
        let weights = compute_sinkhorn_weights(&adj, 20, 1e-10);

        let (row_err, col_err) = measure_stochasticity(&adj, &weights);
        assert!(row_err < 0.01, "Row error too high: {}", row_err);
        assert!(col_err < 0.01, "Col error too high: {}", col_err);
    }

    #[test]
    fn test_blended_weights() {
        let adj = vec![vec![1, 2], vec![0, 2], vec![0, 1]];

        // alpha=0 should give pure degree-based
        let degree_weights = compute_blended_weights(&adj, 0.0, 20, 1e-10);
        for (_, _, w) in &degree_weights {
            let expected = 1.0 / (2.0_f64).sqrt();
            assert!((w - expected).abs() < 0.01);
        }

        // alpha=1 should give pure Sinkhorn
        let sinkhorn_weights = compute_blended_weights(&adj, 1.0, 20, 1e-10);
        let pure_sinkhorn = compute_sinkhorn_weights(&adj, 20, 1e-10);

        // Check they're similar
        assert_eq!(sinkhorn_weights.len(), pure_sinkhorn.len());
    }
}
