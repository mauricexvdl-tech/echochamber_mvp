//! Sinkhorn-Knopp normalization for doubly-stochastic edge weights.
//!
//! This module implements the Sinkhorn-Knopp algorithm to project weight
//! matrices onto the Birkhoff polytope (set of doubly-stochastic matrices).
//!
//! Reference: DeepSeek "Manifold-Constrained Hyper-Connections" (arXiv:2512.24880)
//!
//! Key properties of doubly-stochastic matrices:
//! - All entries are non-negative
//! - Each row sums to 1 (outgoing mass is conserved)
//! - Each column sums to 1 (incoming mass is conserved)
//! - Eigenvalues bounded: prevents signal explosion/collapse
//!
//! This provides "mass-preserving routing" which stabilizes signal propagation
//! across different random seeds.

/// Compute Sinkhorn-Knopp normalized weights for a graph.
///
/// Takes raw edge weights and iteratively normalizes rows and columns
/// until the matrix is approximately doubly-stochastic.
///
/// # Arguments
/// * `num_nodes` - Number of nodes in the graph
/// * `edges` - List of (from, to, raw_weight) tuples
/// * `iterations` - Number of Sinkhorn iterations (typically 10-20)
/// * `eps` - Small constant to avoid division by zero
///
/// # Returns
/// Vector of normalized weights in the same order as input edges
pub fn compute_sinkhorn_weights(
    num_nodes: usize,
    edges: &[(usize, usize, f64)],
    iterations: usize,
    eps: f64,
) -> Vec<f64> {
    if edges.is_empty() || num_nodes == 0 {
        return Vec::new();
    }

    // Build weight matrix from edges
    let mut w = vec![vec![0.0; num_nodes]; num_nodes];
    for &(from, to, weight) in edges {
        if from < num_nodes && to < num_nodes {
            w[from][to] = weight.max(eps); // Ensure positive
        }
    }

    // Sinkhorn-Knopp iterations
    for _ in 0..iterations {
        // Row normalization: each row sums to 1
        for row in &mut w {
            let row_sum: f64 = row.iter().sum();
            if row_sum > eps {
                for val in row.iter_mut() {
                    *val /= row_sum;
                }
            }
        }

        // Column normalization: each column sums to 1
        for j in 0..num_nodes {
            let col_sum: f64 = (0..num_nodes).map(|i| w[i][j]).sum();
            if col_sum > eps {
                for i in 0..num_nodes {
                    w[i][j] /= col_sum;
                }
            }
        }
    }

    // Extract normalized weights in edge order
    edges
        .iter()
        .map(|&(from, to, _)| {
            if from < num_nodes && to < num_nodes {
                w[from][to]
            } else {
                0.0
            }
        })
        .collect()
}

/// Compute blended weights: mix of uniform and Sinkhorn-normalized.
///
/// This allows gradual transition from uniform weights to fully normalized.
///
/// # Arguments
/// * `num_nodes` - Number of nodes
/// * `edges` - List of (from, to, raw_weight) tuples
/// * `iterations` - Sinkhorn iterations
/// * `blend` - Blend factor: 0.0 = uniform, 1.0 = full Sinkhorn
/// * `eps` - Small constant
///
/// # Returns
/// Blended weights
pub fn compute_blended_weights(
    num_nodes: usize,
    edges: &[(usize, usize, f64)],
    iterations: usize,
    blend: f64,
    eps: f64,
) -> Vec<f64> {
    if edges.is_empty() {
        return Vec::new();
    }

    let blend = blend.clamp(0.0, 1.0);

    // Compute uniform weights (1/out_degree for each edge)
    let mut out_degrees = vec![0usize; num_nodes];
    for &(from, _, _) in edges {
        if from < num_nodes {
            out_degrees[from] += 1;
        }
    }

    let uniform_weights: Vec<f64> = edges
        .iter()
        .map(|&(from, _, _)| {
            if from < num_nodes && out_degrees[from] > 0 {
                1.0 / (out_degrees[from] as f64).sqrt()
            } else {
                0.0
            }
        })
        .collect();

    if blend < eps {
        return uniform_weights;
    }

    // Compute Sinkhorn weights
    let sinkhorn_weights = compute_sinkhorn_weights(num_nodes, edges, iterations, eps);

    // Blend: (1 - blend) * uniform + blend * sinkhorn
    uniform_weights
        .iter()
        .zip(sinkhorn_weights.iter())
        .map(|(&u, &s)| (1.0 - blend) * u + blend * s)
        .collect()
}

/// Check if a matrix is approximately doubly-stochastic.
///
/// Returns (max_row_error, max_col_error) where errors are |sum - 1|.
pub fn check_doubly_stochastic(matrix: &[Vec<f64>]) -> (f64, f64) {
    let n = matrix.len();
    if n == 0 {
        return (0.0, 0.0);
    }

    // Check row sums
    let max_row_err = matrix
        .iter()
        .map(|row| (row.iter().sum::<f64>() - 1.0).abs())
        .fold(0.0, f64::max);

    // Check column sums
    let max_col_err = (0..n)
        .map(|j| ((0..n).map(|i| matrix[i][j]).sum::<f64>() - 1.0).abs())
        .fold(0.0, f64::max);

    (max_row_err, max_col_err)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sinkhorn_simple() {
        // Simple 3-node graph: 0->1, 0->2, 1->2, 2->0
        let edges = vec![
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 2, 1.0),
            (2, 0, 1.0),
        ];

        let weights = compute_sinkhorn_weights(3, &edges, 20, 1e-10);

        // All weights should be positive
        assert!(weights.iter().all(|&w| w > 0.0));

        // Weights should be bounded (not exploding)
        assert!(weights.iter().all(|&w| w <= 2.0));
    }

    #[test]
    fn test_sinkhorn_preserves_structure() {
        // Edges that exist should have positive weight
        // Edges that don't exist should remain zero
        let edges = vec![
            (0, 1, 1.0),
            (1, 0, 1.0),
        ];

        let weights = compute_sinkhorn_weights(3, &edges, 20, 1e-10);

        assert_eq!(weights.len(), 2);
        assert!(weights[0] > 0.0);
        assert!(weights[1] > 0.0);
    }

    #[test]
    fn test_sinkhorn_convergence() {
        // Build a complete graph and check doubly-stochastic property
        let n = 4;
        let mut edges = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    edges.push((i, j, 1.0));
                }
            }
        }

        let weights = compute_sinkhorn_weights(n, &edges, 50, 1e-10);

        // Rebuild matrix
        let mut matrix = vec![vec![0.0; n]; n];
        for (idx, &(from, to, _)) in edges.iter().enumerate() {
            matrix[from][to] = weights[idx];
        }

        let (row_err, col_err) = check_doubly_stochastic(&matrix);

        // Should be approximately doubly-stochastic
        assert!(
            row_err < 0.1,
            "Row sums should be ~1, max error: {}",
            row_err
        );
        assert!(
            col_err < 0.1,
            "Col sums should be ~1, max error: {}",
            col_err
        );
    }

    #[test]
    fn test_blended_weights() {
        let edges = vec![
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 2, 1.0),
        ];

        // blend = 0 should give uniform weights
        let uniform = compute_blended_weights(3, &edges, 20, 0.0, 1e-10);
        let full_sinkhorn = compute_blended_weights(3, &edges, 20, 1.0, 1e-10);
        let half = compute_blended_weights(3, &edges, 20, 0.5, 1e-10);

        // Half blend should be between uniform and sinkhorn
        for i in 0..edges.len() {
            let expected = 0.5 * uniform[i] + 0.5 * full_sinkhorn[i];
            assert!(
                (half[i] - expected).abs() < 1e-6,
                "Blend interpolation failed"
            );
        }
    }

    #[test]
    fn test_empty_graph() {
        let weights = compute_sinkhorn_weights(0, &[], 20, 1e-10);
        assert!(weights.is_empty());

        let weights2 = compute_sinkhorn_weights(5, &[], 20, 1e-10);
        assert!(weights2.is_empty());
    }
}
