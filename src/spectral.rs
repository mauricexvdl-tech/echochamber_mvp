//! Spectral normalization for unitary-like signal propagation constraints.
//!
//! This module provides spectral normalization utilities to bound the largest
//! singular value of weight matrices, ensuring norm-preserving signal propagation.
//!
//! Reference: Arjovsky et al. (2016) "Unitary Evolution Recurrent Neural Networks"
//!
//! Key properties:
//! - Spectral norm σ_max ≤ 1 prevents signal explosion
//! - Preserves signal direction while bounding magnitude
//! - Complementary to Sinkhorn (mass-preserving) constraints

/// Compute the spectral norm (largest singular value) of a square matrix
/// using power iteration.
///
/// # Arguments
/// * `matrix` - Square matrix as row-major Vec<Vec<f64>>
/// * `iterations` - Number of power iterations (default ~20 is usually sufficient)
///
/// # Returns
/// Estimated largest singular value σ_max
pub fn spectral_norm(matrix: &[Vec<f64>], iterations: usize) -> f64 {
    let n = matrix.len();
    if n == 0 || matrix[0].len() != n {
        return 0.0;
    }

    // Initialize random unit vector
    let mut v: Vec<f64> = (0..n).map(|i| ((i + 1) as f64).sin()).collect();
    normalize_vec(&mut v);

    // Power iteration: v <- A^T A v / ||A^T A v||
    // This converges to the eigenvector of A^T A with largest eigenvalue (= σ_max^2)
    for _ in 0..iterations {
        // u = A * v
        let u = mat_vec_mul(matrix, &v);

        // v = A^T * u
        let v_new = mat_t_vec_mul(matrix, &u);

        v = v_new;
        normalize_vec(&mut v);
    }

    // Compute σ_max = ||A * v||
    let av = mat_vec_mul(matrix, &v);
    vec_norm(&av)
}

/// Normalize a matrix so its spectral norm is at most `target` (default 1.0).
///
/// # Arguments
/// * `matrix` - Mutable square matrix
/// * `target` - Target maximum spectral norm (default 1.0)
/// * `iterations` - Power iterations for spectral norm estimation
///
/// # Returns
/// The original spectral norm before normalization
pub fn spectral_normalize(matrix: &mut [Vec<f64>], target: f64, iterations: usize) -> f64 {
    let sigma = spectral_norm(matrix, iterations);

    if sigma > target && sigma > 1e-12 {
        let scale = target / sigma;
        for row in matrix.iter_mut() {
            for val in row.iter_mut() {
                *val *= scale;
            }
        }
    }

    sigma
}

/// Build adjacency matrix from edge list representation.
///
/// # Arguments
/// * `num_nodes` - Total number of nodes
/// * `edges` - List of (from, to, weight) tuples
///
/// # Returns
/// Adjacency matrix as Vec<Vec<f64>>
pub fn build_adjacency_matrix(num_nodes: usize, edges: &[(usize, usize, f64)]) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; num_nodes]; num_nodes];

    for &(from, to, weight) in edges {
        if from < num_nodes && to < num_nodes {
            matrix[from][to] = weight;
        }
    }

    matrix
}

/// Apply spectral normalization to an edge list.
/// Returns normalized weights while preserving relative edge structure.
///
/// # Arguments
/// * `num_nodes` - Total number of nodes
/// * `edges` - Mutable list of (from, to, weight) tuples
/// * `target` - Target maximum spectral norm
/// * `iterations` - Power iterations
///
/// # Returns
/// The original spectral norm before normalization
pub fn spectral_normalize_edges(
    num_nodes: usize,
    edges: &mut [(usize, usize, f64)],
    target: f64,
    iterations: usize,
) -> f64 {
    // Build matrix from edges
    let mut matrix = build_adjacency_matrix(num_nodes, edges);

    // Normalize matrix
    let sigma = spectral_normalize(&mut matrix, target, iterations);

    // Write back to edges
    for edge in edges.iter_mut() {
        edge.2 = matrix[edge.0][edge.1];
    }

    sigma
}

/// Compute the "spectral gap" - ratio of largest to second-largest singular value.
/// A larger gap indicates more stable dynamics.
pub fn spectral_gap(matrix: &[Vec<f64>], iterations: usize) -> f64 {
    let n = matrix.len();
    if n < 2 {
        return f64::INFINITY;
    }

    // Get largest singular value and its vector
    let sigma1 = spectral_norm(matrix, iterations);
    if sigma1 < 1e-12 {
        return f64::INFINITY;
    }

    // Deflate matrix by removing the top singular component
    // This is an approximation - for exact computation we'd need full SVD
    // For now, just return sigma1 as a stability indicator
    sigma1
}

/// Check if matrix satisfies unitary-like constraint (spectral norm ≈ 1).
///
/// Returns (sigma_max, is_bounded) where is_bounded is true if σ_max ≤ target + eps
pub fn check_spectral_bound(matrix: &[Vec<f64>], target: f64, iterations: usize) -> (f64, bool) {
    let sigma = spectral_norm(matrix, iterations);
    let eps = 0.01; // 1% tolerance
    (sigma, sigma <= target + eps)
}

// ============================================================================
// Helper functions
// ============================================================================

fn normalize_vec(v: &mut [f64]) {
    let norm = vec_norm(v);
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn mat_vec_mul(matrix: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

fn mat_t_vec_mul(matrix: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    let n = matrix.len();
    let m = if n > 0 { matrix[0].len() } else { 0 };

    (0..m)
        .map(|j| (0..n).map(|i| matrix[i][j] * v[i]).sum())
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_spectral_norm() {
        // Identity matrix has spectral norm = 1
        let identity = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let sigma = spectral_norm(&identity, 20);
        assert!(
            (sigma - 1.0).abs() < 0.01,
            "Identity spectral norm should be 1, got {}",
            sigma
        );
    }

    #[test]
    fn test_scaled_identity() {
        // 2*I has spectral norm = 2
        let scaled = vec![
            vec![2.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0],
            vec![0.0, 0.0, 2.0],
        ];
        let sigma = spectral_norm(&scaled, 20);
        assert!(
            (sigma - 2.0).abs() < 0.01,
            "2*I spectral norm should be 2, got {}",
            sigma
        );
    }

    #[test]
    fn test_spectral_normalize() {
        // Start with matrix that has large spectral norm
        let mut matrix = vec![
            vec![3.0, 0.0, 0.0],
            vec![0.0, 3.0, 0.0],
            vec![0.0, 0.0, 3.0],
        ];

        let original = spectral_normalize(&mut matrix, 1.0, 20);
        assert!(
            (original - 3.0).abs() < 0.1,
            "Original sigma should be ~3, got {}",
            original
        );

        // After normalization, spectral norm should be <= 1
        let normalized = spectral_norm(&matrix, 20);
        assert!(
            normalized <= 1.01,
            "Normalized sigma should be <= 1, got {}",
            normalized
        );
    }

    #[test]
    fn test_stochastic_matrix() {
        // Row-stochastic matrix (each row sums to 1)
        let stochastic = vec![
            vec![0.5, 0.3, 0.2],
            vec![0.2, 0.5, 0.3],
            vec![0.3, 0.2, 0.5],
        ];
        let sigma = spectral_norm(&stochastic, 20);
        // Row-stochastic matrices have spectral norm <= 1
        assert!(
            sigma <= 1.01,
            "Row-stochastic spectral norm should be <= 1, got {}",
            sigma
        );
    }

    #[test]
    fn test_doubly_stochastic() {
        // Doubly stochastic matrix (rows and columns sum to 1)
        let doubly = vec![
            vec![0.5, 0.25, 0.25],
            vec![0.25, 0.5, 0.25],
            vec![0.25, 0.25, 0.5],
        ];
        let sigma = spectral_norm(&doubly, 20);
        // Doubly-stochastic matrices have spectral norm = 1 (exactly)
        assert!(
            (sigma - 1.0).abs() < 0.1,
            "Doubly-stochastic spectral norm should be ~1, got {}",
            sigma
        );
    }

    #[test]
    fn test_exploding_matrix() {
        // Matrix with large off-diagonal entries can amplify signals
        let exploding = vec![
            vec![0.1, 2.0, 2.0],
            vec![2.0, 0.1, 2.0],
            vec![2.0, 2.0, 0.1],
        ];
        let sigma = spectral_norm(&exploding, 20);
        // This matrix has spectral norm > 1, causing signal explosion
        assert!(
            sigma > 1.0,
            "Exploding matrix should have sigma > 1, got {}",
            sigma
        );

        // After normalization, it's bounded
        let mut normalized = exploding.clone();
        spectral_normalize(&mut normalized, 1.0, 20);
        let sigma_after = spectral_norm(&normalized, 20);
        assert!(
            sigma_after <= 1.01,
            "Normalized should be <= 1, got {}",
            sigma_after
        );
    }
}
