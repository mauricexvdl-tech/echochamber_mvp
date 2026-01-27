//! Phase 2.2b: Topology generation for deterministic graph structures.
//!
//! Provides:
//! - Watts-Strogatz Small-World network generation
//! - Uniform spread injector placement via farthest-point sampling

use crate::rng::Rng;
use std::collections::VecDeque;

/// Adjacency list representation of a graph.
pub type AdjList = Vec<Vec<usize>>;

/// Build a Watts-Strogatz Small-World graph.
///
/// Algorithm:
/// 1. Start with ring lattice: each node connects to k/2 neighbors on each side.
/// 2. For each edge (i, j) with j in clockwise direction, rewire with probability beta.
///
/// # Arguments
/// * `n` - Number of nodes
/// * `k` - Number of neighbors (must be even, >= 2)
/// * `beta` - Rewiring probability (0.0 = regular lattice, 1.0 = random)
/// * `seed` - RNG seed for deterministic generation
///
/// # Returns
/// Adjacency list (directed edges, but we add both directions for undirected)
pub fn build_small_world(n: usize, k: usize, beta: f32, seed: u64) -> AdjList {
    assert!(k >= 2, "k must be at least 2");
    assert!(k % 2 == 0, "k must be even");
    assert!(k < n, "k must be less than n");
    assert!((0.0..=1.0).contains(&beta), "beta must be in [0, 1]");

    let mut rng = Rng::new(seed);
    let half_k = k / 2;

    // Initialize adjacency list
    let mut adj: AdjList = vec![Vec::new(); n];

    // Step 1: Create ring lattice (connect each node to k/2 neighbors on each side)
    for i in 0..n {
        for offset in 1..=half_k {
            let j = (i + offset) % n;
            // Add undirected edge (both directions)
            if !adj[i].contains(&j) {
                adj[i].push(j);
            }
            if !adj[j].contains(&i) {
                adj[j].push(i);
            }
        }
    }

    // Step 2: Rewire edges with probability beta
    // Only consider "forward" edges (i -> j where j = (i + offset) % n)
    for i in 0..n {
        for offset in 1..=half_k {
            let j = (i + offset) % n;

            // Rewire with probability beta
            if rng.next_f64() < beta as f64 {
                // Find a new target that is:
                // - Not the same as i
                // - Not already connected to i
                let mut attempts = 0;
                let max_attempts = n * 2; // Prevent infinite loop

                while attempts < max_attempts {
                    let new_target = rng.next_usize(n);
                    if new_target != i && !adj[i].contains(&new_target) {
                        // Remove old edge (i, j) - both directions
                        adj[i].retain(|&x| x != j);
                        adj[j].retain(|&x| x != i);

                        // Add new edge (i, new_target) - both directions
                        adj[i].push(new_target);
                        adj[new_target].push(i);
                        break;
                    }
                    attempts += 1;
                }
            }
        }
    }

    // Debug assertions
    #[cfg(debug_assertions)]
    {
        // Check no self-loops
        for (i, neighbors) in adj.iter().enumerate() {
            assert!(!neighbors.contains(&i), "Self-loop detected at node {}", i);
        }

        // Check no duplicate edges
        for neighbors in &adj {
            let mut sorted = neighbors.clone();
            sorted.sort_unstable();
            for window in sorted.windows(2) {
                assert!(window[0] != window[1], "Duplicate edge detected");
            }
        }

        // Check degree is roughly k (may vary due to rewiring)
        let total_edges: usize = adj.iter().map(|v| v.len()).sum();
        let avg_degree = total_edges as f64 / n as f64;
        // Allow some variance due to rewiring
        assert!(
            avg_degree >= (k as f64 * 0.8) && avg_degree <= (k as f64 * 1.2),
            "Average degree {} is too far from k={}",
            avg_degree,
            k
        );
    }

    adj
}

/// Compute BFS distances from a source node.
///
/// # Returns
/// Vector of distances where `dist[i]` is the shortest path length from source to i.
/// Unreachable nodes have distance usize::MAX.
pub fn bfs_distances(adj: &AdjList, source: usize) -> Vec<usize> {
    let n = adj.len();
    let mut dist = vec![usize::MAX; n];
    let mut queue = VecDeque::new();

    dist[source] = 0;
    queue.push_back(source);

    while let Some(u) = queue.pop_front() {
        for &v in &adj[u] {
            if dist[v] == usize::MAX {
                dist[v] = dist[u] + 1;
                queue.push_back(v);
            }
        }
    }

    dist
}

/// Select injector nodes using farthest-point sampling.
///
/// This spreads injectors evenly across the graph by iteratively picking
/// the node that is farthest from all previously selected nodes.
///
/// # Arguments
/// * `adj` - Adjacency list of the graph
/// * `m` - Number of injectors to select
/// * `seed` - RNG seed (used only for initial node selection)
///
/// # Returns
/// Vector of selected node IDs
pub fn uniform_spread_injectors(adj: &AdjList, m: usize, seed: u64) -> Vec<usize> {
    let n = adj.len();
    assert!(m <= n, "Cannot select more injectors than nodes");

    if m == 0 {
        return Vec::new();
    }

    let mut rng = Rng::new(seed);
    let mut selected: Vec<usize> = Vec::with_capacity(m);

    // Pick first node based on seed
    let first = rng.next_usize(n);
    selected.push(first);

    // Track minimum distance from each node to any selected node
    let mut min_dist = bfs_distances(adj, first);

    // Iteratively select farthest node
    for _ in 1..m {
        // Find node with maximum min_dist (farthest from all selected)
        let mut best_node = 0;
        let mut best_dist = 0;

        for (node, &d) in min_dist.iter().enumerate() {
            if !selected.contains(&node) && d > best_dist && d != usize::MAX {
                best_dist = d;
                best_node = node;
            }
        }

        // If all remaining nodes are unreachable or already selected, pick randomly
        if best_dist == 0 {
            for node in 0..n {
                if !selected.contains(&node) {
                    best_node = node;
                    break;
                }
            }
        }

        selected.push(best_node);

        // Update min_dist with distances from new node
        let new_dist = bfs_distances(adj, best_node);
        for i in 0..n {
            if new_dist[i] < min_dist[i] {
                min_dist[i] = new_dist[i];
            }
        }
    }

    selected
}

/// Build a random Erdős-Rényi style graph (matches original random_graph behavior).
///
/// # Arguments
/// * `n` - Number of nodes
/// * `avg_degree` - Target average degree (total_edges = n * avg_degree)
/// * `seed` - RNG seed
///
/// # Returns
/// Adjacency list
pub fn build_random_er(n: usize, avg_degree: usize, seed: u64) -> AdjList {
    let mut rng = Rng::new(seed);
    let mut adj: AdjList = vec![Vec::new(); n];
    let total_edges = n * avg_degree;

    for _ in 0..total_edges {
        let from = rng.next_usize(n);
        let to = rng.next_usize(n);
        if from != to {
            // Note: allows multi-edges like original implementation
            adj[from].push(to);
        }
    }

    adj
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_world_basic() {
        let adj = build_small_world(32, 8, 0.05, 0xDEADBEEF);
        assert_eq!(adj.len(), 32);

        // Check no self-loops
        for (i, neighbors) in adj.iter().enumerate() {
            assert!(!neighbors.contains(&i));
        }

        // Check average degree is roughly k
        let total_edges: usize = adj.iter().map(|v| v.len()).sum();
        let avg_degree = total_edges as f64 / 32.0;
        assert!(avg_degree >= 6.0 && avg_degree <= 10.0);
    }

    #[test]
    fn test_small_world_deterministic() {
        let adj1 = build_small_world(32, 8, 0.05, 0xDEADBEEF);
        let adj2 = build_small_world(32, 8, 0.05, 0xDEADBEEF);
        assert_eq!(adj1, adj2);
    }

    #[test]
    fn test_uniform_spread() {
        let adj = build_small_world(32, 8, 0.05, 0xDEADBEEF);
        let injectors = uniform_spread_injectors(&adj, 6, 0xCAFE);
        assert_eq!(injectors.len(), 6);

        // Check all unique
        let mut sorted = injectors.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 6);
    }

    #[test]
    fn test_bfs_distances() {
        // Simple ring: 0-1-2-3-0
        let adj = vec![vec![1, 3], vec![0, 2], vec![1, 3], vec![2, 0]];
        let dist = bfs_distances(&adj, 0);
        assert_eq!(dist, vec![0, 1, 2, 1]);
    }
}
