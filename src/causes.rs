//! Latent cause injection for Echo Chamber.
//! Phase 1.3: Config-based parameters.

use crate::complex::Complex;
use crate::config::Config;
use crate::echo::EchoChamber;
use crate::rng::Rng;
use std::f64::consts::PI;

/// Latent cause configuration.
/// Each cause has DISJOINT injector nodes (no overlap between causes).
pub struct Causes {
    /// injectors[cause_id] = vector of node IDs
    pub injectors: Vec<Vec<usize>>,
    /// Phase per cause
    pub cause_phases: Vec<f64>,
    /// Config reference values
    pub inject_amp: f64,
    pub ph_noise: f64,
    pub noise_injects: usize,
    pub noise_amp: f64,
    pub prob_single_cause: f64,
}

impl Causes {
    /// Create a new Causes configuration with DISJOINT injector nodes.
    pub fn new(config: &Config, rng: &mut Rng) -> Self {
        let total_injectors = config.num_causes * config.injectors_per_cause;
        assert!(
            config.num_nodes >= total_injectors,
            "Need at least {} nodes for disjoint injectors",
            total_injectors
        );

        // Choose distinct nodes
        let mut all_chosen = Vec::with_capacity(total_injectors);
        while all_chosen.len() < total_injectors {
            let node = rng.next_usize(config.num_nodes);
            if !all_chosen.contains(&node) {
                all_chosen.push(node);
            }
        }

        // Assign to causes
        let mut injectors = Vec::with_capacity(config.num_causes);
        for cause in 0..config.num_causes {
            let start = cause * config.injectors_per_cause;
            let end = start + config.injectors_per_cause;
            injectors.push(all_chosen[start..end].to_vec());
        }

        Causes {
            injectors,
            cause_phases: config.cause_phases(),
            inject_amp: config.inject_amp,
            ph_noise: config.ph_noise,
            noise_injects: config.noise_injects,
            noise_amp: config.noise_amp,
            prob_single_cause: config.prob_single_cause,
        }
    }

    pub fn num_causes(&self) -> usize {
        self.injectors.len()
    }

    /// Sample which causes are active this tick.
    /// Returns a bitmask (bit i set = cause i active) and count.
    pub fn sample_active(&self, rng: &mut Rng) -> (u8, usize) {
        let p = rng.next_f64();
        let num_causes = self.num_causes();

        if p < self.prob_single_cause {
            // Single cause
            let cause = rng.next_usize(num_causes);
            (1u8 << cause, 1)
        } else {
            // Two distinct causes
            let c1 = rng.next_usize(num_causes);
            let mut c2 = rng.next_usize(num_causes);
            while c2 == c1 {
                c2 = rng.next_usize(num_causes);
            }
            ((1u8 << c1) | (1u8 << c2), 2)
        }
    }

    /// Inject signals for active causes + background noise.
    pub fn inject_for_tick(
        &self,
        rng: &mut Rng,
        chamber: &mut EchoChamber,
        active_mask: u8,
    ) {
        let n_nodes = chamber.nodes.len();

        // Inject for each active cause
        for (cause, injector_nodes) in self.injectors.iter().enumerate() {
            if (active_mask & (1u8 << cause)) != 0 {
                let base_phase = self.cause_phases[cause];

                for &node_id in injector_nodes {
                    let noise = rng.next_range(-self.ph_noise, self.ph_noise);
                    let phase = base_phase + noise;
                    let signal = Complex::from_polar(self.inject_amp, phase);
                    chamber.inject(node_id, signal);
                }
            }
        }

        // Background noise injections
        for _ in 0..self.noise_injects {
            let node_id = rng.next_usize(n_nodes);
            let phase = rng.next_range(0.0, 2.0 * PI);
            let signal = Complex::from_polar(self.noise_amp, phase);
            chamber.inject(node_id, signal);
        }
    }

    /// Compute z_inj: sum of injected complex signals (PRE-MIX context).
    pub fn compute_z_inj(&self, active_mask: u8) -> Complex {
        let mut z_inj = Complex::ZERO;
        for (cause, _) in self.injectors.iter().enumerate() {
            if (active_mask & (1u8 << cause)) != 0 {
                let amp = self.inject_amp * self.injectors[cause].len() as f64;
                z_inj += Complex::from_polar(amp, self.cause_phases[cause]);
            }
        }
        z_inj
    }
}

/// Get Top-K node IDs by amplitude (descending).
pub fn get_top_k(chamber: &EchoChamber, k: usize) -> Vec<(usize, f64)> {
    let mut indexed: Vec<(usize, f64)> = chamber
        .nodes
        .iter()
        .map(|n| (n.id, n.buffer.norm()))
        .collect();

    indexed.sort_by(|a, b| match b.1.partial_cmp(&a.1) {
        Some(std::cmp::Ordering::Equal) => a.0.cmp(&b.0),
        Some(ord) => ord,
        None => std::cmp::Ordering::Equal,
    });

    indexed.into_iter().take(k).collect()
}

/// Statistics for Top-K evaluation.
pub struct TopKStats {
    pub n_nodes: usize,
    pub num_causes: usize,
    pub topk_hits: Vec<u32>,
    pub topk_hits_when_cause: Vec<Vec<u32>>,
    pub sum_amp_hits: Vec<f64>,
}

impl TopKStats {
    pub fn new(n_nodes: usize, num_causes: usize) -> Self {
        TopKStats {
            n_nodes,
            num_causes,
            topk_hits: vec![0u32; n_nodes],
            topk_hits_when_cause: vec![vec![0u32; num_causes]; n_nodes],
            sum_amp_hits: vec![0.0; n_nodes],
        }
    }

    pub fn record_topk(&mut self, topk: &[(usize, f64)], active_mask: u8) {
        for &(node, amp) in topk {
            self.topk_hits[node] += 1;
            self.sum_amp_hits[node] += amp;

            for cause in 0..self.num_causes {
                if (active_mask & (1u8 << cause)) != 0 {
                    self.topk_hits_when_cause[node][cause] += 1;
                }
            }
        }
    }

    pub fn purity(&self, node: usize) -> f64 {
        if self.topk_hits[node] == 0 {
            return 0.0;
        }
        let total = self.topk_hits[node] as f64;
        self.topk_hits_when_cause[node]
            .iter()
            .map(|&h| h as f64 / total)
            .fold(0.0, f64::max)
    }

    pub fn dominant_cause(&self, node: usize) -> usize {
        self.topk_hits_when_cause[node]
            .iter()
            .enumerate()
            .max_by_key(|(_, &h)| h)
            .map(|(c, _)| c)
            .unwrap_or(0)
    }

    pub fn avg_amp_hits(&self, node: usize) -> f64 {
        if self.topk_hits[node] == 0 {
            0.0
        } else {
            self.sum_amp_hits[node] / self.topk_hits[node] as f64
        }
    }

    pub fn top_by_hits(
        &self,
        n: usize,
        total_ticks: usize,
    ) -> Vec<(usize, u32, f64, f64, usize, f64)> {
        let mut indexed: Vec<(usize, u32)> = self
            .topk_hits
            .iter()
            .enumerate()
            .map(|(i, &h)| (i, h))
            .collect();
        indexed.sort_by(|a, b| b.1.cmp(&a.1));

        indexed
            .into_iter()
            .take(n)
            .map(|(node, hits)| {
                let dominance = hits as f64 / total_ticks as f64 * 100.0;
                let purity = self.purity(node);
                let top_cause = self.dominant_cause(node);
                let avg_amp = self.avg_amp_hits(node);
                (node, hits, dominance, purity, top_cause, avg_amp)
            })
            .collect()
    }

    pub fn top_for_cause(&self, cause: usize, n: usize) -> Vec<(usize, u32, f64)> {
        let mut indexed: Vec<(usize, u32)> = self
            .topk_hits_when_cause
            .iter()
            .enumerate()
            .map(|(i, arr)| (i, arr.get(cause).copied().unwrap_or(0)))
            .collect();
        indexed.sort_by(|a, b| b.1.cmp(&a.1));

        indexed
            .into_iter()
            .take(n)
            .map(|(node, hits_c)| {
                let purity = self.purity(node);
                (node, hits_c, purity)
            })
            .collect()
    }

    pub fn coverage(&self) -> usize {
        self.topk_hits.iter().filter(|&&h| h > 0).count()
    }
}
