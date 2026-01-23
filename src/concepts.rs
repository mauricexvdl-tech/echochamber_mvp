//! Concept readout and classification for Echo Chamber.
//! Phase 1.3: Turn emergent context-selective attractors into explicit "Concept" readouts.
//! Phase 1.4a: Added score method on ConceptBank for episodic memory.

/// Concept prototype: probability distribution over nodes.
#[derive(Clone, Debug)]
pub struct ConceptPrototype {
    /// Probability of each node being in Top-K for this concept.
    pub node_probs: Vec<f64>,
    /// Total hits used to build this prototype.
    pub total_hits: usize,
}

impl ConceptPrototype {
    pub fn new(num_nodes: usize) -> Self {
        ConceptPrototype {
            node_probs: vec![0.0; num_nodes],
            total_hits: 0,
        }
    }

    /// Add a Top-K observation: increment counts for the given nodes.
    pub fn record_topk(&mut self, topk_nodes: &[usize]) {
        for &node in topk_nodes {
            if node < self.node_probs.len() {
                self.node_probs[node] += 1.0;
            }
        }
        self.total_hits += 1;
    }

    /// Normalize counts to probabilities.
    pub fn normalize(&mut self) {
        if self.total_hits > 0 {
            let total_count: f64 = self.node_probs.iter().sum();
            if total_count > 0.0 {
                for p in &mut self.node_probs {
                    *p /= total_count;
                }
            }
        }
    }

    /// Compute Shannon entropy of the distribution (in bits).
    /// Lower entropy = sharper/more concentrated prototype.
    pub fn entropy(&self) -> f64 {
        let mut h = 0.0;
        for &p in &self.node_probs {
            if p > 1e-12 {
                h -= p * p.log2();
            }
        }
        h
    }

    /// Get top N nodes by probability.
    pub fn top_nodes(&self, n: usize) -> Vec<(usize, f64)> {
        let mut indexed: Vec<(usize, f64)> = self
            .node_probs
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.into_iter().take(n).collect()
    }

    /// Score a set of Top-K nodes against this prototype.
    /// Returns sum of probabilities for the given nodes.
    pub fn score(&self, topk_nodes: &[usize]) -> f64 {
        topk_nodes
            .iter()
            .filter(|&&n| n < self.node_probs.len())
            .map(|&n| self.node_probs[n])
            .sum()
    }
}

/// Confusion matrix for 3x3 classification.
#[derive(Clone, Debug)]
pub struct ConfusionMatrix {
    /// matrix[true_ctx][predicted_ctx] = count
    pub matrix: [[usize; 3]; 3],
    pub total: usize,
}

impl ConfusionMatrix {
    pub fn new() -> Self {
        ConfusionMatrix {
            matrix: [[0; 3]; 3],
            total: 0,
        }
    }

    pub fn record(&mut self, true_ctx: usize, predicted_ctx: usize) {
        if true_ctx < 3 && predicted_ctx < 3 {
            self.matrix[true_ctx][predicted_ctx] += 1;
            self.total += 1;
        }
    }

    /// Overall accuracy: correct predictions / total.
    pub fn accuracy(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let correct: usize = (0..3).map(|c| self.matrix[c][c]).sum();
        correct as f64 / self.total as f64
    }

    /// Per-class precision: correct_for_c / predicted_as_c.
    pub fn precision(&self, ctx: usize) -> f64 {
        let predicted_as_ctx: usize = (0..3).map(|t| self.matrix[t][ctx]).sum();
        if predicted_as_ctx == 0 {
            return 0.0;
        }
        self.matrix[ctx][ctx] as f64 / predicted_as_ctx as f64
    }

    /// Per-class recall: correct_for_c / true_c.
    pub fn recall(&self, ctx: usize) -> f64 {
        let true_ctx: usize = (0..3).map(|p| self.matrix[ctx][p]).sum();
        if true_ctx == 0 {
            return 0.0;
        }
        self.matrix[ctx][ctx] as f64 / true_ctx as f64
    }

    /// Print the confusion matrix.
    pub fn print(&self) {
        println!("Confusion Matrix (rows=true, cols=predicted):");
        println!("           │  ctx0  │  ctx1  │  ctx2  │");
        println!("───────────┼────────┼────────┼────────┤");
        for true_c in 0..3 {
            print!("  true ctx{} │", true_c);
            for pred_c in 0..3 {
                print!(" {:>5}  │", self.matrix[true_c][pred_c]);
            }
            println!();
        }
        println!();
        println!("  Overall accuracy: {:.1}%", self.accuracy() * 100.0);
        for c in 0..3 {
            println!(
                "  ctx{}: precision={:.1}%, recall={:.1}%",
                c,
                self.precision(c) * 100.0,
                self.recall(c) * 100.0
            );
        }
    }
}

/// Concept bank: stores prototypes per context (concept_id == ctx initially).
pub struct ConceptBank {
    pub prototypes: Vec<ConceptPrototype>,
    pub num_nodes: usize,
}

impl ConceptBank {
    pub fn new(num_ctx: usize, num_nodes: usize) -> Self {
        ConceptBank {
            prototypes: (0..num_ctx)
                .map(|_| ConceptPrototype::new(num_nodes))
                .collect(),
            num_nodes,
        }
    }

    /// Record a Top-K observation for a given context.
    pub fn record(&mut self, ctx: usize, topk_nodes: &[usize]) {
        if ctx < self.prototypes.len() {
            self.prototypes[ctx].record_topk(topk_nodes);
        }
    }

    /// Normalize all prototypes.
    pub fn normalize_all(&mut self) {
        for proto in &mut self.prototypes {
            proto.normalize();
        }
    }

    /// Score a set of Top-K nodes against a specific context's prototype.
    /// Returns the similarity score (sum of prototype probabilities for the nodes).
    pub fn score(&self, ctx: usize, topk_nodes: &[usize]) -> f64 {
        if ctx < self.prototypes.len() {
            self.prototypes[ctx].score(topk_nodes)
        } else {
            0.0
        }
    }

    /// Classify: given Top-K nodes, return predicted context.
    /// Uses argmax over prototype scores.
    pub fn classify(&self, topk_nodes: &[usize]) -> usize {
        let mut best_ctx = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (ctx, proto) in self.prototypes.iter().enumerate() {
            let score = proto.score(topk_nodes);
            if score > best_score {
                best_score = score;
                best_ctx = ctx;
            }
        }
        best_ctx
    }

    /// Print diagnostic info for all prototypes.
    pub fn print_diagnostics(&self) {
        println!("Concept Prototypes:");
        for (ctx, proto) in self.prototypes.iter().enumerate() {
            let top8 = proto.top_nodes(8);
            let entropy = proto.entropy();

            let top_str: String = top8
                .iter()
                .map(|(node, prob)| format!("{}:{:.3}", node, prob))
                .collect::<Vec<_>>()
                .join(", ");

            println!("  ctx{}: entropy={:.3} bits", ctx, entropy);
            println!("         top 8: {}", top_str);
        }
    }

    /// Compute concept purity: how often concept_id matches true ctx.
    /// This is just accuracy, but framed as "concept purity".
    pub fn concept_purity(&self, confusion: &ConfusionMatrix) -> f64 {
        confusion.accuracy()
    }
}
