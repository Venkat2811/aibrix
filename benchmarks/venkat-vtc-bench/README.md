## tmp placeholder in Venkat's branchmarks & simulations

Just stashing it here for now

### refer to:
- benchmark_mp_advanced.py - Simulation implementation
- analyze_results_gpu.py - Analysis with GPU acceleration
- results/ - Basic statistics & Summary analysis
  - `run1--different-algos` - Analysis on different algorithms.  Points out that clamped linear is simple yet superior.
  - `run2--clamped_linear--several_variations` - Analysis on clamped linear with several variations. 
  - `run3--clamped_linear--several_variations` - Analysis on clamped linear with balanced and equal weights. 


### Note
Benchmark simulation & analysis was done with aid of several LLMs. I monitored changes,analyzed results closely and verified them at each step.

### Analysis Summary:

#### Algorithms Benchmarked

- **Modulo**
  - Distributes tokens by count % num_pods
  - Simple, but poor fairness characteristics
- **Log**
  - Applies a logarithmic scaling to token counts before assignment
  - Softens extreme variations in token counts
- **Quotient-Remainder**
  - Splits counts into count / num_pods (quotient) plus a remainder-based tie-breaker
  - Attempts to improve on modulo's stability
- **Clamped Linear**
  - Linearly scales assignments within a min/max bound
  - Improved fairness over simpler algorithms
- **Variants of Clamped Linear**
  - Fixed-threshold (e.g. min2500, min5000) - Set minimum token thresholds
  - Adaptive Balanced (0.5 fairness, 0.5 utilization) - Equal weight to both concerns
  - Adaptive Equal Weights (1.0 fairness, 1.0 utilization) - Full weight to both metrics
  - Plus adaptive versions with min-thresholds for hybrid approaches

#### Why "Adaptive Clamped Linear – Equal Weights" Excels

- **High Fairness Correlation**
  - Consistently ≈0.84–0.96 across bucket sizes (vs. ~0.00–0.05 for modulo/QR)
  - Maintains fairness even under high-usage and bursty workload patterns
- **Balanced Utilization**
  - Std dev ≈0.47–0.60 across configurations (vs. clamped_linear's extreme imbalance)
  - Avoids hotspots while preserving fairness properties
- **Strong Monotonicity**
  - Only ≈4–5% non-monotonic pairs vs. 45% in modulo at small bucket sizes
  - Consistent across all three workload patterns
- **Predictability**
  - More consistent per-user experience than baseline algorithms
  - Stable assignment across usage patterns
- **Robustness**
  - Performs well across all three workload distributions (balanced, high-usage, bursty)
  - Minimal parameter sensitivity across bucket sizes (1000-4000)
  - Resilient to changes in window size configuration

#### Trade-off Considerations
- Slightly higher computational complexity than simple algorithms
- Equal weights approach balances fairness and utilization without requiring custom tuning