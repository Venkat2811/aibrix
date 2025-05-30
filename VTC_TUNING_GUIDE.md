# VTC Configuration Tuning Guide

## 🎯 Current Performance Issues

Based on benchmark results with 10 QPS, 100 requests each (VTC vs Random):

### Performance Metrics

- **VTC Average Latency**: 19.47s (all user categories)
- **Random Average Latency**: 16.74s (all user categories)
- **Performance Degradation**: -16.3% (VTC worse than random)
- **Pod Concentration**: 74% of VTC requests → single pod

### Root Cause Analysis

1. **Pod Load Timing Issue**: VTC queries metrics BEFORE request forwarding
2. **Resource Constraint Paradox**: Limited CPU makes "busy" pod perform better
3. **Fairness vs Performance**: VTC prioritizes fairness over raw throughput

## ⚙️ Configuration Parameters

### Current Configuration (Problematic)

```yaml
AIBRIX_ROUTER_VTC_TOKEN_TRACKER_WINDOW_SIZE: "10"
AIBRIX_ROUTER_VTC_TOKEN_TRACKER_TIME_UNIT: "minutes"
AIBRIX_ROUTER_VTC_TOKEN_TRACKER_MIN_TOKENS: "20"
AIBRIX_ROUTER_VTC_TOKEN_TRACKER_MAX_TOKENS: "200"
```

### Recommended Optimized Configuration

```yaml
AIBRIX_ROUTER_VTC_TOKEN_TRACKER_WINDOW_SIZE: "2"
AIBRIX_ROUTER_VTC_TOKEN_TRACKER_TIME_UNIT: "seconds"
AIBRIX_ROUTER_VTC_TOKEN_TRACKER_MIN_TOKENS: "10"
AIBRIX_ROUTER_VTC_TOKEN_TRACKER_MAX_TOKENS: "100"
AIBRIX_ROUTER_VTC_BASIC_FAIRNESS_WEIGHT: "0.3"
AIBRIX_ROUTER_VTC_BASIC_UTILIZATION_WEIGHT: "0.7"
```

## 🔧 Step-by-Step Tuning Process

### Step 1: Apply Optimized Configuration

```bash
kubectl apply -f benchmarks/plot/aibrix0.3-routing/vtc-basic/config/gateway/vtc-bench-env-patch-optimized.yaml
```

### Step 2: Wait for Pod Restart

```bash
kubectl rollout status deployment/aibrix-gateway-plugins -n aibrix-system
```

### Step 3: Run Benchmark Test

```bash
python3 bench_and_collect_stats.py --requests 50 --pattern balanced --algorithms vtc-basic random --qps 5 --strict
```

### Step 4: Analyze Results

```bash
python3 run_analysis.py /tmp/aibrix-benchmark/run_results/[LATEST_RUN] --vtc-only
```

## 📊 Parameter Explanations

### Token Tracking Parameters

| Parameter     | Current | Optimized | Rationale                                 |
| ------------- | ------- | --------- | ----------------------------------------- |
| `WINDOW_SIZE` | 10 min  | 2 sec     | Faster adaptation to load changes         |
| `TIME_UNIT`   | minutes | seconds   | More granular tracking                    |
| `MIN_TOKENS`  | 20      | 10        | Capture smaller requests (~53 tokens avg) |
| `MAX_TOKENS`  | 200     | 100       | Tighter clustering for better fairness    |

### Routing Weight Parameters

| Parameter            | Default | Optimized | Rationale                 |
| -------------------- | ------- | --------- | ------------------------- |
| `FAIRNESS_WEIGHT`    | 0.5     | 0.3       | Reduce fairness priority  |
| `UTILIZATION_WEIGHT` | 0.5     | 0.7       | Prioritize load balancing |

## 🎯 Expected Improvements

### Performance Targets

- **Latency Improvement**: 10-15% reduction vs current VTC
- **Pod Distribution**: <60% concentration (vs 74% current)
- **Load Balancing**: More even distribution across 3 pods

### Monitoring Metrics

1. **Pod Load Distribution**: Check `podLoad=0,1,2...` in logs
2. **Request Distribution**: Monitor pod request counts
3. **Latency Variance**: Compare user category latencies

## 🔍 Advanced Tuning Options

### For High-Throughput Scenarios (>10 QPS)

```yaml
AIBRIX_ROUTER_VTC_TOKEN_TRACKER_WINDOW_SIZE: "1"
AIBRIX_ROUTER_VTC_TOKEN_TRACKER_TIME_UNIT: "seconds"
AIBRIX_ROUTER_VTC_BASIC_UTILIZATION_WEIGHT: "0.8"
```

### For Fairness-Critical Scenarios

```yaml
AIBRIX_ROUTER_VTC_BASIC_FAIRNESS_WEIGHT: "0.7"
AIBRIX_ROUTER_VTC_BASIC_UTILIZATION_WEIGHT: "0.3"
AIBRIX_ROUTER_VTC_TOKEN_TRACKER_MIN_TOKENS: "5"
```

### For Resource-Constrained Environments

```yaml
AIBRIX_ROUTER_VTC_TOKEN_TRACKER_WINDOW_SIZE: "5"
AIBRIX_ROUTER_VTC_TOKEN_TRACKER_TIME_UNIT: "seconds"
# Use alternative load metrics if available
```

## 🚨 Troubleshooting

### Issue: Still High Pod Concentration

**Solution**: Increase `UTILIZATION_WEIGHT` to 0.8-0.9

### Issue: Poor Fairness Between Users

**Solution**: Decrease `MIN_TOKENS` and increase `FAIRNESS_WEIGHT`

### Issue: Erratic Routing Behavior

**Solution**: Increase `WINDOW_SIZE` for more stable token tracking

### Issue: VTC Still Worse Than Random

**Potential Solutions**:

1. Use different load metrics (e.g., `gpu_cache_usage_perc`)
2. Implement request queuing awareness
3. Consider hybrid routing approach

## 📈 Validation Commands

### Check VTC Configuration Status

```bash
kubectl logs -n aibrix-system deployment/aibrix-gateway-plugins | grep -E "(VTC|bucket|token)" | tail -20
```

### Monitor Pod Load Distribution

```bash
kubectl logs -n aibrix-system deployment/aibrix-gateway-plugins --since=5m | grep "podLoad=" | grep -oE "podLoad=[0-9]+" | sort | uniq -c
```

### Verify Token Tracking

```bash
kubectl logs -n aibrix-system deployment/aibrix-gateway-plugins --since=5m | grep "UpdateTokenCount" | tail -10
```

## 🎯 Success Criteria

### Minimum Acceptable Performance

- VTC latency within 5% of random routing
- Pod concentration <70%
- All 3 pods receiving requests

### Optimal Performance

- VTC latency equal to or better than random
- Pod concentration <50%
- Even load distribution with fairness maintained
