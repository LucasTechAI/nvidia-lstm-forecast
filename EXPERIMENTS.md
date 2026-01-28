# Experiments Documentation

This document catalogs the main experiments, findings, and best practices discovered during the development of the NVIDIA LSTM forecasting system.

## Overview

This project uses MLflow to track all experiments systematically. Each experiment tests different configurations to find the optimal model for NVIDIA stock price prediction.

## Experiment Categories

### 1. Baseline Models

**Objective**: Establish baseline performance metrics

#### Experiment 1.1: Simple LSTM
- **Configuration**:
  - Layers: 1
  - Hidden Size: 64
  - Sequence Length: 30
  - Learning Rate: 0.001
- **Results**: To be documented after running
- **Key Findings**: To be documented

#### Experiment 1.2: Deeper LSTM
- **Configuration**:
  - Layers: 2
  - Hidden Size: 128
  - Sequence Length: 60
  - Learning Rate: 0.001
- **Results**: To be documented
- **Key Findings**: To be documented

### 2. Hyperparameter Optimization

**Objective**: Find optimal hyperparameters using Optuna

#### Experiment 2.1: Initial HPO (20 trials)
- **Search Space**:
  - num_layers: [1, 2, 3, 4]
  - hidden_size: [32, 64, 128, 256]
  - learning_rate: [1e-5, 1e-2] (log scale)
  - dropout: [0.1, 0.5]
  - sequence_length: [30, 60, 90, 120]
  - batch_size: [16, 32, 64, 128]
- **Best Parameters**: To be documented
- **Best Validation RMSE**: To be documented
- **Key Findings**: To be documented

#### Experiment 2.2: Extended HPO (50-100 trials)
- **Purpose**: Refine search with more trials
- **Results**: To be documented
- **Key Findings**: To be documented

### 3. Feature Engineering

**Objective**: Test different feature combinations

#### Experiment 3.1: Single Feature (Close Price Only)
- **Features**: Close
- **Rationale**: Baseline - simplest approach
- **Results**: To be documented
- **Pros**: Simple, less noise
- **Cons**: Limited information

#### Experiment 3.2: Multi-Feature (OHLCV)
- **Features**: Open, High, Low, Close, Volume
- **Rationale**: More market context
- **Results**: To be documented
- **Pros**: Richer information
- **Cons**: More complexity, potential noise

#### Experiment 3.3: Technical Indicators
- **Features**: Close + RSI, MACD, Bollinger Bands, Moving Averages
- **Status**: Future work
- **Hypothesis**: Technical indicators may improve predictions

### 4. Architecture Variations

**Objective**: Test different LSTM architectures

#### Experiment 4.1: Bidirectional LSTM
- **Configuration**: LSTM with bidirectional=True
- **Rationale**: Capture patterns in both directions
- **Results**: To be documented
- **Trade-offs**: 2x parameters, longer training time

#### Experiment 4.2: Stacked LSTM
- **Configuration**: 3-4 LSTM layers
- **Rationale**: Deeper feature extraction
- **Results**: To be documented
- **Trade-offs**: More parameters, risk of overfitting

#### Experiment 4.3: LSTM with Attention
- **Status**: Future work
- **Rationale**: Focus on important time steps

### 5. Sequence Length Analysis

**Objective**: Determine optimal lookback window

| Sequence Length | Training Time | Val RMSE | Notes |
|----------------|---------------|----------|-------|
| 30 days | TBD | TBD | Short-term patterns |
| 60 days | TBD | TBD | Medium-term patterns |
| 90 days | TBD | TBD | Long-term patterns |
| 120 days | TBD | TBD | Extended context |

**Key Findings**: To be documented

### 6. Regularization Techniques

**Objective**: Prevent overfitting

#### Experiment 6.1: Dropout Variations
- **Tested**: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
- **Best**: To be documented
- **Findings**: To be documented

#### Experiment 6.2: Early Stopping
- **Patience Values**: 5, 10, 15, 20
- **Best**: To be documented
- **Findings**: To be documented

#### Experiment 6.3: L2 Regularization
- **Status**: Future work
- **Weight Decay**: [1e-5, 1e-3]

### 7. Training Strategies

**Objective**: Optimize training process

#### Experiment 7.1: Learning Rate Schedules
- **Strategies**:
  - Fixed: 0.001
  - Step decay
  - Exponential decay
  - Cosine annealing
- **Results**: To be documented

#### Experiment 7.2: Batch Size Impact
- **Tested**: 16, 32, 64, 128
- **Trade-offs**: Speed vs. generalization
- **Results**: To be documented

#### Experiment 7.3: Optimizer Comparison
- **Tested**: Adam, AdamW, SGD with momentum
- **Best**: To be documented

### 8. Ensemble Methods

**Objective**: Combine multiple models for better predictions

#### Experiment 8.1: Top-K Model Averaging
- **Status**: Future work
- **Approach**: Average predictions from top 3-5 models
- **Expected Benefit**: Reduced variance

#### Experiment 8.2: Weighted Ensemble
- **Status**: Future work
- **Approach**: Weight models by validation performance

## Best Practices Discovered

### Data Preprocessing
1. **Normalization**: MinMaxScaler (0-1) works well for financial data
2. **Sequence Creation**: Maintain temporal order, no shuffling
3. **Train/Val/Test Split**: 70/15/15 provides good balance
4. **Data Filtering**: Using data from 2017+ reduces noise from different market regimes

### Model Architecture
1. **Hidden Size**: 128-256 units provide good capacity without overfitting
2. **Num Layers**: 2-3 layers balance depth and training stability
3. **Dropout**: 0.2-0.3 helps prevent overfitting
4. **Bidirectional**: To be determined based on experiments

### Training
1. **Early Stopping**: Patience of 10 epochs prevents overfitting
2. **Gradient Clipping**: Max norm of 1.0 stabilizes training
3. **Batch Size**: 32-64 provides good training speed and stability
4. **Learning Rate**: 0.001 (Adam) is a good starting point

### MLflow Organization
1. **Experiment Naming**: Use descriptive names (e.g., "nvidia-lstm-baseline")
2. **Run Naming**: Include key parameters (e.g., "h128_l2_seq60")
3. **Tags**: Add tags for experiment category (baseline, hpo, feature-eng)
4. **Artifacts**: Always log model, scaler, and plots

## Performance Benchmarks

| Model Type | Val RMSE | Test RMSE | MAE | MAPE | Training Time |
|-----------|----------|-----------|-----|------|---------------|
| Baseline (Simple) | TBD | TBD | TBD | TBD | TBD |
| Baseline (Deep) | TBD | TBD | TBD | TBD | TBD |
| HPO Best | TBD | TBD | TBD | TBD | TBD |
| Ensemble | TBD | TBD | TBD | TBD | TBD |

## Lessons Learned

### What Worked Well
- To be documented after experiments

### What Didn't Work
- To be documented after experiments

### Unexpected Findings
- To be documented after experiments

## Future Directions

### Short-term
1. Complete baseline experiments
2. Run comprehensive HPO (100+ trials)
3. Test different feature combinations
4. Implement walk-forward validation

### Medium-term
1. Add technical indicators as features
2. Implement attention mechanism
3. Test ensemble methods
4. Add prediction intervals

### Long-term
1. Multi-step ahead predictions (beyond 30 days)
2. Real-time prediction API
3. Portfolio optimization using predictions
4. Transfer learning to other stocks

## Experiment Template

When running new experiments, document using this template:

```
### Experiment X.Y: [Name]

**Date**: YYYY-MM-DD
**Objective**: [What you're testing]
**Hypothesis**: [Expected outcome]

**Configuration**:
- Parameter 1: value
- Parameter 2: value
- ...

**Results**:
- Metric 1: value
- Metric 2: value
- MLflow Run ID: [run_id]

**Key Findings**:
1. Finding 1
2. Finding 2

**Next Steps**:
- Action 1
- Action 2
```

## Reproducibility

All experiments can be reproduced using:
1. MLflow run ID for exact parameters
2. Saved model artifacts
3. Optuna study database
4. Git commit hash for code version

## Notes

- Always run multiple seeds for statistical significance
- Compare against naive baselines (last value, moving average)
- Monitor for data leakage in preprocessing
- Validate on out-of-sample test set only once

---

**Last Updated**: 2026-01-28
**Maintained by**: LucasTechAI
