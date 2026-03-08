# GD-Attention

Minimal public implementation of **GD-Attention** with a small quantitative comparison against softmax attention.

This repository is intended as a **research demo**, not as a production-ready training framework.  
It accompanies the public preprint and exposes the core mechanism in the simplest possible form:

- a semantic energy function
- coherence-point computation
- discrete key selection by minimum energy
- a small out-of-sample comparison on the Iris dataset

## What is GD-Attention?

GD-Attention replaces softmax-style weighted blending with an **energy-based selection mechanism**.

In this demo:

- **Softmax attention** computes scores, normalizes them, and returns a weighted combination.
- **GD-Attention** computes a semantic energy landscape, finds a coherence point for each query-key pair, and selects the key with the minimum energy.

This makes the mechanism structurally closer to **discrete semantic selection** than to probability-weighted averaging.

## Repository contents

```text
main.py
iris_comparison.py
outputs/
  fig1_gd_vs_softmax.png
  fig2_energy_landscape.png
  fig3_energy_slices.png
  fig4_toy_classification.png
  iris_quantitative_comparison.png
  iris_comparison_metrics.csv
README.md
```

## Files

### `main.py`
Core minimal implementation and figure generation.

It includes:

- semantic energy definition
- coherence point computation
- GD-Attention selection
- softmax baseline
- toy visualizations

Running this file produces the main qualitative figures.

### `iris_comparison.py`
Small quantitative comparison on the Iris dataset using **leave-one-out evaluation**.

It reports:

- classification accuracy
- selection consistency
- reference runtime per sample

This script is included only as a **minimal supplementary comparison**.  
It is not intended as a benchmark against optimized deep learning systems.

## How to run

### 1. Main demo

```bash
python main.py
```

This generates:

- `outputs/fig1_gd_vs_softmax.png`
- `outputs/fig2_energy_landscape.png`
- `outputs/fig3_energy_slices.png`
- `outputs/fig4_toy_classification.png`

### 2. Iris comparison

```bash
python iris_comparison.py
```

This generates:

- `outputs/iris_quantitative_comparison.png`
- `outputs/iris_comparison_metrics.csv`

## Minimal quantitative result

In the included leave-one-out Iris comparison, GD-Attention shows higher classification accuracy than the softmax baseline in this small fixed setting.

This result should be read carefully:

- it is a **small fixed comparison**
- it supports the claim that GD-Attention can behave differently from softmax in a discrete-selection setting
- it does **not** establish general superiority
- it is **not** evidence of faster runtime

## Positioning

This repository is best read as a public demonstration of the following point:

> GD-Attention is not primarily a speedup method.  
> It is an **energy-based semantic selection mechanism**.

Its potential value lies in settings where the important question is not only *how much weight is distributed*, but also *which candidate is structurally selected*.

## Status

- public research demo
- qualitative core figures included
- minimal quantitative comparison included
- not optimized for scale
- not a training library

## Citation

If you refer to this repository, please cite the associated preprint alongside the code release.
