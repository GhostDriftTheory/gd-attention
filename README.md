# GD-Attention Minimal Demo

Minimal public demo of **GD-Attention** with a small **Iris leave-one-out reference comparison**.

This repository is a **research/demo implementation**, not a training library and not an optimized benchmark package.
Its purpose is to expose the core mechanism in a small reproducible form and to show, in one fixed setting, how GD-Attention behaves differently from a softmax baseline.

## What is included

```text
README.md
main.py
iris_comparison.py
outputs/
  iris_quantitative_comparison.png
  iris_comparison_metrics.csv
```

Only the files above are included in this minimal public version.

## What `main.py` does

`main.py` contains the core minimal implementation:

- semantic energy function
- coherence-point computation
- GD-Attention key selection by minimum energy
- softmax baseline
- toy plotting utilities

It also contains optional demo routines for:

- qualitative query/key comparison
- semantic energy landscape visualization
- energy-slice visualization
- toy classification example
- a small synthetic quantitative comparison

If you run:

```bash
python main.py
```

it will generate additional files under `outputs/`, including toy figures and a synthetic comparison table. Those generated files are **not required** for the present minimal repository listing above.

## What `iris_comparison.py` does

`iris_comparison.py` provides the small reference comparison included in this repo.

It:

1. loads the classic Iris dataset from scikit-learn,
2. standardizes the features,
3. runs a **leave-one-out** evaluation,
4. compares GD-Attention with a softmax baseline,
5. writes the committed output files:
   - `outputs/iris_quantitative_comparison.png`
   - `outputs/iris_comparison_metrics.csv`

If you run:

```bash
python iris_comparison.py
```

the script predicts each held-out sample as follows:

- **Softmax baseline**: select the label of the key with the largest softmax weight.
- **GD-Attention**: select the label of the key with the minimum semantic energy.

It reports three simple metrics:

- **classification accuracy**
- **selection consistency**
- **average runtime per sample (ms)**

Here, **selection consistency** means the fraction of evaluation samples for which GD-Attention and the softmax baseline selected the **same key index**.

## Included reference result

The committed Iris output pair corresponds to the following fixed reference result:

- Softmax accuracy: **0.7733**
- GD-Attention accuracy: **0.9467**
- Softmax average runtime: **0.059445 ms/sample**
- GD-Attention average runtime: **5.113311 ms/sample**
- Selection consistency: **0.0600**

## How to read this result

This result should be read narrowly.

It shows that, in this small fixed leave-one-out Iris setting, GD-Attention:

- selects very differently from the softmax baseline,
- reaches higher accuracy in this specific reference comparison,
- is much slower in this unoptimized implementation.

It does **not** establish general superiority, training-time advantage, or runtime advantage over optimized attention implementations.

## Requirements

```bash
pip install numpy matplotlib scikit-learn
```

## Positioning

This repository is best read as a compact public demonstration of the following point:

> GD-Attention is an **energy-based semantic selection mechanism**.
> It should not be read primarily as a speedup claim.

## Status

- minimal public demo
- core implementation included
- Iris reference comparison included
- not optimized for scale
- not a production training framework
