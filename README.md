# GD-Attention: Iris comparison artifacts

This repository snapshot contains **result artifacts only** for a small Iris comparison between a softmax baseline and GD-Attention.

It currently includes:

```text
README.md
iris_quantitative_comparison.png
iris_comparison_metrics.csv
```

## Included files

### `iris_quantitative_comparison.png`
A summary figure for the leave-one-out Iris comparison.

It reports:
- classification accuracy
- average runtime per sample (ms)
- selection consistency

### `iris_comparison_metrics.csv`
A CSV file containing the numeric values shown in the figure.

## Reported values

From `iris_comparison_metrics.csv`:

- **Accuracy**: Softmax `0.7733`, GD-Attention `0.9467`
- **Average runtime per sample**: Softmax `0.059445 ms`, GD-Attention `5.113311 ms`
- **Selection consistency**: Softmax `0.0600`, GD-Attention `0.0600`

## Interpretation

This snapshot supports only the following narrow statement:

> In this small leave-one-out Iris comparison, GD-Attention achieved higher accuracy than the softmax baseline, while requiring substantially more runtime per sample.

It does **not** support a general claim of runtime superiority.
It also does **not** provide a broad benchmark across datasets, model scales, or optimized implementations.

## Scope

This repository snapshot is best read as a **small result bundle** for a toy comparison, not as a full implementation release.

In particular:
- source code is **not included** in this snapshot
- no training library is included
- no claim is made here about large-scale performance

## Note on selection consistency

The current artifact reports selection consistency as `0.0600` for both methods.
Any interpretation of that metric should therefore be made cautiously and should be tied to the exact evaluation definition used in the underlying experiment.
