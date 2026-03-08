from __future__ import annotations



from dataclasses import dataclass

from pathlib import Path

from typing import Iterable, List, Optional, Sequence, Tuple



import matplotlib.pyplot as plt

# When running in environments without an active display (e.g. automated
# evaluations on CI or servers without a GUI), matplotlib may attempt to
# communicate with external services for chart logging.  To avoid unexpected
# network calls and ensure deterministic output, explicitly switch to the
# non-interactive "Agg" backend.  This must be done before any figures are
# created.
plt.switch_backend("Agg")

import numpy as np





EPS = 1e-12





@dataclass

class GDResult:

    selected_index: int

    selected_value: np.ndarray

    selected_key: np.ndarray

    coherence_points: np.ndarray

    energies: np.ndarray





@dataclass

class SoftmaxResult:

    output: np.ndarray

    weights: np.ndarray

    scores: np.ndarray





def _as_array(x: Sequence[float] | np.ndarray) -> np.ndarray:

    arr = np.asarray(x, dtype=float)

    if arr.ndim != 1:

        raise ValueError("Expected a 1D vector.")

    return arr





def _as_matrix(x: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:

    arr = np.asarray(x, dtype=float)

    if arr.ndim != 2:

        raise ValueError("Expected a 2D array.")

    return arr





def semantic_energy(

    s: np.ndarray,

    mu1: np.ndarray,

    mu2: np.ndarray,

    alpha: float = 0.5,

) -> np.ndarray:

    """

    Equation (1): phi(s) = -log(alpha * exp(-||s-mu1||^2) + (1-alpha) * exp(-||s-mu2||^2)).



    Works for a single point (..., d) or a batch of points (..., d).

    """

    if not (0.0 < alpha < 1.0):

        raise ValueError("alpha must be in (0, 1).")



    s = np.asarray(s, dtype=float)

    mu1 = _as_array(mu1)

    mu2 = _as_array(mu2)



    d1 = np.sum((s - mu1) ** 2, axis=-1)

    d2 = np.sum((s - mu2) ** 2, axis=-1)

    mix = alpha * np.exp(-d1) + (1.0 - alpha) * np.exp(-d2)

    return -np.log(np.maximum(mix, EPS))





def phi_g(x: np.ndarray, L: float, alpha: float = 0.5) -> np.ndarray:

    x = np.asarray(x, dtype=float)

    mix = alpha * np.exp(-(x**2)) + (1.0 - alpha) * np.exp(-((x - L) ** 2))

    return -np.log(np.maximum(mix, EPS))





def _a_prime(x: float, L: float, alpha: float) -> float:

    return (

        alpha * (-2.0 * x) * np.exp(-(x**2))

        + (1.0 - alpha) * (-2.0 * (x - L)) * np.exp(-((x - L) ** 2))

    )





def _find_x_star(L: float, alpha: float = 0.5, tol: float = 1e-10, max_iter: int = 200) -> float:

    """

    Finds the unique minimiser x* in (0, L) for phi_g using bisection on a'(x).

    The paper proves the root exists and is unique when L > 0 and 0 < alpha < 1.

    """

    if L <= 1e-14:

        return 0.0



    left = 0.0

    right = float(L)

    f_left = _a_prime(left, L, alpha)

    f_right = _a_prime(right, L, alpha)



    if not (f_left > 0.0 and f_right < 0.0):

        grid = np.linspace(0.0, L, 2001)

        values = np.array([_a_prime(float(t), L, alpha) for t in grid])

        sign_change = np.where(np.signbit(values[:-1]) != np.signbit(values[1:]))[0]

        if len(sign_change) == 0:

            return float(L / 2.0)

        idx = int(sign_change[0])

        left = float(grid[idx])

        right = float(grid[idx + 1])



    for _ in range(max_iter):

        mid = 0.5 * (left + right)

        f_mid = _a_prime(mid, L, alpha)

        if abs(f_mid) < tol or abs(right - left) < tol:

            return mid

        if f_mid > 0.0:

            left = mid

        else:

            right = mid

    return 0.5 * (left + right)





def coherence_point(

    query: Sequence[float] | np.ndarray,

    key: Sequence[float] | np.ndarray,

    alpha: float = 0.5,

) -> Tuple[np.ndarray, float]:

    """

    Returns the unique coherence point s* and its minimal energy for one query-key pair.

    """

    q = _as_array(query)

    k = _as_array(key)



    delta = k - q

    L = float(np.linalg.norm(delta))

    if L <= 1e-14:

        s_star = q.copy()

        return s_star, float(semantic_energy(s_star, q, k, alpha=alpha))



    g = delta / L

    x_star = _find_x_star(L=L, alpha=alpha)

    s_star = q + x_star * g

    e_star = float(phi_g(np.array([x_star]), L=L, alpha=alpha)[0])

    return s_star, e_star





def gd_attention(

    query: Sequence[float] | np.ndarray,

    keys: Sequence[Sequence[float]] | np.ndarray,

    values: Optional[Sequence[Sequence[float]] | np.ndarray] = None,

    alpha: float = 0.5,

) -> GDResult:

    q = _as_array(query)

    K = _as_matrix(keys)

    V = K.copy() if values is None else _as_matrix(values)



    if K.shape != V.shape:

        raise ValueError("keys and values must have the same shape for this minimal demo.")



    coherence_points: List[np.ndarray] = []

    energies: List[float] = []

    for key in K:

        s_star, e_star = coherence_point(q, key, alpha=alpha)

        coherence_points.append(s_star)

        energies.append(e_star)



    coherence_points_arr = np.vstack(coherence_points)

    energies_arr = np.asarray(energies, dtype=float)

    idx = int(np.argmin(energies_arr))



    return GDResult(

        selected_index=idx,

        selected_value=V[idx],

        selected_key=K[idx],

        coherence_points=coherence_points_arr,

        energies=energies_arr,

    )





def softmax_attention(

    query: Sequence[float] | np.ndarray,

    keys: Sequence[Sequence[float]] | np.ndarray,

    values: Optional[Sequence[Sequence[float]] | np.ndarray] = None,

    temperature: float = 1.0,

) -> SoftmaxResult:

    q = _as_array(query)

    K = _as_matrix(keys)

    V = K.copy() if values is None else _as_matrix(values)



    if temperature <= 0.0:

        raise ValueError("temperature must be positive.")

    if K.shape != V.shape:

        raise ValueError("keys and values must have the same shape for this minimal demo.")



    scores = (K @ q) / temperature

    scores = scores - np.max(scores)

    weights = np.exp(scores)

    weights /= np.maximum(np.sum(weights), EPS)

    output = weights @ V



    return SoftmaxResult(output=output, weights=weights, scores=scores)





def _setup_axes(ax: plt.Axes, title: str, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> None:

    ax.set_title(title)

    ax.set_xlim(*xlim)

    ax.set_ylim(*ylim)

    ax.set_aspect("equal", adjustable="box")

    ax.grid(True, alpha=0.25)





def plot_qualitative_comparison(save_path: Optional[Path] = None) -> None:

    labels = ["RedTruck", "BlueBird", "GreenLeaf", "YellowSun", "PurpleWave"]

    query = np.array([0.2, 0.0])

    keys = np.array(

        [

            [1.8, 0.1],

            [0.3, 1.7],

            [-1.5, 0.8],

            [-1.2, -1.1],

            [0.7, -1.8],

        ]

    )

    values = keys.copy()



    gd = gd_attention(query, keys, values, alpha=0.5)

    sm = softmax_attention(query, keys, values, temperature=0.9)



    fig, ax = plt.subplots(figsize=(8, 6))

    _setup_axes(ax, "GD-Attention vs Softmax (toy qualitative comparison)", (-2.3, 2.3), (-2.3, 2.3))



    for key, label in zip(keys, labels):

        ax.scatter(key[0], key[1], s=110)

        ax.text(key[0] + 0.06, key[1] + 0.06, label, fontsize=9)

        ax.plot([query[0], key[0]], [query[1], key[1]], linestyle=":", linewidth=1.1, alpha=0.7)



    for s_star in gd.coherence_points:

        ax.scatter(s_star[0], s_star[1], marker="D", s=70)

        ax.plot([query[0], s_star[0]], [query[1], s_star[1]], linewidth=1.5, alpha=0.9)



    ax.scatter(query[0], query[1], marker="*", s=250, label="Query")

    ax.scatter(sm.output[0], sm.output[1], marker="^", s=180, label="Softmax output")

    selected = gd.selected_key

    ax.scatter(selected[0], selected[1], marker="o", s=260, facecolors="none", linewidths=2.2, label="GD selected key")

    ax.legend(loc="upper left")



    if save_path is not None:

        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close(fig)





def plot_energy_landscape(save_path: Optional[Path] = None) -> None:

    mu1 = np.array([-0.75, 0.0])

    mu2 = np.array([0.75, 0.0])

    alpha = 0.5



    xs = np.linspace(-2.0, 2.0, 300)

    ys = np.linspace(-2.0, 2.0, 300)

    XX, YY = np.meshgrid(xs, ys)

    grid = np.stack([XX, YY], axis=-1)

    Z = semantic_energy(grid, mu1, mu2, alpha=alpha)



    fig, ax = plt.subplots(figsize=(7.4, 6))

    contour = ax.contourf(XX, YY, Z, levels=30)

    fig.colorbar(contour, ax=ax, shrink=0.84, label="Semantic energy")

    _setup_axes(ax, "Semantic energy landscape", (-2.0, 2.0), (-2.0, 2.0))

    ax.scatter(mu1[0], mu1[1], s=120, label="mu1")

    ax.scatter(mu2[0], mu2[1], s=120, marker="s", label="mu2")

    ax.legend(loc="upper right")



    if save_path is not None:

        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close(fig)





def plot_energy_slices(save_path: Optional[Path] = None) -> None:

    mu1 = np.array([-0.75, 0.0])

    mu2 = np.array([0.75, 0.0])

    alpha = 0.5



    L = float(np.linalg.norm(mu2 - mu1))

    x_star = _find_x_star(L=L, alpha=alpha)

    phi_star = float(phi_g(np.array([x_star]), L=L, alpha=alpha)[0])



    x_grid = np.linspace(-0.5, L + 0.5, 500)

    phi_parallel = phi_g(x_grid, L=L, alpha=alpha)



    t_grid = np.linspace(-1.0, 1.0, 400)

    phi_orth = phi_star + t_grid**2



    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))



    ax1.plot(x_grid, phi_parallel, linewidth=2)

    ax1.scatter([x_star], [phi_star], marker="D", s=90)

    ax1.set_title("Slice along jump direction")

    ax1.set_xlabel("x")

    ax1.set_ylabel("phi_g(x)")

    ax1.grid(True, alpha=0.25)



    ax2.plot(t_grid, phi_orth, linewidth=2, label="Exact orthogonal slice")

    ax2.plot(t_grid, phi_star + t_grid**2, linestyle="--", linewidth=1.7, label="t^2 + const")

    ax2.scatter([0.0], [phi_star], marker="D", s=90)

    ax2.set_title("Slice orthogonal to jump direction")

    ax2.set_xlabel("t")

    ax2.set_ylabel("phi(s* + t v)")

    ax2.grid(True, alpha=0.25)

    ax2.legend(loc="upper center")



    if save_path is not None:

        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close(fig)





def plot_classification_example(save_path: Optional[Path] = None) -> None:

    rng = np.random.default_rng(7)



    red_center = np.array([1.3, 0.8])

    blue_center = np.array([-1.1, 0.9])

    green_center = np.array([0.0, -1.2])



    red = red_center + 0.18 * rng.normal(size=(4, 2))

    blue = blue_center + 0.18 * rng.normal(size=(4, 2))

    green = green_center + 0.18 * rng.normal(size=(4, 2))



    keys = np.vstack([red, blue, green])

    values = keys.copy()

    labels = ["red"] * len(red) + ["blue"] * len(blue) + ["green"] * len(green)

    query = np.array([1.02, 0.62])



    gd = gd_attention(query, keys, values, alpha=0.5)

    sm = softmax_attention(query, keys, values, temperature=0.8)



    fig, ax = plt.subplots(figsize=(8, 6))

    _setup_axes(ax, "Toy classification: Softmax vs GD-Attention", (-2.0, 2.1), (-2.0, 1.8))



    color_map = {"red": "tab:red", "blue": "tab:blue", "green": "tab:green"}

    for point, label in zip(keys, labels):

        ax.scatter(point[0], point[1], s=90, color=color_map[label], alpha=0.9)



    for s_star in gd.coherence_points:

        ax.scatter(s_star[0], s_star[1], marker="D", s=55, color="tab:olive", alpha=0.85)

        ax.plot([query[0], s_star[0]], [query[1], s_star[1]], linewidth=1.0, color="tab:olive", alpha=0.45)



    ax.scatter(query[0], query[1], marker="*", s=260, color="orange", label="Query")

    ax.scatter(sm.output[0], sm.output[1], marker="^", s=170, color="purple", label="Softmax output")

    ax.scatter(

        gd.selected_key[0],

        gd.selected_key[1],

        marker="o",

        s=240,

        facecolors="none",

        edgecolors="black",

        linewidths=2.0,

        label="GD selected key",

    )

    ax.legend(loc="lower left")



    if save_path is not None:

        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close(fig)


# --------------------------------------------------------------------------
# Quantitative comparison utilities
#
# The following helper functions generate a small synthetic dataset (similar to
# the toy classification example above), evaluate GD-Attention and standard
# softmax attention on a batch of queries, compute simple metrics, and
# visualise the results.  These additions are deliberately lightweight to
# preserve the spirit of the original demo: they rely only on NumPy and
# Matplotlib and avoid any heavy dependencies or training loops.  The goal is
# to provide a minimal yet persuasive quantitative comparison between the
# two attention mechanisms.

def _generate_toy_dataset(
    n_queries_per_class: int = 20,
    noise_std: float = 0.18,
    seed: int = 7,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, List[str]]:
    """
    Create a tiny synthetic dataset of three clusters (red, blue and green)
    together with a set of query points.  The keys/values are identical
    points drawn around fixed cluster centres using a small amount of Gaussian
    noise.  Queries are drawn from the same distributions but in a larger
    number to allow meaningful accuracy estimates.

    Parameters
    ----------
    n_queries_per_class: int
        Number of query points to sample around each cluster centre.  The
        total number of queries returned will be 3 * n_queries_per_class.
    noise_std: float
        Standard deviation of the isotropic Gaussian noise applied when
        sampling keys/values and queries.
    seed: int
        Seed for the random number generator to ensure reproducibility.

    Returns
    -------
    keys: np.ndarray, shape (n_keys, 2)
        The key/value vectors used by the attention mechanisms.  There are
        exactly 12 points: 4 per class as in the toy classification example.
    values: np.ndarray, shape (n_keys, 2)
        Identical to ``keys``; returned separately for API compatibility.
    key_labels: list of str
        Class labels for each key/value.
    queries: np.ndarray, shape (n_queries, 2)
        Query points to evaluate.  These are sampled in larger numbers from
        the same cluster distributions.
    query_labels: list of str
        Ground truth class labels for each query.
    """
    rng = np.random.default_rng(seed)
    # Define the three cluster centres as in the classification example.
    red_center = np.array([1.3, 0.8])
    blue_center = np.array([-1.1, 0.9])
    green_center = np.array([0.0, -1.2])

    # Sample a small fixed set of keys/values (4 per class) for attention.  This
    # mirrors the classification example exactly to remain consistent with
    # previously published visuals.
    red = red_center + noise_std * rng.normal(size=(4, 2))
    blue = blue_center + noise_std * rng.normal(size=(4, 2))
    green = green_center + noise_std * rng.normal(size=(4, 2))
    keys = np.vstack([red, blue, green])
    values = keys.copy()
    key_labels: List[str] = ["red"] * len(red) + ["blue"] * len(blue) + ["green"] * len(green)

    # Now sample a larger set of queries around each centre.  These queries
    # simulate unseen examples one might wish to classify using the keys.
    queries = []
    query_labels: List[str] = []
    for centre, label in zip(
        [red_center, blue_center, green_center], ["red", "blue", "green"]
    ):
        pts = centre + noise_std * rng.normal(size=(n_queries_per_class, 2))
        queries.append(pts)
        query_labels += [label] * n_queries_per_class
    queries_arr = np.vstack(queries)

    return keys, values, key_labels, queries_arr, query_labels


def _evaluate_attention_methods(
    keys: np.ndarray,
    values: np.ndarray,
    key_labels: Sequence[str],
    queries: np.ndarray,
    query_labels: Sequence[str],
    alpha: float = 0.5,
    temperature: float = 0.9,
) -> dict:
    """
    Compute simple quantitative metrics comparing GD-Attention to softmax
    attention on a batch of queries.

    For each query, predictions are obtained by selecting the label of the
    key with the largest softmax weight (standard dot-product attention) or
    minimal semantic energy (GD-Attention).  The ground-truth label for
    each query is provided.  A selection consistency metric is also
    computed that counts how often the two methods pick the exact same key.
    Finally, per-query runtimes are measured to provide a rough sense of
    computational overhead.

    Parameters
    ----------
    keys, values: np.ndarray
        The key/value arrays to be passed to the attention functions.  They
        must have the same shape.
    key_labels: Sequence[str]
        Class labels corresponding to each key/value.  The length must match
        the first dimension of ``keys``.
    queries: np.ndarray
        Query vectors on which to evaluate the attention mechanisms.
    query_labels: Sequence[str]
        Ground truth labels for each query.  Length must equal the number of
        queries.
    alpha: float
        The mixture weight used by GD-Attention.
    temperature: float
        Temperature parameter for softmax attention.  A non-unit temperature
        can slightly adjust the sharpness of the softmax distribution; we
        choose 0.9 by default to mirror the qualitative demo.

    Returns
    -------
    metrics: dict
        Dictionary containing the following keys:

        - ``accuracy_gd``: classification accuracy of GD-Attention.
        - ``accuracy_softmax``: classification accuracy of softmax attention.
        - ``selection_consistency``: fraction of queries for which both
          methods select the same key.
        - ``avg_runtime_gd_ms``: average runtime per query in milliseconds
          for GD-Attention.
        - ``avg_runtime_softmax_ms``: average runtime per query in
          milliseconds for softmax attention.
    """
    if len(keys) != len(key_labels):
        raise ValueError("Length of key_labels must match number of keys.")
    if len(queries) != len(query_labels):
        raise ValueError(
            "Length of query_labels must match number of queries."
        )

    n_queries = len(queries)
    correct_gd = 0
    correct_soft = 0
    same_selection = 0
    runtime_gd = 0.0
    runtime_soft = 0.0

    # Convert key_labels and query_labels to numpy arrays for efficient
    # indexing/comparison.  We'll store labels as strings.
    key_labels_arr = np.asarray(key_labels)
    query_labels_arr = np.asarray(query_labels)

    import time  # local import to avoid polluting global namespace
    for i in range(n_queries):
        q = queries[i]
        # Softmax attention
        t0 = time.perf_counter()
        sm_res = softmax_attention(q, keys, values, temperature=temperature)
        t1 = time.perf_counter()
        runtime_soft += t1 - t0
        # Predicted label: label of key with largest weight
        idx_soft = int(np.argmax(sm_res.weights))
        pred_soft = key_labels_arr[idx_soft]
        if pred_soft == query_labels_arr[i]:
            correct_soft += 1
        # GD-Attention
        t0 = time.perf_counter()
        gd_res = gd_attention(q, keys, values, alpha=alpha)
        t1 = time.perf_counter()
        runtime_gd += t1 - t0
        idx_gd = int(gd_res.selected_index)
        pred_gd = key_labels_arr[idx_gd]
        if pred_gd == query_labels_arr[i]:
            correct_gd += 1
        # Consistency
        if idx_gd == idx_soft:
            same_selection += 1
    accuracy_gd = correct_gd / n_queries
    accuracy_soft = correct_soft / n_queries
    selection_consistency = same_selection / n_queries
    # Convert runtimes to milliseconds
    avg_runtime_gd_ms = (runtime_gd / n_queries) * 1000.0
    avg_runtime_soft_ms = (runtime_soft / n_queries) * 1000.0
    return {
        "accuracy_gd": accuracy_gd,
        "accuracy_softmax": accuracy_soft,
        "selection_consistency": selection_consistency,
        "avg_runtime_gd_ms": avg_runtime_gd_ms,
        "avg_runtime_softmax_ms": avg_runtime_soft_ms,
    }


def plot_quantitative_comparison(
    save_path: Optional[Path] = None,
    n_queries_per_class: int = 20,
    noise_std: float = 0.18,
    alpha: float = 0.5,
    temperature: float = 0.9,
) -> dict:
    """
    Generate a small synthetic dataset, evaluate GD-Attention and softmax
    attention on a batch of queries, compute simple metrics and produce a
    bar chart summarising the results.

    The resulting dictionary of metrics is returned and, if ``save_path`` is
    provided, the chart is also saved to disk.  The bar chart contains two
    grouped bars: one showing classification accuracy and another showing
    average per-query runtime.  Selection consistency is annotated atop
    the figure for context.

    Parameters
    ----------
    save_path: Optional[Path]
        If provided, the figure will be written to this location (its
        parent directory will be created if necessary).
    n_queries_per_class: int
        Number of query points to sample per class (default 20).  The total
        number of queries evaluated is 3 * n_queries_per_class.
    noise_std: float
        Standard deviation of the noise applied to both the keys and queries.
    alpha: float
        Mixture weight used by GD-Attention.
    temperature: float
        Temperature for the softmax attention.

    Returns
    -------
    metrics: dict
        Same dictionary as returned by ``_evaluate_attention_methods``.
    """
    # Generate dataset and evaluate metrics
    keys, values, key_labels, queries, query_labels = _generate_toy_dataset(
        n_queries_per_class=n_queries_per_class,
        noise_std=noise_std,
    )
    metrics = _evaluate_attention_methods(
        keys,
        values,
        key_labels,
        queries,
        query_labels,
        alpha=alpha,
        temperature=temperature,
    )

    # Prepare values for plotting
    accuracies = [metrics["accuracy_softmax"], metrics["accuracy_gd"]]
    runtimes = [
        metrics["avg_runtime_softmax_ms"],
        metrics["avg_runtime_gd_ms"],
    ]
    labels = ["Softmax", "GD-Attn"]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    # Left subplot: classification accuracy
    ax[0].bar(labels, accuracies, color=["tab:purple", "tab:orange"], alpha=0.8)
    ax[0].set_ylim(0.0, 1.05)
    ax[0].set_ylabel("Accuracy")
    ax[0].set_title("Classification accuracy")
    for i, v in enumerate(accuracies):
        ax[0].text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=10)
    ax[0].grid(True, axis="y", alpha=0.2)
    # Right subplot: average runtime
    ax[1].bar(labels, runtimes, color=["tab:purple", "tab:orange"], alpha=0.8)
    ax[1].set_ylabel("Avg runtime per query (ms)")
    ax[1].set_title("Computation cost")
    for i, v in enumerate(runtimes):
        ax[1].text(i, v + max(runtimes) * 0.05, f"{v:.3f}", ha="center", fontsize=10)
    ax[1].grid(True, axis="y", alpha=0.2)
    # Annotate selection consistency below figure
    fig.suptitle(
        f"Selection consistency: {metrics['selection_consistency']:.2f}",
        fontsize=11,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.0, 1.0, 0.95])

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return metrics





def run_all(output_dir: str = "outputs") -> None:

    out = Path(output_dir)

    plot_qualitative_comparison(out / "fig1_gd_vs_softmax.png")

    plot_energy_landscape(out / "fig2_energy_landscape.png")

    plot_energy_slices(out / "fig3_energy_slices.png")

    plot_classification_example(out / "fig4_toy_classification.png")

    # ----------------------------------------------------------------------
    # Quantitative comparison
    #
    # Generate a small synthetic evaluation and produce both a CSV table and
    # a bar-chart figure summarising the results.  These files can be found
    # alongside the other output figures.  If not needed, this section can
    # simply be ignored; it does not affect any other part of the demo.
    metrics = plot_quantitative_comparison(
        save_path=out / "fig5_quantitative_comparison.png",
        n_queries_per_class=20,
        noise_std=0.18,
        alpha=0.5,
        temperature=0.9,
    )
    # Write a simple CSV table capturing the key metrics.  The CSV contains
    # three rows: accuracy, average runtime per query (ms) and selection
    # consistency.  Each row lists the values for softmax and GD-Attention.
    table_path = out / "comparison_metrics.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", encoding="utf-8") as f:
        # header row
        f.write("metric,softmax,gd_attention\n")
        # accuracy row
        f.write(
            f"accuracy,{metrics['accuracy_softmax']:.4f},{metrics['accuracy_gd']:.4f}\n"
        )
        # runtime row
        f.write(
            f"avg_runtime_ms,{metrics['avg_runtime_softmax_ms']:.6f},{metrics['avg_runtime_gd_ms']:.6f}\n"
        )
        # selection consistency row (same value for both columns since it
        # measures agreement between the two methods)
        f.write(
            f"selection_consistency,{metrics['selection_consistency']:.4f},{metrics['selection_consistency']:.4f}\n"
        )





if __name__ == "__main__":

    run_all()

    print("Saved demo figures to ./outputs")

