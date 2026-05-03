"""Per-sample subset-selection strategies for the SNR search.

PROPOSALS.md (under ``results/smooth_subtasks/per_sample/``) lists five
options for picking the subset of samples that maximises SNR for a
single (task, size) pair. This module implements them on top of the
primitives that already live in ``smooth_subtasks_per_sample`` so the
runner can compare them side by side without duplicating the eval-log
parsing or the (n_ckpts × n_samples) matrix-builder.

Each strategy has the same signature::

    select_X(A, mix_rows, doc_ids, rng=None, **opts) -> StrategyResult

where:

* ``A``         — (n_ckpts, n_samples) acc matrix, no NaNs.
* ``mix_rows``  — {mix: list[row indices]}, last-N-ckpt rows per mix.
* ``doc_ids``   — column→doc_id map (length n_samples).
* ``rng``       — np.random.Generator (E and the random-baseline use it).

``StrategyResult`` is a plain dict with the keys the runner writes out.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


### Shared SNR primitives — vectorised so B can score thousands of
### candidates per greedy step without a Python loop. ###


def signal_noise_1d(combined: np.ndarray, mix_rows: dict[str, list[int]]
                    ) -> tuple[float, float, float]:
    """SNR for a single combined-score vector (length n_ckpts)."""
    arrs = [combined[rows] for rows in mix_rows.values()]
    arrs = [a for a in arrs if a.size >= 2]
    if len(arrs) < 2:
        return float("nan"), float("nan"), float("nan")
    mix_means = np.array([a.mean() for a in arrs])
    pooled = np.concatenate(arrs)
    if pooled.size == 0 or pooled.mean() == 0:
        return float("nan"), float("nan"), float("nan")
    dispersion = mix_means.max() - mix_means.min()
    signal = dispersion / mix_means.mean() if mix_means.mean() else float("nan")
    noise = pooled.std() / pooled.mean() if pooled.mean() else float("nan")
    if not (np.isfinite(signal) and np.isfinite(noise)) or noise == 0:
        return signal, noise, float("nan")
    return float(signal), float(noise), float(signal / noise)


def signal_noise_batch(combined_mat: np.ndarray,
                       mix_rows: dict[str, list[int]]) -> np.ndarray:
    """SNR over each column of ``combined_mat`` (n_ckpts × n_cand).

    Used by Option B's forward-greedy step to score every candidate
    sample against the current subset in a single numpy call.
    """
    if combined_mat.size == 0:
        return np.empty((0,), dtype=np.float64)
    mixes = list(mix_rows.keys())
    if len(mixes) < 2:
        return np.full(combined_mat.shape[1], np.nan, dtype=np.float64)
    mix_means = np.stack([combined_mat[mix_rows[m], :].mean(axis=0)
                          for m in mixes], axis=0)
    pooled_rows = [r for m in mixes for r in mix_rows[m]]
    pooled = combined_mat[pooled_rows, :]
    overall_mix_mean = mix_means.mean(axis=0)
    pooled_mean = pooled.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        signal = (mix_means.max(axis=0) - mix_means.min(axis=0)) / overall_mix_mean
        noise = pooled.std(axis=0) / pooled_mean
        snr = signal / noise
    return np.where(np.isfinite(snr), snr, -np.inf)


def per_sample_snr(A: np.ndarray, mix_rows: dict[str, list[int]]) -> np.ndarray:
    """Per-sample SNR scoring each column of A independently."""
    if A.size == 0:
        return np.empty((0,), dtype=np.float64)
    mixes = list(mix_rows.keys())
    if len(mixes) < 2:
        return np.full(A.shape[1], np.nan, dtype=np.float64)
    mix_means = np.stack([A[mix_rows[m], :].mean(axis=0) for m in mixes], axis=0)
    pooled_rows = [r for m in mixes for r in mix_rows[m]]
    pooled = A[pooled_rows, :]
    overall_mix_mean = mix_means.mean(axis=0)
    pooled_mean = pooled.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        signal = (mix_means.max(axis=0) - mix_means.min(axis=0)) / overall_mix_mean
        noise = pooled.std(axis=0) / pooled_mean
        snr = signal / noise
    return np.where(np.isfinite(snr), snr, np.nan)


def variance_prefilter_mask(A: np.ndarray, mix_rows: dict[str, list[int]]
                            ) -> np.ndarray:
    """True where per-mix mean acc varies across mixes (sample is not 'dead')."""
    if A.size == 0:
        return np.empty((0,), dtype=bool)
    mixes = list(mix_rows.keys())
    if len(mixes) < 2:
        return np.zeros(A.shape[1], dtype=bool)
    mix_means = np.stack([A[mix_rows[m], :].mean(axis=0) for m in mixes], axis=0)
    return mix_means.std(axis=0) > 0


def cumulative_subset_snrs(A: np.ndarray, ordered_cols: np.ndarray,
                           mix_rows: dict[str, list[int]]) -> list[float]:
    """SNR of A[:, ordered[:N]].mean(axis=1) for N=1..len(ordered).

    Vectorised: builds the (n_ckpts, n_ordered) matrix of cumulative
    means in one numpy call and scores every column with
    ``signal_noise_batch``. -inf sentinels (degenerate combined
    vectors) are converted back to NaN so callers can use
    ``argmax_safe`` without special-casing.
    """
    if ordered_cols.size == 0:
        return []
    cumsum = A[:, ordered_cols].cumsum(axis=1)
    counts = np.arange(1, ordered_cols.size + 1, dtype=np.float64)
    combined_mat = cumsum / counts  # (n_ckpts, n_ordered)
    snrs = signal_noise_batch(combined_mat, mix_rows)
    return [float(s) if np.isfinite(s) else float("nan") for s in snrs]


def argmax_safe(values: list[float] | np.ndarray) -> int:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return -1
    arr = np.where(np.isfinite(arr), arr, -np.inf)
    return int(np.argmax(arr))


### Strategy primitive: order column indices then sweep ###


def _sweep_and_summarise(A: np.ndarray,
                         ordered: np.ndarray,
                         mix_rows: dict[str, list[int]],
                         per_sample_snrs_arr: np.ndarray | None = None) -> dict:
    """Common tail: cumulative sweep + best-N + full-set SNR + roundup."""
    cumulative = cumulative_subset_snrs(A, ordered, mix_rows)
    best_idx = argmax_safe(cumulative)
    full_idx = np.arange(A.shape[1])
    _, _, full_set_snr = signal_noise_1d(A[:, full_idx].mean(axis=1), mix_rows)
    return {
        "ordered": ordered,
        "cumulative_snrs": cumulative,
        "best_n": best_idx + 1 if best_idx >= 0 else 0,
        "best_snr": cumulative[best_idx] if best_idx >= 0 else float("nan"),
        "full_set_snr": full_set_snr,
        "per_sample_snrs": (per_sample_snrs_arr
                            if per_sample_snrs_arr is not None
                            else per_sample_snr(A, mix_rows)),
    }


def _sort_by_score(scores: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Sort ``candidates`` by ``scores[candidates]`` desc, NaN/-inf to bottom."""
    s = scores[candidates]
    finite = np.isfinite(s)
    head = candidates[finite][np.argsort(-s[finite])]
    tail = candidates[~finite]
    return np.concatenate([head, tail])


### Option A: per-sample SNR + sort ###


def select_a(A: np.ndarray, mix_rows: dict[str, list[int]],
             doc_ids: list[int], rng=None, **opts) -> dict:
    """Option A: rank every sample by per-sample SNR; sweep cumulative
    subsets in that order. No prefilter — dead samples (signal=0) come
    out NaN and sort to the back."""
    snrs = per_sample_snr(A, mix_rows)
    candidates = np.arange(A.shape[1])
    ordered = _sort_by_score(snrs, candidates)
    out = _sweep_and_summarise(A, ordered, mix_rows, per_sample_snrs_arr=snrs)
    out["strategy"] = "A"
    out["n_total"] = A.shape[1]
    out["n_candidates"] = A.shape[1]
    return out


### Option B: forward greedy selection ###


def select_b(A: np.ndarray, mix_rows: dict[str, list[int]],
             doc_ids: list[int], rng=None, *,
             pool_cap: int = 1000, budget: int | None = None,
             stop_on_drop: bool = False, patience: int = 50, **opts) -> dict:
    """Option B: forward greedy. Seed with the highest-SNR sample, then
    at each step pick the candidate that maximally raises combined SNR.

    ``pool_cap`` caps the candidate pool to the top-K samples by Option-A
    SNR — for mmlu_* (N≈14k) this is the difference between tractable
    and not. Set to 0 / None to disable.

    ``budget`` caps the maximum subset size (None → pool_cap).

    ``stop_on_drop`` short-circuits when SNR fails to improve for
    ``patience`` consecutive steps; the cumulative curve is padded with
    -inf afterwards so per-N comparisons still line up.
    """
    snrs = per_sample_snr(A, mix_rows)
    n_total = A.shape[1]

    # Restrict to a candidate pool — either everything (small N) or the
    # top-pool_cap by Option A.
    if pool_cap and pool_cap > 0 and n_total > pool_cap:
        finite_mask = np.isfinite(snrs)
        finite_idx = np.flatnonzero(finite_mask)
        order_finite = finite_idx[np.argsort(-snrs[finite_idx])]
        pool = order_finite[:pool_cap]
    else:
        pool = np.arange(n_total)

    if pool.size == 0:
        ordered = np.empty(0, dtype=np.int64)
        out = _sweep_and_summarise(A, ordered, mix_rows, per_sample_snrs_arr=snrs)
        out["strategy"] = "B"
        out["n_total"] = n_total
        out["n_candidates"] = 0
        return out

    if budget is None:
        budget = pool.size
    budget = min(budget, pool.size)

    # Seed: highest single-sample SNR within pool.
    pool_snrs = snrs[pool]
    seed_pos = int(np.nanargmax(np.where(np.isfinite(pool_snrs), pool_snrs, -np.inf)))
    seed_col = int(pool[seed_pos])

    selected = [seed_col]
    selected_set = {seed_col}
    running_sum = A[:, seed_col].astype(np.float64).copy()
    cumulative = []
    _, _, snr0 = signal_noise_1d(running_sum, mix_rows)
    cumulative.append(snr0)

    best_so_far = snr0 if np.isfinite(snr0) else -np.inf
    no_improve = 0

    pool_set = set(int(c) for c in pool.tolist())
    remaining = np.array(sorted(pool_set - selected_set), dtype=np.int64)

    for step in range(2, budget + 1):
        if remaining.size == 0:
            break
        # candidate combined = (running_sum[:, None] + A[:, remaining]) / step
        cand_combined = (running_sum[:, None] + A[:, remaining]) / step
        cand_snrs = signal_noise_batch(cand_combined, mix_rows)
        best_local = int(np.argmax(cand_snrs))
        best_snr = float(cand_snrs[best_local])
        chosen = int(remaining[best_local])

        running_sum += A[:, chosen]
        selected.append(chosen)
        selected_set.add(chosen)
        cumulative.append(best_snr)

        # Drop chosen from remaining.
        remaining = np.delete(remaining, best_local)

        if best_snr > best_so_far + 1e-9:
            best_so_far = best_snr
            no_improve = 0
        else:
            no_improve += 1
            if stop_on_drop and no_improve >= patience:
                break

    ordered = np.array(selected, dtype=np.int64)

    # The summarise helper recomputes a sweep; we already have ours from
    # the greedy descent so feed it ours.
    full_idx = np.arange(A.shape[1])
    _, _, full_set_snr = signal_noise_1d(A[:, full_idx].mean(axis=1), mix_rows)
    best_idx = argmax_safe(cumulative)
    return {
        "strategy": "B",
        "ordered": ordered,
        "cumulative_snrs": cumulative,
        "best_n": best_idx + 1 if best_idx >= 0 else 0,
        "best_snr": cumulative[best_idx] if best_idx >= 0 else float("nan"),
        "full_set_snr": full_set_snr,
        "per_sample_snrs": snrs,
        "n_total": n_total,
        "n_candidates": int(pool.size),
        "pool_cap_used": int(min(pool_cap or 0, n_total)) if pool_cap else n_total,
    }


### Option C: IRT-style discrimination filter, then Option A ###


def discrimination_index(A: np.ndarray) -> np.ndarray:
    """Classical-test-theory discrimination = corr(item, total).

    This is the point-biserial correlation between each item's binary
    acc vector across ckpts and the ckpt-level total score. It's a
    cheap stand-in for the 2PL ``a_i`` parameter (PROPOSALS.md) — the
    rank order is identical in the limit and we avoid an extra
    optimisation dependency. Constant items (item std=0) get NaN.
    """
    if A.size == 0:
        return np.empty((0,), dtype=np.float64)
    n_ckpts = A.shape[0]
    if n_ckpts < 2:
        return np.full(A.shape[1], np.nan, dtype=np.float64)
    total = A.mean(axis=1)  # (n_ckpts,) — ckpt total score
    total_mean = total.mean()
    total_std = total.std()
    if total_std == 0:
        return np.full(A.shape[1], np.nan, dtype=np.float64)

    item_mean = A.mean(axis=0)  # (n_samples,)
    item_std = A.std(axis=0)
    cov = ((A - item_mean) * (total - total_mean)[:, None]).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = cov / (item_std * total_std)
    return np.where(np.isfinite(rho), rho, np.nan)


def select_c(A: np.ndarray, mix_rows: dict[str, list[int]],
             doc_ids: list[int], rng=None, *,
             keep_frac: float = 0.5, min_keep: int = 50, **opts) -> dict:
    """Option C: drop items with the lowest discrimination, then Option A.

    Keeps the top ``keep_frac`` of samples by the classical-test-theory
    discrimination index (or at least ``min_keep``). Items with NaN
    discrimination (constant across all ckpts → no information) are
    dropped first. Ranks the survivors by per-sample SNR exactly like
    Option A, then sweeps the cumulative subset.
    """
    snrs = per_sample_snr(A, mix_rows)
    disc = discrimination_index(A)
    n_total = A.shape[1]

    finite_disc = np.isfinite(disc)
    n_finite = int(finite_disc.sum())
    if n_finite == 0:
        ordered = np.empty(0, dtype=np.int64)
        out = _sweep_and_summarise(A, ordered, mix_rows, per_sample_snrs_arr=snrs)
        out["strategy"] = "C"
        out["n_total"] = n_total
        out["n_candidates"] = 0
        out["discrimination"] = disc
        return out

    keep_n = max(int(round(keep_frac * n_finite)), min(min_keep, n_finite))
    finite_idx = np.flatnonzero(finite_disc)
    order_by_disc = finite_idx[np.argsort(-disc[finite_idx])]
    survivors = order_by_disc[:keep_n]

    ordered = _sort_by_score(snrs, survivors)
    out = _sweep_and_summarise(A, ordered, mix_rows, per_sample_snrs_arr=snrs)
    out["strategy"] = "C"
    out["n_total"] = n_total
    out["n_candidates"] = int(survivors.size)
    out["discrimination"] = disc
    out["keep_frac_used"] = float(keep_frac)
    return out


### Option D: variance prefilter then Option A (kept here so the
### runner can pull every strategy from one place; mirrors the original
### in smooth_subtasks_per_sample._run_one_size). ###


def select_d(A: np.ndarray, mix_rows: dict[str, list[int]],
             doc_ids: list[int], rng=None, **opts) -> dict:
    snrs = per_sample_snr(A, mix_rows)
    keep = variance_prefilter_mask(A, mix_rows)
    n_total = A.shape[1]
    survivor_idx = np.flatnonzero(keep)
    if survivor_idx.size == 0:
        ordered = np.empty(0, dtype=np.int64)
        out = _sweep_and_summarise(A, ordered, mix_rows, per_sample_snrs_arr=snrs)
        out["strategy"] = "D"
        out["n_total"] = n_total
        out["n_candidates"] = 0
        return out
    ordered = _sort_by_score(snrs, survivor_idx)
    out = _sweep_and_summarise(A, ordered, mix_rows, per_sample_snrs_arr=snrs)
    out["strategy"] = "D"
    out["n_total"] = n_total
    out["n_candidates"] = int(survivor_idx.size)
    return out


### Option E: random / black-box search ###


def select_e(A: np.ndarray, mix_rows: dict[str, list[int]],
             doc_ids: list[int], rng=None, *,
             n_random_orders: int = 32, **opts) -> dict:
    """Option E: random search over orderings.

    Picks the random permutation whose cumulative-SNR sweep peaks
    highest. The reported ``ordered`` is that winning permutation, so
    its cumulative curve is what gets plotted. Useful as a sanity check
    against A/B/C/D — if random reliably beats them, the per-sample SNR
    ranking is the wrong primitive.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    snrs = per_sample_snr(A, mix_rows)
    n_total = A.shape[1]
    if n_total == 0:
        ordered = np.empty(0, dtype=np.int64)
        out = _sweep_and_summarise(A, ordered, mix_rows, per_sample_snrs_arr=snrs)
        out["strategy"] = "E"
        out["n_total"] = n_total
        out["n_candidates"] = 0
        return out

    base = np.arange(n_total)
    best_peak = -np.inf
    best_curve: list[float] = []
    best_perm = base.copy()
    for _ in range(max(1, int(n_random_orders))):
        perm = base.copy()
        rng.shuffle(perm)
        curve = cumulative_subset_snrs(A, perm, mix_rows)
        peak_idx = argmax_safe(curve)
        peak = curve[peak_idx] if peak_idx >= 0 else -np.inf
        if peak > best_peak:
            best_peak = peak
            best_curve = curve
            best_perm = perm

    best_idx = argmax_safe(best_curve)
    full_idx = np.arange(A.shape[1])
    _, _, full_set_snr = signal_noise_1d(A[:, full_idx].mean(axis=1), mix_rows)
    return {
        "strategy": "E",
        "ordered": best_perm,
        "cumulative_snrs": best_curve,
        "best_n": best_idx + 1 if best_idx >= 0 else 0,
        "best_snr": best_curve[best_idx] if best_idx >= 0 else float("nan"),
        "full_set_snr": full_set_snr,
        "per_sample_snrs": snrs,
        "n_total": n_total,
        "n_candidates": n_total,
        "n_random_orders": int(n_random_orders),
    }


STRATEGIES: dict[str, callable] = {
    "A": select_a,
    "B": select_b,
    "C": select_c,
    "D": select_d,
    "E": select_e,
}


def run_strategy(name: str, A: np.ndarray,
                 mix_rows: dict[str, list[int]],
                 doc_ids: list[int],
                 rng=None, **opts) -> dict:
    if name not in STRATEGIES:
        raise KeyError(f"unknown strategy {name!r}; have {list(STRATEGIES)}")
    return STRATEGIES[name](A, mix_rows, doc_ids, rng=rng, **opts)


### Random-order baseline (the dashed red curve in Option D's plot) — ###
### kept separate so it can be reused by both runners.                ###


def random_order_curve(A: np.ndarray, ordered: np.ndarray,
                       mix_rows: dict[str, list[int]],
                       rng) -> list[float]:
    """Cumulative-SNR sweep of a random permutation of ``ordered``."""
    if ordered.size == 0:
        return []
    perm = ordered.copy()
    rng.shuffle(perm)
    return cumulative_subset_snrs(A, perm, mix_rows)
