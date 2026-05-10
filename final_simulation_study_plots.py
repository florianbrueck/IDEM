from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GROUP_STYLES = {
    0: {
        "name": "Group 1",
        "color": "#1d4ed8",
        "light": "#93c5fd",
    },
    1: {
        "name": "Group 2",
        "color": "#b91c1c",
        "light": "#fca5a5",
    },
}


def _to_bivariate_array(samples, name):
    arr = np.asarray(samples, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 2:
        out = arr
    elif arr.ndim == 3 and arr.shape[2] == 2:
        out = arr.reshape(-1, 2)
    else:
        raise ValueError(f"{name} must have shape (M, 2) or (M, N, 2).")

    out = out[np.isfinite(out).all(axis=1)]
    if out.shape[0] == 0:
        raise ValueError(f"{name} contains no finite bivariate samples.")
    return out


def _coerce_survival_curve(spec, name):
    if isinstance(spec, dict):
        x_grid = spec.get("x_grid")
        survival = spec.get("survival")
        if survival is None:
            survival = spec.get("survival_values")
    elif isinstance(spec, (list, tuple)) and len(spec) == 2:
        x_grid, survival = spec
    else:
        raise ValueError(
            f"{name} must be a dict with x_grid/survival entries or a (x_grid, survival_values) pair."
        )

    x_grid = np.asarray(x_grid, dtype=float)
    survival = np.asarray(survival, dtype=float)
    if x_grid.ndim != 1 or survival.ndim != 1:
        raise ValueError(f"{name} must contain one-dimensional arrays.")
    if x_grid.size == 0 or survival.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if x_grid.shape != survival.shape:
        raise ValueError(f"{name} x_grid and survival must have the same shape.")
    if not np.isfinite(x_grid).all() or not np.isfinite(survival).all():
        raise ValueError(f"{name} contains non-finite values.")
    if np.any(np.diff(x_grid) < 0.0):
        raise ValueError(f"{name} x_grid must be nondecreasing.")

    survival = np.clip(survival, 0.0, 1.0)
    survival = np.minimum.accumulate(survival)
    return x_grid, survival


def _coerce_prior_reference(prior_reference):
    try:
        prior_samples = _to_bivariate_array(prior_reference, "prior_predictive_samples")
        return {"kind": "samples", "value": prior_samples}
    except (TypeError, ValueError):
        pass

    try:
        if len(prior_reference) == 2:
            survival_curves = [
                _coerce_survival_curve(prior_reference[0], "prior_predictive_samples[0]"),
                _coerce_survival_curve(prior_reference[1], "prior_predictive_samples[1]"),
            ]
            return {"kind": "survival_curves", "value": survival_curves}
    except (TypeError, ValueError, KeyError):
        pass

    raise ValueError(
        "prior_predictive_samples must be either bivariate samples with shape (M, 2) or (M, N, 2), or two marginal survival curves."
    )


def _coerce_observed(observed_event_times_by_group):
    if len(observed_event_times_by_group) != 2:
        raise ValueError("observed_event_times_by_group must contain two arrays.")

    obs = [
        np.asarray(observed_event_times_by_group[0], dtype=float),
        np.asarray(observed_event_times_by_group[1], dtype=float),
    ]
    obs = [values[np.isfinite(values)] for values in obs]
    if obs[0].size == 0 or obs[1].size == 0:
        raise ValueError("Both observed groups must contain at least one event time.")
    return obs


def _survival_scores(sample_times, t_grid):
    sample_times = np.asarray(sample_times, dtype=float)
    sample_times = sample_times[np.isfinite(sample_times)]
    if sample_times.size == 0:
        raise ValueError("Cannot compute survival from empty sample.")
    return sample_times[:, None] > t_grid[None, :]


def _survival_from_curve(x_grid, survival_values, t_grid):
    t_grid = np.asarray(t_grid, dtype=float)
    return np.interp(t_grid, x_grid, survival_values, left=1.0, right=float(survival_values[-1]))


def _cdf_from_curve(x_grid, survival_values, points):
    points = np.asarray(points, dtype=float)
    survival_at_points = _survival_from_curve(x_grid, survival_values, points)
    return np.clip(1.0 - survival_at_points, 0.0, 1.0)


def _local_mass_scores(sample_times, observed_times, r_grid):
    sample_times = np.asarray(sample_times, dtype=float)
    observed_times = np.asarray(observed_times, dtype=float)
    sample_times = sample_times[np.isfinite(sample_times)]
    observed_times = observed_times[np.isfinite(observed_times)]
    if sample_times.size == 0:
        raise ValueError("Cannot compute local mass from empty sample.")
    if observed_times.size == 0:
        raise ValueError("Observed event times are empty.")

    distances = np.abs(sample_times[:, None] - observed_times[None, :])
    return (distances[:, :, None] <= r_grid[None, None, :]).mean(axis=1)


def _one_sided_local_mass_scores(sample_times, observed_times, r_grid, *, direction):
    sample_times = np.asarray(sample_times, dtype=float)
    observed_times = np.asarray(observed_times, dtype=float)
    sample_times = sample_times[np.isfinite(sample_times)]
    observed_times = observed_times[np.isfinite(observed_times)]
    if sample_times.size == 0:
        raise ValueError("Cannot compute local mass from empty sample.")
    if observed_times.size == 0:
        raise ValueError("Observed event times are empty.")

    signed_differences = sample_times[:, None] - observed_times[None, :]
    if direction == "positive":
        scores = (signed_differences[:, :, None] >= 0.0) & (signed_differences[:, :, None] <= r_grid[None, None, :])
    elif direction == "negative":
        scores = (signed_differences[:, :, None] <= 0.0) & (-signed_differences[:, :, None] <= r_grid[None, None, :])
    else:
        raise ValueError(f"Unknown direction: {direction}")
    return scores.mean(axis=1)


def _normalized_one_sided_local_mass_scores(sample_times, observed_times, r_grid, *, direction):
    sample_times = np.asarray(sample_times, dtype=float)
    observed_times = np.asarray(observed_times, dtype=float)
    sample_times = sample_times[np.isfinite(sample_times)]
    observed_times = observed_times[np.isfinite(observed_times)]
    if sample_times.size == 0:
        raise ValueError("Cannot compute local mass from empty sample.")
    if observed_times.size == 0:
        raise ValueError("Observed event times are empty.")

    signed_differences = sample_times[:, None] - observed_times[None, :]
    if direction == "positive":
        denominators = np.mean(sample_times[:, None] > observed_times[None, :], axis=0)
        if np.any(denominators <= 0.0):
            raise ValueError("Normalization by P(X > t_obs) is undefined because at least one estimated survival probability is zero.")
        scores = (signed_differences[:, :, None] >= 0.0) & (signed_differences[:, :, None] <= r_grid[None, None, :])
    elif direction == "negative":
        denominators = 1.0 - np.mean(sample_times[:, None] > observed_times[None, :], axis=0)
        if np.any(denominators <= 0.0):
            raise ValueError("Normalization by P(X <= t_obs) is undefined because at least one estimated lower-tail probability is zero.")
        scores = (signed_differences[:, :, None] <= 0.0) & (-signed_differences[:, :, None] <= r_grid[None, None, :])
    else:
        raise ValueError(f"Unknown direction: {direction}")
    return (scores / denominators[None, :, None]).mean(axis=1)


def _local_mass_curve_from_survival(x_grid, survival_values, observed_times, r_grid):
    observed_times = np.asarray(observed_times, dtype=float)
    observed_times = observed_times[np.isfinite(observed_times)]
    if observed_times.size == 0:
        raise ValueError("Observed event times are empty.")

    out = []
    for radius in r_grid:
        lower = np.maximum(0.0, observed_times - radius)
        upper = observed_times + radius
        probs = _cdf_from_curve(x_grid, survival_values, upper) - _cdf_from_curve(x_grid, survival_values, lower)
        out.append(np.mean(probs))

    return np.asarray(out)


def _one_sided_local_mass_curve_from_survival(x_grid, survival_values, observed_times, r_grid, *, direction):
    observed_times = np.asarray(observed_times, dtype=float)
    observed_times = observed_times[np.isfinite(observed_times)]
    if observed_times.size == 0:
        raise ValueError("Observed event times are empty.")

    out = []
    for radius in r_grid:
        if direction == "positive":
            lower = observed_times
            upper = observed_times + radius
        elif direction == "negative":
            lower = np.maximum(0.0, observed_times - radius)
            upper = observed_times
        else:
            raise ValueError(f"Unknown direction: {direction}")
        probs = _cdf_from_curve(x_grid, survival_values, upper) - _cdf_from_curve(x_grid, survival_values, lower)
        out.append(np.mean(probs))

    return np.asarray(out)


def _normalized_one_sided_local_mass_curve_from_survival(x_grid, survival_values, observed_times, r_grid, *, direction):
    observed_times = np.asarray(observed_times, dtype=float)
    observed_times = observed_times[np.isfinite(observed_times)]
    if observed_times.size == 0:
        raise ValueError("Observed event times are empty.")

    out = []
    for radius in r_grid:
        if direction == "positive":
            denominators = _survival_from_curve(x_grid, survival_values, observed_times)
            if np.any(denominators <= 0.0):
                raise ValueError("Normalization by P(X > t_obs) is undefined because at least one survival probability is zero.")
            lower = observed_times
            upper = observed_times + radius
        elif direction == "negative":
            denominators = _cdf_from_curve(x_grid, survival_values, observed_times)
            if np.any(denominators <= 0.0):
                raise ValueError("Normalization by P(X <= t_obs) is undefined because at least one lower-tail probability is zero.")
            lower = np.maximum(0.0, observed_times - radius)
            upper = observed_times
        else:
            raise ValueError(f"Unknown direction: {direction}")
        probs = _cdf_from_curve(x_grid, survival_values, upper) - _cdf_from_curve(x_grid, survival_values, lower)
        out.append(np.mean(probs / denominators))

    return np.asarray(out)


def _one_sided_local_mass_term_probabilities(sample_times, observed_times, r_grid, *, direction):
    sample_times = np.asarray(sample_times, dtype=float)
    observed_times = np.asarray(observed_times, dtype=float)
    sample_times = sample_times[np.isfinite(sample_times)]
    observed_times = observed_times[np.isfinite(observed_times)]
    if sample_times.size == 0:
        raise ValueError("Cannot compute local mass from empty sample.")
    if observed_times.size == 0:
        raise ValueError("Observed event times are empty.")

    signed_differences = sample_times[:, None] - observed_times[None, :]
    if direction == "positive":
        indicators = (signed_differences[:, :, None] >= 0.0) & (signed_differences[:, :, None] <= r_grid[None, None, :])
    elif direction == "negative":
        indicators = (signed_differences[:, :, None] <= 0.0) & (-signed_differences[:, :, None] <= r_grid[None, None, :])
    else:
        raise ValueError(f"Unknown direction: {direction}")
    return np.mean(indicators, axis=0)


def _one_sided_local_mass_term_probabilities_from_survival(x_grid, survival_values, observed_times, r_grid, *, direction):
    observed_times = np.asarray(observed_times, dtype=float)
    observed_times = observed_times[np.isfinite(observed_times)]
    if observed_times.size == 0:
        raise ValueError("Observed event times are empty.")

    term_probabilities = np.empty((observed_times.size, len(r_grid)), dtype=float)
    for idx, t_obs in enumerate(observed_times):
        if direction == "positive":
            lower = np.full_like(r_grid, t_obs, dtype=float)
            upper = t_obs + r_grid
        elif direction == "negative":
            lower = np.maximum(0.0, t_obs - r_grid)
            upper = np.full_like(r_grid, t_obs, dtype=float)
        else:
            raise ValueError(f"Unknown direction: {direction}")
        term_probabilities[idx, :] = _cdf_from_curve(x_grid, survival_values, upper) - _cdf_from_curve(x_grid, survival_values, lower)

    return term_probabilities


def _hazard_ratio_one_sided_local_mass_from_samples(sample_times, observed_times, r_grid, *, direction, survival_floor):
    sample_times = np.asarray(sample_times, dtype=float)
    observed_times = np.asarray(observed_times, dtype=float)
    sample_times = sample_times[np.isfinite(sample_times)]
    observed_times = observed_times[np.isfinite(observed_times)]
    if sample_times.size == 0:
        raise ValueError("Cannot compute cumulative-hazard ratio from empty sample.")
    if observed_times.size == 0:
        raise ValueError("Observed event times are empty.")

    denominator_hazards = _cumhaz_from_survival(np.mean(sample_times[:, None] > observed_times[None, :], axis=0), survival_floor)
    if np.any(denominator_hazards <= 0.0):
        raise ValueError("Hazard-ratio normalization is undefined because at least one denominator cumulative hazard is zero.")

    if direction == "positive":
        shifted_times = observed_times[:, None] + r_grid[None, :]
    elif direction == "negative":
        shifted_times = np.maximum(0.0, observed_times[:, None] - r_grid[None, :])
    else:
        raise ValueError(f"Unknown direction: {direction}")

    numerator_survival = np.mean(sample_times[:, None, None] > shifted_times[None, :, :], axis=0)
    numerator_hazards = _cumhaz_from_survival(numerator_survival, survival_floor)
    return np.mean(numerator_hazards / denominator_hazards[:, None], axis=0)


def _hazard_ratio_one_sided_local_mass_from_survival(x_grid, survival_values, observed_times, r_grid, *, direction, survival_floor):
    observed_times = np.asarray(observed_times, dtype=float)
    observed_times = observed_times[np.isfinite(observed_times)]
    if observed_times.size == 0:
        raise ValueError("Observed event times are empty.")

    survival_probabilities = _survival_from_curve(x_grid, survival_values, observed_times)
    denominator_hazards = _cumhaz_from_survival(survival_probabilities, survival_floor)
    if np.any(denominator_hazards <= 0.0):
        raise ValueError("Hazard-ratio normalization is undefined because at least one denominator cumulative hazard is zero.")

    if direction == "positive":
        shifted_times = observed_times[:, None] + r_grid[None, :]
    elif direction == "negative":
        shifted_times = np.maximum(0.0, observed_times[:, None] - r_grid[None, :])
    else:
        raise ValueError(f"Unknown direction: {direction}")

    numerator_hazards = _cumhaz_from_survival(_survival_from_curve(x_grid, survival_values, shifted_times), survival_floor)
    return np.mean(numerator_hazards / denominator_hazards[:, None], axis=0)


def _nearest_distance_scores(sample_times, observed_times, r_grid):
    sample_times = np.asarray(sample_times, dtype=float)
    observed_times = np.asarray(observed_times, dtype=float)
    sample_times = sample_times[np.isfinite(sample_times)]
    observed_times = observed_times[np.isfinite(observed_times)]
    if sample_times.size == 0:
        raise ValueError("Cannot compute nearest distance from empty sample.")
    if observed_times.size == 0:
        raise ValueError("Observed event times are empty.")

    distances = np.min(np.abs(sample_times[:, None] - observed_times[None, :]), axis=1)
    scores = distances[:, None] <= r_grid[None, :]
    return scores, distances


def _one_sided_nearest_scores(sample_times, observed_times, r_grid, *, direction):
    sample_times = np.asarray(sample_times, dtype=float)
    observed_times = np.asarray(observed_times, dtype=float)
    sample_times = sample_times[np.isfinite(sample_times)]
    observed_times = observed_times[np.isfinite(observed_times)]
    if sample_times.size == 0:
        raise ValueError("Cannot compute nearest distance from empty sample.")
    if observed_times.size == 0:
        raise ValueError("Observed event times are empty.")

    observed_times = np.sort(observed_times)
    signed_differences = sample_times[:, None] - observed_times[None, :]
    nearest_indices = np.argmin(np.abs(signed_differences), axis=1)
    nearest_signed_distances = signed_differences[np.arange(sample_times.size), nearest_indices]

    if direction == "positive":
        hits = (nearest_signed_distances[:, None] >= 0.0) & (nearest_signed_distances[:, None] <= r_grid[None, :])
    elif direction == "negative":
        hits = (nearest_signed_distances[:, None] < 0.0) & (-nearest_signed_distances[:, None] <= r_grid[None, :])
    else:
        raise ValueError(f"Unknown direction: {direction}")
    return hits


def _merge_intervals(intervals):
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda interval: interval[0])
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        current = merged[-1]
        if start <= current[1]:
            current[1] = max(current[1], end)
        else:
            merged.append([start, end])

    return [(start, end) for start, end in merged]


def _distance_cdf_from_survival(x_grid, survival_values, observed_times, r_grid):
    observed_times = np.asarray(observed_times, dtype=float)
    observed_times = observed_times[np.isfinite(observed_times)]
    if observed_times.size == 0:
        raise ValueError("Observed event times are empty.")

    out = []
    for radius in r_grid:
        intervals = _merge_intervals([
            (max(0.0, float(t_obs - radius)), float(t_obs + radius))
            for t_obs in observed_times
        ])

        probability = 0.0
        for lower, upper in intervals:
            probability += float(_cdf_from_curve(x_grid, survival_values, upper) - _cdf_from_curve(x_grid, survival_values, lower))
        out.append(probability)

    return np.clip(np.asarray(out), 0.0, 1.0)


def _one_sided_distance_cdf_from_survival(x_grid, survival_values, observed_times, r_grid, *, direction):
    observed_times = np.asarray(observed_times, dtype=float)
    observed_times = observed_times[np.isfinite(observed_times)]
    if observed_times.size == 0:
        raise ValueError("Observed event times are empty.")

    observed_times = np.sort(observed_times)
    midpoints = 0.5 * (observed_times[:-1] + observed_times[1:]) if observed_times.size > 1 else np.asarray([], dtype=float)

    out = []
    for radius in r_grid:
        intervals = []
        for idx, t_obs in enumerate(observed_times):
            left_cell = 0.0 if idx == 0 else float(midpoints[idx - 1])
            right_cell = np.inf if idx == observed_times.size - 1 else float(midpoints[idx])

            if direction == "positive":
                lower = float(t_obs)
                upper = float(min(t_obs + radius, right_cell))
            elif direction == "negative":
                lower = float(max(0.0, left_cell, t_obs - radius))
                upper = float(t_obs)
            else:
                raise ValueError(f"Unknown direction: {direction}")

            if upper > lower:
                intervals.append((lower, upper))

        intervals = _merge_intervals(intervals)

        probability = 0.0
        for lower, upper in intervals:
            probability += float(_cdf_from_curve(x_grid, survival_values, upper) - _cdf_from_curve(x_grid, survival_values, lower))
        out.append(probability)

    return np.clip(np.asarray(out), 0.0, 1.0)


def _dkw_epsilon(n_samples, confidence_level):
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be between 0 and 1.")

    alpha = 1.0 - confidence_level
    return np.sqrt(np.log(2.0 / alpha) / (2.0 * n_samples))


def _dkw_band_from_scores(scores, confidence_level=0.90, epsilon_multiplier=1.0, clip_to_unit_interval=True):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("scores must have shape (M, G).")

    estimate = np.mean(scores, axis=0)
    eps = epsilon_multiplier * _dkw_epsilon(scores.shape[0], confidence_level)
    lower = estimate - eps
    upper = estimate + eps

    if clip_to_unit_interval:
        lower = np.clip(lower, 0.0, 1.0)
        upper = np.clip(upper, 0.0, 1.0)

    return estimate, lower, upper, eps


def _cumhaz_from_survival(survival, survival_floor):
    survival = np.asarray(survival, dtype=float)
    survival = np.maximum(survival, survival_floor)
    return -np.log(survival)


def _cumhaz_band_from_survival_band(surv_lower, surv_upper, survival_floor):
    if surv_lower is None or surv_upper is None:
        return None, None
    h_lower = _cumhaz_from_survival(surv_upper, survival_floor)
    h_upper = _cumhaz_from_survival(surv_lower, survival_floor)
    return h_lower, h_upper


def _cumhaz_from_event_probability(probability, probability_floor):
    probability = np.asarray(probability, dtype=float)
    probability = np.clip(probability, 0.0, 1.0)
    return -np.log(np.maximum(1.0 - probability, probability_floor))


def _inverse_transform_sample_from_survival(x_grid, survival_values, n_samples, *, rng_seed):
    x_grid = np.asarray(x_grid, dtype=np.float64)
    survival_values = np.asarray(survival_values, dtype=np.float64)
    cdf_values = np.clip(1.0 - survival_values, 0.0, 1.0)
    cdf_values = np.maximum.accumulate(cdf_values)

    unique_cdf, unique_indices = np.unique(cdf_values, return_index=True)
    quantile_grid = x_grid[unique_indices]

    if unique_cdf[0] > 0.0:
        unique_cdf = np.insert(unique_cdf, 0, 0.0)
        quantile_grid = np.insert(quantile_grid, 0, x_grid[0])
    if unique_cdf[-1] < 1.0:
        unique_cdf = np.append(unique_cdf, 1.0)
        quantile_grid = np.append(quantile_grid, x_grid[-1])

    rng = np.random.default_rng(rng_seed)
    uniforms = rng.random(int(n_samples))
    return np.interp(uniforms, unique_cdf, quantile_grid, left=quantile_grid[0], right=quantile_grid[-1])


def _suggest_time_window(t_grid, posterior_samples, observed_data, *, quantile=0.995):
    t_grid = np.asarray(t_grid, dtype=np.float64)
    posterior_samples = np.asarray(posterior_samples, dtype=np.float64)
    observed_data = [np.asarray(values, dtype=np.float64) for values in observed_data]

    x_left = max(0.0, float(t_grid[0]))
    observed_max = max(float(np.max(observed_data[0])), float(np.max(observed_data[1])))
    posterior_max = float(np.quantile(posterior_samples.reshape(-1), quantile))
    x_right = min(float(t_grid[-1]), max(observed_max * 1.1, posterior_max))
    mask = (t_grid >= x_left) & (t_grid <= x_right)
    return x_left, x_right, mask


def _setup_matplotlib():
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["xtick.major.width"] = 1.2
    plt.rcParams["ytick.major.width"] = 1.2
    plt.rcParams["xtick.major.size"] = 6
    plt.rcParams["ytick.major.size"] = 6


def _format_axis(ax, xlabel: str, ylabel: str = "", title: str | None = None):
    ax.set_xlabel(xlabel, fontsize=18)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=18)
    if title is not None:
        ax.set_title(title, fontsize=19)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.grid(False)


def _add_group_observation_lines(ax, observed_data, primary_group, *, primary_label=False, secondary_label=False):
    primary = np.asarray(observed_data[primary_group], dtype=np.float64)
    secondary = np.asarray(observed_data[1 - primary_group], dtype=np.float64)

    for idx, value in enumerate(np.sort(primary)):
        ax.axvline(
            value,
            ymin=0.0,
            ymax=1.0,
            linewidth=1.1,
            linestyle="-",
            color=GROUP_STYLES[primary_group]["color"],
            alpha=0.16,
            zorder=2.0,
            label=f"{GROUP_STYLES[primary_group]['name']} observations" if primary_label and idx == 0 else None,
        )

    for idx, value in enumerate(np.sort(secondary)):
        ax.axvline(
            value,
            ymin=0.0,
            ymax=1.0,
            linewidth=1.0,
            linestyle="--",
            color=GROUP_STYLES[1 - primary_group]["color"],
            alpha=0.12,
            zorder=1.9,
            label=f"{GROUP_STYLES[1 - primary_group]['name']} observations" if secondary_label and idx == 0 else None,
        )


def _add_mean_reference_lines(
    ax,
    prior_mean,
    posterior_mean,
    *,
    color,
    empirical_mean=None,
    label_empirical=False,
    prior_linestyle=(0, (1.0, 1.2)),
    posterior_linestyle=(0, (10, 2.5)),
    empirical_linestyle=(0, (12, 2.5, 2.0, 2.5)),
):
    ax.axvline(
        prior_mean,
        color=color,
        linewidth=2.2,
        linestyle=prior_linestyle,
        alpha=0.98,
        zorder=6,
        label=None,
    )
    ax.axvline(
        posterior_mean,
        color=color,
        linewidth=2.6,
        linestyle=posterior_linestyle,
        alpha=0.98,
        zorder=6,
        label=None,
    )
    if empirical_mean is not None:
        ax.axvline(
            empirical_mean,
            color=color,
            linewidth=3.0,
            linestyle=empirical_linestyle,
            alpha=0.98,
            zorder=6,
            label="Empirical mean" if label_empirical else None,
        )


def _plot_curve_with_optional_band(
    ax,
    x,
    y,
    *,
    lower=None,
    upper=None,
    label,
    linestyle="-",
    linewidth=2.7,
    band_label=None,
    band_alpha=0.22,
    step=False,
):
    if step:
        line = ax.step(x, y, where="post", linewidth=linewidth, linestyle=linestyle, label=label, zorder=4.0)[0]
    else:
        line = ax.plot(x, y, linewidth=linewidth, linestyle=linestyle, label=label, zorder=4.0)[0]

    if lower is not None and upper is not None:
        fill_kwargs = {"step": "post"} if step else {}
        ax.fill_between(
            x,
            lower,
            upper,
            color=line.get_color(),
            alpha=band_alpha,
            linewidth=0.0,
            label=band_label,
            zorder=1.5,
            **fill_kwargs,
        )

    return line


def _add_full_height_observation_lines(
    ax,
    primary_times,
    secondary_times=None,
    *,
    primary_label="Observed events, same group",
    secondary_label="Observed events, other group",
):
    primary_times = np.asarray(primary_times, dtype=float)
    primary_times = primary_times[np.isfinite(primary_times)]

    if secondary_times is not None:
        secondary_times = np.asarray(secondary_times, dtype=float)
        secondary_times = secondary_times[np.isfinite(secondary_times)]

    for idx, value in enumerate(primary_times):
        ax.axvline(value, ymin=0.0, ymax=1.0, linewidth=1.6, linestyle="-", color="0.15", alpha=0.30, zorder=2.5, label=primary_label if idx == 0 else None)

    if secondary_times is not None:
        for idx, value in enumerate(secondary_times):
            ax.axvline(value, ymin=0.0, ymax=1.0, linewidth=1.5, linestyle="--", color="0.55", alpha=0.25, zorder=2.5, label=secondary_label if idx == 0 else None)


def _save_figure(fig, save_dir: Path, file_name: str, *, save_pdf: bool, dpi: int):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    png_path = save_dir / f"{file_name}.png"
    if png_path.exists():
        png_path.unlink()

    pdf_path = None
    if save_pdf:
        pdf_path = save_dir / f"{file_name}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")

    plt.close(fig)
    return {"png": None, "pdf": pdf_path}


def _plot_two_margin_directional_diagnostic(
    margin_results,
    r_grid,
    save_dir: Path,
    *,
    prior_key: str,
    posterior_key: str,
    xlabel: str,
    ylabel: str,
    file_name: str,
    step: bool,
    fixed_ylim: tuple[float, float] | None = (-0.02, 1.02),
    save_pdf: bool,
    dpi: int,
):
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8), sharey=True)
    y_max = None
    if fixed_ylim is None:
        all_values = []
        for res in margin_results:
            all_values.append(np.asarray(res[prior_key], dtype=float))
            all_values.append(np.asarray(res[posterior_key], dtype=float))
        finite_values = np.concatenate([values[np.isfinite(values)] for values in all_values])
        max_value = float(np.max(finite_values)) if finite_values.size else 1.0
        y_max = max(1.02, 1.05 * max_value)
    safe_y_max = 1.02 if y_max is None else y_max
    for margin, ax in enumerate(axes):
        res = margin_results[margin]
        _plot_curve_with_optional_band(ax, r_grid, res[prior_key], lower=None, upper=None, label="Prior predictive", linestyle="--", linewidth=2.6, band_label=None, step=step)
        _plot_curve_with_optional_band(ax, r_grid, res[posterior_key], lower=None, upper=None, label="Posterior predictive", linestyle="-", linewidth=2.8, band_label=None, step=step)
        ax.set_xlim(0.0, 1.0)
        if fixed_ylim is None:
            ax.set_ylim(-0.02 * safe_y_max, safe_y_max)
        else:
            ax.set_ylim(*fixed_ylim)
        _format_axis(ax, xlabel=xlabel, ylabel=ylabel if margin == 0 else "", title=f"Group {margin + 1}")
        ax.legend(fontsize=11, frameon=False, loc="lower right")

    fig.tight_layout()
    return _save_figure(fig, save_dir, file_name, save_pdf=save_pdf, dpi=dpi)


def _posterior_cumhaz_derivative_approximation(sample_times, anchor_times, r_grid, delta_values, *, survival_floor):
    sample_times = np.asarray(sample_times, dtype=float)
    sample_times = sample_times[np.isfinite(sample_times)]
    anchor_times = np.asarray(anchor_times, dtype=float)
    anchor_times = anchor_times[np.isfinite(anchor_times)]
    if sample_times.size == 0:
        raise ValueError("Cannot compute cumulative-hazard derivative approximation from empty sample.")
    if anchor_times.size == 0:
        raise ValueError("Cannot compute cumulative-hazard derivative approximation with empty anchor times.")

    anchor_hazards = _cumhaz_from_survival(np.mean(sample_times[:, None] > anchor_times[None, :], axis=0), survival_floor)
    safe_r_grid = np.asarray(r_grid, dtype=float)
    curves = {}
    for delta in delta_values:
        if delta <= 0.0:
            raise ValueError("delta_values must be positive.")
        shifted_points = np.maximum(0.0, anchor_times[:, None] + safe_r_grid[None, :] * float(delta))
        shifted_hazards = _cumhaz_from_survival(
            np.mean(sample_times[:, None, None] > shifted_points[None, :, :], axis=0),
            survival_floor,
        )
        denominator = float(delta) * safe_r_grid[None, :]
        quotients = np.full_like(shifted_hazards, np.nan, dtype=float)
        valid_mask = np.abs(denominator) > 0.0
        np.divide(shifted_hazards - anchor_hazards[:, None], denominator, out=quotients, where=valid_mask)
        column_counts = np.sum(valid_mask, axis=0)
        column_sums = np.sum(np.where(valid_mask, quotients, 0.0), axis=0)
        curves[float(delta)] = np.divide(column_sums, column_counts, out=np.full_like(column_sums, np.nan), where=column_counts > 0)
    return curves


def _cumhaz_derivative_approximation_from_survival(x_grid, survival_values, anchor_times, r_grid, delta_values, *, survival_floor):
    x_grid = np.asarray(x_grid, dtype=float)
    survival_values = np.asarray(survival_values, dtype=float)
    anchor_times = np.asarray(anchor_times, dtype=float)
    anchor_times = anchor_times[np.isfinite(anchor_times)]
    if anchor_times.size == 0:
        raise ValueError("Cannot compute cumulative-hazard derivative approximation with empty anchor times.")

    anchor_hazards = _cumhaz_from_survival(_survival_from_curve(x_grid, survival_values, anchor_times), survival_floor)
    safe_r_grid = np.asarray(r_grid, dtype=float)
    curves = {}
    for delta in delta_values:
        if delta <= 0.0:
            raise ValueError("delta_values must be positive.")
        shifted_points = np.maximum(0.0, anchor_times[:, None] + safe_r_grid[None, :] * float(delta))
        shifted_hazards = _cumhaz_from_survival(_survival_from_curve(x_grid, survival_values, shifted_points), survival_floor)
        denominator = float(delta) * safe_r_grid[None, :]
        quotients = np.full_like(shifted_hazards, np.nan, dtype=float)
        valid_mask = np.abs(denominator) > 0.0
        np.divide(shifted_hazards - anchor_hazards[:, None], denominator, out=quotients, where=valid_mask)
        column_counts = np.sum(valid_mask, axis=0)
        column_sums = np.sum(np.where(valid_mask, quotients, 0.0), axis=0)
        curves[float(delta)] = np.divide(column_sums, column_counts, out=np.full_like(column_sums, np.nan), where=column_counts > 0)
    return curves


def _hazard_estimate_from_cumhaz_curve(x_grid, cumhaz_values, r_grid, delta):
    x_grid = np.asarray(x_grid, dtype=float)
    cumhaz_values = np.asarray(cumhaz_values, dtype=float)
    r_grid = np.asarray(r_grid, dtype=float)

    if x_grid.ndim != 1 or cumhaz_values.ndim != 1 or x_grid.shape != cumhaz_values.shape:
        raise ValueError("x_grid and cumhaz_values must be one-dimensional arrays of the same shape.")
    if np.any(np.diff(x_grid) < 0.0):
        raise ValueError("x_grid must be nondecreasing.")

    delta = float(delta)
    if delta <= 0.0:
        raise ValueError("delta must be positive.")
    if np.any(r_grid < 0.0):
        raise ValueError("r_grid must be nonnegative.")

    left_points = np.maximum(0.0, r_grid - delta)
    right_points = r_grid + delta
    left_h = np.interp(left_points, x_grid, cumhaz_values, left=float(cumhaz_values[0]), right=float(cumhaz_values[-1]))
    right_h = np.interp(right_points, x_grid, cumhaz_values, left=float(cumhaz_values[0]), right=float(cumhaz_values[-1]))
    return (right_h - left_h) / (2.0 * delta)


def _plot_posterior_cumhaz_derivative_approximation(
    diagnostics,
    save_dir: Path,
    *,
    file_prefix: str,
    delta_values,
    derivative_r_grid,
    save_pdf: bool,
    dpi: int,
):
    posterior_samples = np.asarray(diagnostics["posterior_samples"], dtype=np.float64)
    survival_floor = float(diagnostics["survival_floor"])
    r_grid = np.asarray(derivative_r_grid, dtype=np.float64)
    prior_sample_size = 1000
    zero_mask = np.isclose(r_grid, 0.0)
    y_cap = 10.0
    panel_curves = []

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8), sharey=True)
    for margin, ax in enumerate(axes):
        res = diagnostics["margin_results"][margin]
        anchor_times = np.asarray(res["observed"], dtype=np.float64)
        posterior_curves = _posterior_cumhaz_derivative_approximation(
            posterior_samples[:, margin],
            anchor_times,
            r_grid,
            delta_values,
            survival_floor=survival_floor,
        )
        prior_samples = _inverse_transform_sample_from_survival(
            diagnostics["t_grid"],
            res["prior_survival"],
            prior_sample_size,
            rng_seed=9100 + margin,
        )
        prior_curves = _posterior_cumhaz_derivative_approximation(
            prior_samples,
            anchor_times,
            r_grid,
            delta_values,
            survival_floor=survival_floor,
        )
        for delta, values in posterior_curves.items():
            plot_values = np.asarray(values, dtype=float).copy()
            plot_values[zero_mask] = np.nan
            panel_curves.append(plot_values)
            displayed_values = plot_values.copy()
            displayed_values[displayed_values > y_cap] = np.nan
            ax.plot(r_grid, displayed_values, linewidth=2.5, label=fr"Posterior, $D={delta:g}$")
            prior_values = np.asarray(prior_curves[delta], dtype=float).copy()
            prior_values[zero_mask] = np.nan
            panel_curves.append(prior_values)
            displayed_prior_values = prior_values.copy()
            displayed_prior_values[displayed_prior_values > y_cap] = np.nan
            ax.plot(r_grid, displayed_prior_values, linewidth=2.2, linestyle=":", label=fr"Prior, $D={delta:g}$")
        ax.axhline(0.0, color="0.35", linewidth=1.0, linestyle="--", zorder=1)
        ax.set_xlim(float(r_grid[0]), min(5.0, float(r_grid[-1])))
        _format_axis(ax, xlabel=r"Scaled offset $r$", ylabel=r"$\frac{H_i(x + rD) - H_i(x)}{Dr}$" if margin == 0 else "", title=f"Group {margin + 1}")
        ax.legend(fontsize=11, frameon=False, loc="best")

    finite_parts = []
    for curve in panel_curves:
        finite = np.asarray(curve, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            finite_parts.append(finite)
    y_lower = -0.25
    if finite_parts:
        y_lower = min(-0.25, 1.05 * float(np.min(np.concatenate(finite_parts))))
    for ax in axes:
        ax.set_ylim(y_lower, y_cap)

    fig.tight_layout()
    return _save_figure(fig, save_dir, f"{file_prefix}_posterior_cumhaz_derivative_approx", save_pdf=save_pdf, dpi=dpi)


def _plot_fixed_anchor_cumhaz_derivative_approximation(
    diagnostics,
    save_dir: Path,
    *,
    file_prefix: str,
    fixed_anchor: float,
    delta_values,
    derivative_r_grid,
    save_pdf: bool,
    dpi: int,
):
    posterior_samples = np.asarray(diagnostics["posterior_samples"], dtype=np.float64)
    survival_floor = float(diagnostics["survival_floor"])
    r_grid = np.asarray(derivative_r_grid, dtype=np.float64)
    prior_sample_size = 1000
    zero_mask = np.isclose(r_grid, 0.0)
    y_cap = 2.0
    panel_curves = []

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8), sharey=True)
    for margin, ax in enumerate(axes):
        res = diagnostics["margin_results"][margin]
        anchor_times = np.asarray([max(0.0, float(fixed_anchor))], dtype=np.float64)
        posterior_curves = _posterior_cumhaz_derivative_approximation(
            posterior_samples[:, margin],
            anchor_times,
            r_grid,
            delta_values,
            survival_floor=survival_floor,
        )
        prior_samples = _inverse_transform_sample_from_survival(
            diagnostics["t_grid"],
            res["prior_survival"],
            prior_sample_size,
            rng_seed=9200 + margin,
        )
        prior_curves = _posterior_cumhaz_derivative_approximation(
            prior_samples,
            anchor_times,
            r_grid,
            delta_values,
            survival_floor=survival_floor,
        )
        for delta, values in posterior_curves.items():
            plot_values = np.asarray(values, dtype=float).copy()
            plot_values[zero_mask] = np.nan
            panel_curves.append(plot_values)
            displayed_values = plot_values.copy()
            displayed_values[displayed_values > y_cap] = np.nan
            ax.plot(r_grid, displayed_values, linewidth=2.5, label=fr"Posterior, $D={delta:g}$")
            prior_values = np.asarray(prior_curves[delta], dtype=float).copy()
            prior_values[zero_mask] = np.nan
            panel_curves.append(prior_values)
            displayed_prior_values = prior_values.copy()
            displayed_prior_values[displayed_prior_values > y_cap] = np.nan
            ax.plot(r_grid, displayed_prior_values, linewidth=2.2, linestyle=":", label=fr"Prior, $D={delta:g}$")
        ax.axhline(0.0, color="0.35", linewidth=1.0, linestyle="--", zorder=1)
        ax.set_xlim(float(r_grid[0]), min(5.0, float(r_grid[-1])))
        _format_axis(
            ax,
            xlabel=r"Scaled offset $r$",
            ylabel=r"$\frac{H_i(x + rD) - H_i(x)}{Dr}$" if margin == 0 else "",
            title=f"Group {margin + 1}, anchor $x={float(fixed_anchor):g}$",
        )
        ax.legend(fontsize=11, frameon=False, loc="best")

    finite_parts = []
    for curve in panel_curves:
        finite = np.asarray(curve, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            finite_parts.append(finite)
    y_lower = -0.25
    if finite_parts:
        y_lower = min(-0.25, 1.05 * float(np.min(np.concatenate(finite_parts))))
    for ax in axes:
        ax.set_ylim(y_lower, y_cap)

    fig.tight_layout()
    return _save_figure(fig, save_dir, f"{file_prefix}_posterior_cumhaz_derivative_fixed_anchor", save_pdf=save_pdf, dpi=dpi)


def _plot_shifted_anchor_cumhaz_derivative_collection(
    diagnostics,
    save_dir: Path,
    *,
    file_prefix: str,
    derivative_r_grid,
    anchor_shifts,
    fixed_delta: float,
    suffix: str,
    xlabel: str,
    save_pdf: bool,
    dpi: int,
):
    posterior_samples = np.asarray(diagnostics["posterior_samples"], dtype=np.float64)
    survival_floor = float(diagnostics["survival_floor"])
    r_grid = np.asarray(derivative_r_grid, dtype=np.float64)
    prior_sample_size = 1000
    zero_mask = np.isclose(r_grid, 0.0)
    y_cap = 10.0
    panel_curves = []

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8), sharey=True)
    for margin, ax in enumerate(axes):
        res = diagnostics["margin_results"][margin]
        observed_anchor_times = np.asarray(res["observed"], dtype=np.float64)
        prior_samples = _inverse_transform_sample_from_survival(
            diagnostics["t_grid"],
            res["prior_survival"],
            prior_sample_size,
            rng_seed=9300 + margin,
        )
        for anchor_shift in anchor_shifts:
            shifted_anchor_times = np.maximum(0.0, observed_anchor_times + float(anchor_shift))
            posterior_curve = _posterior_cumhaz_derivative_approximation(
                posterior_samples[:, margin],
                shifted_anchor_times,
                r_grid,
                (fixed_delta,),
                survival_floor=survival_floor,
            )[float(fixed_delta)]
            prior_curve = _posterior_cumhaz_derivative_approximation(
                prior_samples,
                shifted_anchor_times,
                r_grid,
                (fixed_delta,),
                survival_floor=survival_floor,
            )[float(fixed_delta)]

            plot_values = np.asarray(posterior_curve, dtype=float).copy()
            plot_values[zero_mask] = np.nan
            panel_curves.append(plot_values)
            displayed_values = plot_values.copy()
            displayed_values[displayed_values > y_cap] = np.nan
            ax.plot(r_grid, displayed_values, linewidth=2.5, label=fr"Posterior, $U={anchor_shift:g}$")

            prior_values = np.asarray(prior_curve, dtype=float).copy()
            prior_values[zero_mask] = np.nan
            panel_curves.append(prior_values)
            displayed_prior_values = prior_values.copy()
            displayed_prior_values[displayed_prior_values > y_cap] = np.nan
            ax.plot(r_grid, displayed_prior_values, linewidth=2.2, linestyle=":", label=fr"Prior, $U={anchor_shift:g}$")

        ax.axhline(0.0, color="0.35", linewidth=1.0, linestyle="--", zorder=1)
        ax.set_xlim(float(r_grid[0]), float(r_grid[-1]))
        _format_axis(ax, xlabel=xlabel, ylabel=fr"$\frac{{H_i(x + U + rD) - H_i(x + U)}}{{Dr}}$" if margin == 0 else "", title=f"Group {margin + 1}")
        ax.legend(fontsize=10, frameon=False, loc="best")

    finite_parts = []
    for curve in panel_curves:
        finite = np.asarray(curve, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            finite_parts.append(finite)
    y_lower = -0.25
    if finite_parts:
        y_lower = min(-0.25, 1.05 * float(np.min(np.concatenate(finite_parts))))
    for ax in axes:
        ax.set_ylim(y_lower, y_cap)

    fig.tight_layout()
    return _save_figure(fig, save_dir, f"{file_prefix}_{suffix}", save_pdf=save_pdf, dpi=dpi)


def _plot_hazard_rate_estimate_from_cumhaz(
    diagnostics,
    save_dir: Path,
    *,
    file_prefix: str,
    hazard_delta_values,
    hazard_r_grid,
    save_pdf: bool,
    dpi: int,
):
    t_grid = np.asarray(diagnostics["t_grid"], dtype=np.float64)
    delta_values = tuple(float(delta) for delta in hazard_delta_values)
    r_grid = np.asarray(hazard_r_grid, dtype=np.float64)
    if r_grid.ndim != 1 or r_grid.size < 2:
        raise ValueError("hazard_r_grid must be a one-dimensional array with at least two points.")

    panel_curves = []

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8), sharey=True)
    for margin, ax in enumerate(axes):
        res = diagnostics["margin_results"][margin]
        for delta in delta_values:
            posterior_curve = _hazard_estimate_from_cumhaz_curve(
                t_grid,
                res["posterior_cumulative_hazard"],
                r_grid,
                delta,
            )
            prior_curve = _hazard_estimate_from_cumhaz_curve(
                t_grid,
                res["prior_cumulative_hazard"],
                r_grid,
                delta,
            )

            posterior_curve = np.asarray(posterior_curve, dtype=float)
            prior_curve = np.asarray(prior_curve, dtype=float)
            panel_curves.extend([posterior_curve, prior_curve])

            ax.plot(r_grid, posterior_curve, linewidth=2.5, label=fr"Posterior, $D={delta:g}$")
            ax.plot(r_grid, prior_curve, linewidth=2.2, linestyle=":", label=fr"Prior, $D={delta:g}$")

        ax.set_xlim(float(r_grid[0]), min(5.0, float(r_grid[-1])))
        ax.axhline(0.0, color="0.35", linewidth=1.0, linestyle="--", zorder=1)
        _format_axis(
            ax,
            xlabel=r"Location $r$",
            ylabel=r"$\frac{H_i(r + D) - H_i(\max(r - D, 0))}{2D}$" if margin == 0 else "",
            title=f"Group {margin + 1}",
        )
        ax.legend(fontsize=10, frameon=False, loc="best")

    finite_parts = []
    for curve in panel_curves:
        finite = np.asarray(curve, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            finite_parts.append(finite)

    if finite_parts:
        combined = np.concatenate(finite_parts)
        y_lower = min(-0.05, 1.05 * float(np.nanmin(combined)))
        y_upper = max(0.25, 1.05 * float(np.nanpercentile(combined, 99.0)))
    else:
        y_lower, y_upper = -0.05, 1.0

    if not np.isfinite(y_lower) or not np.isfinite(y_upper) or y_upper <= y_lower:
        y_lower, y_upper = -0.05, 1.0

    for ax in axes:
        ax.set_ylim(y_lower, y_upper)

    fig.tight_layout()
    return _save_figure(fig, save_dir, f"{file_prefix}_hazard_rate_estimate", save_pdf=save_pdf, dpi=dpi)


def create_prior_posterior_separate_diagnostics(
    prior_predictive_samples,
    posterior_predictive_samples,
    observed_event_times_by_group,
    save_dir,
    *,
    group_labels=(r"$T_1$", r"$T_2$"),
    file_prefix="prior_posterior_diagnostics",
    n_time_grid=500,
    n_radius_grid=300,
    max_radius=None,
    fixed_radius=None,
    confidence_level=0.90,
    show_dkw_bands=True,
    survival_floor=None,
    save_pdf=True,
    dpi=300,
):
    _setup_matplotlib()

    prior_reference = _coerce_prior_reference(prior_predictive_samples)
    post = _to_bivariate_array(posterior_predictive_samples, "posterior_predictive_samples")
    obs = _coerce_observed(observed_event_times_by_group)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_values_parts = [post.reshape(-1), obs[0], obs[1]]
    prior = None
    if prior_reference["kind"] == "samples":
        prior = prior_reference["value"]
        all_values_parts.insert(0, prior.reshape(-1))
    else:
        for x_grid, _ in prior_reference["value"]:
            all_values_parts.append(np.asarray(x_grid, dtype=float))

    all_values = np.concatenate(all_values_parts)
    all_values = all_values[np.isfinite(all_values)]
    t_min = max(0.0, float(np.min(all_values)))
    t_max = float(np.max(all_values))
    if t_max <= t_min:
        t_min -= 0.5
        t_max += 0.5

    time_range = t_max - t_min
    pad = 0.05 * time_range
    t_grid = np.linspace(max(0.0, t_min - pad), t_max + pad, n_time_grid)

    if max_radius is None:
        max_radius = 0.15 * time_range
    if fixed_radius is None:
        fixed_radius = 0.05 * time_range
    r_grid = np.linspace(0.0, max_radius, n_radius_grid)

    if survival_floor is None:
        survival_floor = 0.5 / max(post.shape[0], 1)

    prior_means = np.zeros(2, dtype=np.float64)
    posterior_means = np.mean(post, axis=0)
    empirical_means = np.array([np.mean(obs[0]), np.mean(obs[1])], dtype=np.float64)

    margin_results = []
    summaries = {}

    for margin in range(2):
        obs_j = obs[margin]
        other_obs_j = obs[1 - margin]
        post_j = post[:, margin]
        prior_mean_samples = None

        post_surv_scores = _survival_scores(post_j, t_grid)
        post_surv, post_surv_lower, post_surv_upper, post_surv_eps = _dkw_band_from_scores(post_surv_scores, confidence_level=confidence_level, epsilon_multiplier=1.0)

        post_local_scores = _local_mass_scores(post_j, obs_j, r_grid)
        post_local, post_local_lower, post_local_upper, post_local_eps = _dkw_band_from_scores(post_local_scores, confidence_level=confidence_level, epsilon_multiplier=2.0)
        post_local_positive_scores = _one_sided_local_mass_scores(post_j, obs_j, r_grid, direction="positive")
        post_local_positive, _, _, _ = _dkw_band_from_scores(post_local_positive_scores, confidence_level=confidence_level, epsilon_multiplier=2.0)
        post_local_negative_scores = _one_sided_local_mass_scores(post_j, obs_j, r_grid, direction="negative")
        post_local_negative, _, _, _ = _dkw_band_from_scores(post_local_negative_scores, confidence_level=confidence_level, epsilon_multiplier=2.0)
        post_local_positive_normalized_scores = _normalized_one_sided_local_mass_scores(post_j, obs_j, r_grid, direction="positive")
        post_local_positive_normalized, _, _, _ = _dkw_band_from_scores(post_local_positive_normalized_scores, confidence_level=confidence_level, epsilon_multiplier=2.0, clip_to_unit_interval=False)
        post_local_negative_normalized_scores = _normalized_one_sided_local_mass_scores(post_j, obs_j, r_grid, direction="negative")
        post_local_negative_normalized, _, _, _ = _dkw_band_from_scores(post_local_negative_normalized_scores, confidence_level=confidence_level, epsilon_multiplier=2.0, clip_to_unit_interval=False)
        post_local_positive_normalized_cumhaz = _hazard_ratio_one_sided_local_mass_from_samples(post_j, obs_j, r_grid, direction="positive", survival_floor=survival_floor)
        post_local_negative_normalized_cumhaz = _hazard_ratio_one_sided_local_mass_from_samples(post_j, obs_j, r_grid, direction="negative", survival_floor=survival_floor)

        post_nearest_scores, post_nearest_distances = _nearest_distance_scores(post_j, obs_j, r_grid)
        post_nearest, post_nearest_lower, post_nearest_upper, post_nearest_eps = _dkw_band_from_scores(post_nearest_scores, confidence_level=confidence_level, epsilon_multiplier=1.0)
        post_nearest_positive_scores = _one_sided_nearest_scores(post_j, obs_j, r_grid, direction="positive")
        post_nearest_positive, _, _, _ = _dkw_band_from_scores(post_nearest_positive_scores, confidence_level=confidence_level, epsilon_multiplier=1.0)
        post_nearest_negative_scores = _one_sided_nearest_scores(post_j, obs_j, r_grid, direction="negative")
        post_nearest_negative, _, _, _ = _dkw_band_from_scores(post_nearest_negative_scores, confidence_level=confidence_level, epsilon_multiplier=1.0)

        if prior_reference["kind"] == "samples":
            if prior is None:
                raise ValueError("Prior samples were expected but not available.")
            prior_j = prior[:, margin]
            prior_surv_scores = _survival_scores(prior_j, t_grid)
            prior_surv, prior_surv_lower, prior_surv_upper, prior_surv_eps = _dkw_band_from_scores(prior_surv_scores, confidence_level=confidence_level, epsilon_multiplier=1.0)

            prior_local_scores = _local_mass_scores(prior_j, obs_j, r_grid)
            prior_local, prior_local_lower, prior_local_upper, prior_local_eps = _dkw_band_from_scores(prior_local_scores, confidence_level=confidence_level, epsilon_multiplier=2.0)
            prior_local_positive_scores = _one_sided_local_mass_scores(prior_j, obs_j, r_grid, direction="positive")
            prior_local_positive, _, _, _ = _dkw_band_from_scores(prior_local_positive_scores, confidence_level=confidence_level, epsilon_multiplier=2.0)
            prior_local_negative_scores = _one_sided_local_mass_scores(prior_j, obs_j, r_grid, direction="negative")
            prior_local_negative, _, _, _ = _dkw_band_from_scores(prior_local_negative_scores, confidence_level=confidence_level, epsilon_multiplier=2.0)
            prior_local_positive_normalized_scores = _normalized_one_sided_local_mass_scores(prior_j, obs_j, r_grid, direction="positive")
            prior_local_positive_normalized, _, _, _ = _dkw_band_from_scores(prior_local_positive_normalized_scores, confidence_level=confidence_level, epsilon_multiplier=2.0, clip_to_unit_interval=False)
            prior_local_negative_normalized_scores = _normalized_one_sided_local_mass_scores(prior_j, obs_j, r_grid, direction="negative")
            prior_local_negative_normalized, _, _, _ = _dkw_band_from_scores(prior_local_negative_normalized_scores, confidence_level=confidence_level, epsilon_multiplier=2.0, clip_to_unit_interval=False)
            prior_local_positive_normalized_cumhaz = _hazard_ratio_one_sided_local_mass_from_samples(prior_j, obs_j, r_grid, direction="positive", survival_floor=survival_floor)
            prior_local_negative_normalized_cumhaz = _hazard_ratio_one_sided_local_mass_from_samples(prior_j, obs_j, r_grid, direction="negative", survival_floor=survival_floor)

            prior_nearest_scores, prior_nearest_distances = _nearest_distance_scores(prior_j, obs_j, r_grid)
            prior_nearest, prior_nearest_lower, prior_nearest_upper, prior_nearest_eps = _dkw_band_from_scores(prior_nearest_scores, confidence_level=confidence_level, epsilon_multiplier=1.0)
            prior_nearest_positive_scores = _one_sided_nearest_scores(prior_j, obs_j, r_grid, direction="positive")
            prior_nearest_positive, _, _, _ = _dkw_band_from_scores(prior_nearest_positive_scores, confidence_level=confidence_level, epsilon_multiplier=1.0)
            prior_nearest_negative_scores = _one_sided_nearest_scores(prior_j, obs_j, r_grid, direction="negative")
            prior_nearest_negative, _, _, _ = _dkw_band_from_scores(prior_nearest_negative_scores, confidence_level=confidence_level, epsilon_multiplier=1.0)
            prior_mean_nearest_distance = float(np.mean(prior_nearest_distances))
        else:
            prior_x_grid, prior_survival_values = prior_reference["value"][margin]
            prior_surv = _survival_from_curve(prior_x_grid, prior_survival_values, t_grid)
            prior_surv_lower = None
            prior_surv_upper = None
            prior_surv_eps = None

            prior_local = _local_mass_curve_from_survival(prior_x_grid, prior_survival_values, obs_j, r_grid)
            prior_local_lower = None
            prior_local_upper = None
            prior_local_eps = None
            prior_local_positive = _one_sided_local_mass_curve_from_survival(prior_x_grid, prior_survival_values, obs_j, r_grid, direction="positive")
            prior_local_negative = _one_sided_local_mass_curve_from_survival(prior_x_grid, prior_survival_values, obs_j, r_grid, direction="negative")
            prior_local_positive_normalized = _normalized_one_sided_local_mass_curve_from_survival(prior_x_grid, prior_survival_values, obs_j, r_grid, direction="positive")
            prior_local_negative_normalized = _normalized_one_sided_local_mass_curve_from_survival(prior_x_grid, prior_survival_values, obs_j, r_grid, direction="negative")
            prior_local_positive_normalized_cumhaz = _hazard_ratio_one_sided_local_mass_from_survival(prior_x_grid, prior_survival_values, obs_j, r_grid, direction="positive", survival_floor=survival_floor)
            prior_local_negative_normalized_cumhaz = _hazard_ratio_one_sided_local_mass_from_survival(prior_x_grid, prior_survival_values, obs_j, r_grid, direction="negative", survival_floor=survival_floor)

            prior_nearest = _distance_cdf_from_survival(prior_x_grid, prior_survival_values, obs_j, r_grid)
            prior_nearest_lower = None
            prior_nearest_upper = None
            prior_nearest_eps = None
            prior_nearest_positive = _one_sided_distance_cdf_from_survival(prior_x_grid, prior_survival_values, obs_j, r_grid, direction="positive")
            prior_nearest_negative = _one_sided_distance_cdf_from_survival(prior_x_grid, prior_survival_values, obs_j, r_grid, direction="negative")
            prior_mean_nearest_distance = None
            prior_mean_samples = _inverse_transform_sample_from_survival(
                prior_x_grid,
                prior_survival_values,
                post.shape[0],
                rng_seed=1234 + margin,
            )

        if prior_reference["kind"] == "samples":
            if prior is None:
                raise ValueError("Prior samples were expected but not available.")
            prior_means[margin] = float(np.mean(prior[:, margin]))
        else:
            if prior_mean_samples is None:
                raise ValueError("Prior mean samples were expected but not available.")
            prior_means[margin] = float(np.mean(prior_mean_samples))

        post_cumhaz = _cumhaz_from_survival(post_surv, survival_floor)
        post_cumhaz_lower, post_cumhaz_upper = _cumhaz_band_from_survival_band(post_surv_lower, post_surv_upper, survival_floor)

        prior_cumhaz = _cumhaz_from_survival(prior_surv, survival_floor)
        prior_cumhaz_lower, prior_cumhaz_upper = _cumhaz_band_from_survival_band(prior_surv_lower, prior_surv_upper, survival_floor)

        fixed_idx = int(np.argmin(np.abs(r_grid - fixed_radius)))
        summaries[f"margin_{margin + 1}"] = {
            "fixed_radius": float(fixed_radius),
            "confidence_level": float(confidence_level),
            "posterior_survival_dkw_epsilon": float(post_surv_eps),
            "posterior_local_mass_dkw_epsilon": float(post_local_eps),
            "posterior_nearest_event_dkw_epsilon": float(post_nearest_eps),
            "prior_survival_dkw_epsilon": None if prior_surv_eps is None else float(prior_surv_eps),
            "prior_local_mass_dkw_epsilon": None if prior_local_eps is None else float(prior_local_eps),
            "prior_nearest_event_dkw_epsilon": None if prior_nearest_eps is None else float(prior_nearest_eps),
            "prior_average_local_mass": float(prior_local[fixed_idx]),
            "posterior_average_local_mass": float(post_local[fixed_idx]),
            "posterior_minus_prior_average_local_mass": float(post_local[fixed_idx] - prior_local[fixed_idx]),
            "prior_average_local_mass_positive": float(prior_local_positive[fixed_idx]),
            "posterior_average_local_mass_positive": float(post_local_positive[fixed_idx]),
            "posterior_minus_prior_average_local_mass_positive": float(post_local_positive[fixed_idx] - prior_local_positive[fixed_idx]),
            "prior_average_local_mass_negative": float(prior_local_negative[fixed_idx]),
            "posterior_average_local_mass_negative": float(post_local_negative[fixed_idx]),
            "posterior_minus_prior_average_local_mass_negative": float(post_local_negative[fixed_idx] - prior_local_negative[fixed_idx]),
            "prior_average_local_mass_positive_normalized": float(prior_local_positive_normalized[fixed_idx]),
            "posterior_average_local_mass_positive_normalized": float(post_local_positive_normalized[fixed_idx]),
            "posterior_minus_prior_average_local_mass_positive_normalized": float(post_local_positive_normalized[fixed_idx] - prior_local_positive_normalized[fixed_idx]),
            "prior_average_local_mass_negative_normalized": float(prior_local_negative_normalized[fixed_idx]),
            "posterior_average_local_mass_negative_normalized": float(post_local_negative_normalized[fixed_idx]),
            "posterior_minus_prior_average_local_mass_negative_normalized": float(post_local_negative_normalized[fixed_idx] - prior_local_negative_normalized[fixed_idx]),
            "prior_probability_near_any_observed_event": float(prior_nearest[fixed_idx]),
            "posterior_probability_near_any_observed_event": float(post_nearest[fixed_idx]),
            "posterior_minus_prior_probability_near_any_observed_event": float(post_nearest[fixed_idx] - prior_nearest[fixed_idx]),
            "prior_probability_near_any_observed_event_positive": float(prior_nearest_positive[fixed_idx]),
            "posterior_probability_near_any_observed_event_positive": float(post_nearest_positive[fixed_idx]),
            "posterior_minus_prior_probability_near_any_observed_event_positive": float(post_nearest_positive[fixed_idx] - prior_nearest_positive[fixed_idx]),
            "prior_probability_near_any_observed_event_negative": float(prior_nearest_negative[fixed_idx]),
            "posterior_probability_near_any_observed_event_negative": float(post_nearest_negative[fixed_idx]),
            "posterior_minus_prior_probability_near_any_observed_event_negative": float(post_nearest_negative[fixed_idx] - prior_nearest_negative[fixed_idx]),
            "prior_mean_distance_to_nearest_observed_event": None if prior_mean_nearest_distance is None else float(prior_mean_nearest_distance),
            "posterior_mean_distance_to_nearest_observed_event": float(np.mean(post_nearest_distances)),
            "prior_predictive_mean": float(prior_means[margin]),
            "posterior_predictive_mean": float(posterior_means[margin]),
        }

        margin_results.append({
            "observed": obs_j,
            "other_observed": other_obs_j,
            "prior_survival": prior_surv,
            "prior_survival_lower": prior_surv_lower,
            "prior_survival_upper": prior_surv_upper,
            "posterior_survival": post_surv,
            "posterior_survival_lower": post_surv_lower,
            "posterior_survival_upper": post_surv_upper,
            "prior_cumulative_hazard": prior_cumhaz,
            "prior_cumulative_hazard_lower": prior_cumhaz_lower,
            "prior_cumulative_hazard_upper": prior_cumhaz_upper,
            "posterior_cumulative_hazard": post_cumhaz,
            "posterior_cumulative_hazard_lower": post_cumhaz_lower,
            "posterior_cumulative_hazard_upper": post_cumhaz_upper,
            "prior_local_mass": prior_local,
            "prior_local_mass_lower": prior_local_lower,
            "prior_local_mass_upper": prior_local_upper,
            "posterior_local_mass": post_local,
            "posterior_local_mass_lower": post_local_lower,
            "posterior_local_mass_upper": post_local_upper,
                "prior_local_mass_positive": prior_local_positive,
                "posterior_local_mass_positive": post_local_positive,
                "prior_local_mass_negative": prior_local_negative,
                "posterior_local_mass_negative": post_local_negative,
                "prior_local_mass_positive_normalized": prior_local_positive_normalized,
                "posterior_local_mass_positive_normalized": post_local_positive_normalized,
                "prior_local_mass_negative_normalized": prior_local_negative_normalized,
                "posterior_local_mass_negative_normalized": post_local_negative_normalized,
                "prior_local_mass_positive_normalized_cumhaz": prior_local_positive_normalized_cumhaz,
                "posterior_local_mass_positive_normalized_cumhaz": post_local_positive_normalized_cumhaz,
                "prior_local_mass_negative_normalized_cumhaz": prior_local_negative_normalized_cumhaz,
                "posterior_local_mass_negative_normalized_cumhaz": post_local_negative_normalized_cumhaz,
            "prior_nearest_cdf": prior_nearest,
            "prior_nearest_cdf_lower": prior_nearest_lower,
            "prior_nearest_cdf_upper": prior_nearest_upper,
            "posterior_nearest_cdf": post_nearest,
            "posterior_nearest_cdf_lower": post_nearest_lower,
            "posterior_nearest_cdf_upper": post_nearest_upper,
                "prior_nearest_cdf_positive": prior_nearest_positive,
                "posterior_nearest_cdf_positive": post_nearest_positive,
                "prior_nearest_cdf_negative": prior_nearest_negative,
                "posterior_nearest_cdf_negative": post_nearest_negative,
            "prior_mean": float(prior_means[margin]),
            "posterior_mean": float(posterior_means[margin]),
                    "empirical_mean": float(empirical_means[margin]),
        })

    saved_paths = {}
    x_left, x_right, cumhaz_mask = _suggest_time_window(t_grid, post, obs)

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8), sharey=True)
    for margin, ax in enumerate(axes):
        res = margin_results[margin]
        style = GROUP_STYLES[margin]
        prior_band_label = f"Prior {int(confidence_level * 100)}% DKW band" if prior_reference["kind"] == "samples" else None
        _plot_curve_with_optional_band(ax, t_grid, res["prior_survival"], lower=res["prior_survival_lower"] if show_dkw_bands else None, upper=res["prior_survival_upper"] if show_dkw_bands else None, label="Prior predictive", linestyle="--", linewidth=2.6, band_label=prior_band_label, step=True)
        ax.lines[-1].set_color(style["light"])
        _plot_curve_with_optional_band(ax, t_grid, res["posterior_survival"], lower=res["posterior_survival_lower"] if show_dkw_bands else None, upper=res["posterior_survival_upper"] if show_dkw_bands else None, label="Posterior predictive", linestyle="-", linewidth=2.8, band_label=f"Posterior {int(confidence_level * 100)}% DKW band", step=True)
        ax.lines[-1].set_color(style["color"])
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(x_left, x_right)
        _add_group_observation_lines(ax, obs, margin)
        _add_mean_reference_lines(
            ax,
            res["prior_mean"],
            res["posterior_mean"],
            color=style["color"],
            empirical_mean=res["empirical_mean"],
            label_empirical=True,
            prior_linestyle=(0, (1.0, 2.2)),
            posterior_linestyle=(0, (9.0, 2.6, 1.8, 2.6)),
            empirical_linestyle=(0, (16.0, 3.2)),
        )
        _format_axis(ax, xlabel=f"Time {group_labels[margin]}", ylabel="Survival probability" if margin == 0 else "", title=f"Group {margin + 1}")

    fig.tight_layout()
    saved_paths["survival"] = _save_figure(fig, save_dir, f"{file_prefix}_survival", save_pdf=save_pdf, dpi=dpi)

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8), sharey=True)
    all_h_values = []
    for res in margin_results:
        all_h_values.append(res["prior_cumulative_hazard"])
        all_h_values.append(res["posterior_cumulative_hazard"])
        if res["prior_cumulative_hazard_upper"] is not None:
            all_h_values.append(res["prior_cumulative_hazard_upper"])
        if res["posterior_cumulative_hazard_upper"] is not None:
            all_h_values.append(res["posterior_cumulative_hazard_upper"])

    all_h_values = np.concatenate([values[cumhaz_mask][np.isfinite(values[cumhaz_mask])] for values in all_h_values])
    h_max = max(float(np.nanpercentile(all_h_values, 99.0)), 1.0)
    for margin, ax in enumerate(axes):
        res = margin_results[margin]
        style = GROUP_STYLES[margin]
        prior_band_label = f"Prior {int(confidence_level * 100)}% DKW band" if prior_reference["kind"] == "samples" else None
        _plot_curve_with_optional_band(ax, t_grid, res["prior_cumulative_hazard"], lower=res["prior_cumulative_hazard_lower"] if show_dkw_bands else None, upper=res["prior_cumulative_hazard_upper"] if show_dkw_bands else None, label="Prior predictive", linestyle="--", linewidth=2.6, band_label=prior_band_label, step=True)
        ax.lines[-1].set_color(style["light"])
        _plot_curve_with_optional_band(ax, t_grid, res["posterior_cumulative_hazard"], lower=res["posterior_cumulative_hazard_lower"] if show_dkw_bands else None, upper=res["posterior_cumulative_hazard_upper"] if show_dkw_bands else None, label="Posterior predictive", linestyle="-", linewidth=2.8, band_label=f"Posterior {int(confidence_level * 100)}% DKW band", step=True)
        ax.lines[-1].set_color(style["color"])
        ax.set_xlim(x_left, x_right)
        ax.set_ylim(-0.02 * h_max, 1.08 * h_max)
        _add_group_observation_lines(ax, obs, margin)
        _format_axis(ax, xlabel=f"Time {group_labels[margin]}", ylabel="Cumulative hazard" if margin == 0 else "", title=f"Group {margin + 1}")
        ax.legend(fontsize=11, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.14)
    saved_paths["cumulative_hazard"] = _save_figure(fig, save_dir, f"{file_prefix}_cumulative_hazard", save_pdf=save_pdf, dpi=dpi)

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8), sharey=True)
    for margin, ax in enumerate(axes):
        res = margin_results[margin]
        _plot_curve_with_optional_band(ax, r_grid, res["prior_local_mass"], lower=None, upper=None, label="Prior predictive", linestyle="--", linewidth=2.6, band_label=None, step=False)
        _plot_curve_with_optional_band(ax, r_grid, res["posterior_local_mass"], lower=None, upper=None, label="Posterior predictive", linestyle="-", linewidth=2.8, band_label=None, step=False)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.02, 1.02)
        _format_axis(ax, xlabel=r"Window radius $r$", ylabel=r"$L_I(r)$" if margin == 0 else "", title=f"Group {margin + 1}")
        ax.legend(fontsize=11, frameon=False, loc="lower right")

    fig.tight_layout()
    saved_paths["local_mass"] = _save_figure(fig, save_dir, f"{file_prefix}_local_mass", save_pdf=save_pdf, dpi=dpi)
    saved_paths["local_mass_positive"] = _plot_two_margin_directional_diagnostic(
        margin_results,
        r_grid,
        save_dir,
        prior_key="prior_local_mass_positive",
        posterior_key="posterior_local_mass_positive",
        xlabel=r"Positive deviation radius $r$",
        ylabel=r"$L_i^{+}(r)$",
        file_name=f"{file_prefix}_local_mass_positive",
        step=False,
        save_pdf=save_pdf,
        dpi=dpi,
    )
    saved_paths["local_mass_negative"] = _plot_two_margin_directional_diagnostic(
        margin_results,
        r_grid,
        save_dir,
        prior_key="prior_local_mass_negative",
        posterior_key="posterior_local_mass_negative",
        xlabel=r"Negative deviation radius $r$",
        ylabel=r"$L_i^{-}(r)$",
        file_name=f"{file_prefix}_local_mass_negative",
        step=False,
        save_pdf=save_pdf,
        dpi=dpi,
    )
    saved_paths["local_mass_positive_normalized"] = _plot_two_margin_directional_diagnostic(
        margin_results,
        r_grid,
        save_dir,
        prior_key="prior_local_mass_positive_normalized",
        posterior_key="posterior_local_mass_positive_normalized",
        xlabel=r"Positive deviation radius $r$",
        ylabel=r"$L_{i,\mathrm{norm}}^{+}(r)$",
        file_name=f"{file_prefix}_local_mass_positive_normalized",
        step=False,
        fixed_ylim=None,
        save_pdf=save_pdf,
        dpi=dpi,
    )
    saved_paths["local_mass_negative_normalized"] = _plot_two_margin_directional_diagnostic(
        margin_results,
        r_grid,
        save_dir,
        prior_key="prior_local_mass_negative_normalized",
        posterior_key="posterior_local_mass_negative_normalized",
        xlabel=r"Negative deviation radius $r$",
        ylabel=r"$L_{i,\mathrm{norm}}^{-}(r)$",
        file_name=f"{file_prefix}_local_mass_negative_normalized",
        step=False,
        fixed_ylim=None,
        save_pdf=save_pdf,
        dpi=dpi,
    )
    saved_paths["local_mass_positive_normalized_cumhaz"] = _plot_two_margin_directional_diagnostic(
        margin_results,
        r_grid,
        save_dir,
        prior_key="prior_local_mass_positive_normalized_cumhaz",
        posterior_key="posterior_local_mass_positive_normalized_cumhaz",
        xlabel=r"Positive deviation radius $r$",
        ylabel=r"$R_i^{+}(r)$",
        file_name=f"{file_prefix}_local_mass_positive_normalized_cumhaz",
        step=False,
        fixed_ylim=None,
        save_pdf=save_pdf,
        dpi=dpi,
    )
    saved_paths["local_mass_negative_normalized_cumhaz"] = _plot_two_margin_directional_diagnostic(
        margin_results,
        r_grid,
        save_dir,
        prior_key="prior_local_mass_negative_normalized_cumhaz",
        posterior_key="posterior_local_mass_negative_normalized_cumhaz",
        xlabel=r"Negative deviation radius $r$",
        ylabel=r"$R_i^{-}(r)$",
        file_name=f"{file_prefix}_local_mass_negative_normalized_cumhaz",
        step=False,
        fixed_ylim=None,
        save_pdf=save_pdf,
        dpi=dpi,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8), sharey=True)
    for margin, ax in enumerate(axes):
        res = margin_results[margin]
        _plot_curve_with_optional_band(ax, r_grid, res["prior_nearest_cdf"], lower=None, upper=None, label="Prior predictive", linestyle="--", linewidth=2.6, band_label=None, step=True)
        _plot_curve_with_optional_band(ax, r_grid, res["posterior_nearest_cdf"], lower=None, upper=None, label="Posterior predictive", linestyle="-", linewidth=2.8, band_label=None, step=True)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.02, 1.02)
        _format_axis(ax, xlabel=r"Distance $r$", ylabel=r"$M_i(r)$" if margin == 0 else "", title=f"Group {margin + 1}")
        ax.legend(fontsize=11, frameon=False, loc="lower right")

    fig.tight_layout()
    saved_paths["nearest_observed_event"] = _save_figure(fig, save_dir, f"{file_prefix}_nearest_observed_event", save_pdf=save_pdf, dpi=dpi)
    saved_paths["nearest_observed_event_positive"] = _plot_two_margin_directional_diagnostic(
        margin_results,
        r_grid,
        save_dir,
        prior_key="prior_nearest_cdf_positive",
        posterior_key="posterior_nearest_cdf_positive",
        xlabel=r"Positive deviation radius $r$",
        ylabel=r"$M_i^{+}(r)$",
        file_name=f"{file_prefix}_nearest_observed_event_positive",
        step=True,
        save_pdf=save_pdf,
        dpi=dpi,
    )
    saved_paths["nearest_observed_event_negative"] = _plot_two_margin_directional_diagnostic(
        margin_results,
        r_grid,
        save_dir,
        prior_key="prior_nearest_cdf_negative",
        posterior_key="posterior_nearest_cdf_negative",
        xlabel=r"Negative deviation radius $r$",
        ylabel=r"$M_i^{-}(r)$",
        file_name=f"{file_prefix}_nearest_observed_event_negative",
        step=True,
        save_pdf=save_pdf,
        dpi=dpi,
    )

    return {
        "t_grid": t_grid,
        "r_grid": r_grid,
        "fixed_radius": fixed_radius,
        "confidence_level": confidence_level,
        "survival_floor": survival_floor,
        "prior_means": prior_means,
        "posterior_means": posterior_means,
        "empirical_means": empirical_means,
        "saved_paths": saved_paths,
        "summaries": summaries,
        "margin_results": margin_results,
    }


def _load_simulation_study_inputs(results_dir: Path):
    results_dir = Path(results_dir)
    arrays_path = results_dir / "data_study_arrays.npz"
    if not arrays_path.exists():
        raise FileNotFoundError(f"Could not find saved simulation arrays at {arrays_path}")

    with np.load(arrays_path) as loaded:
        required = {
            "data_row0",
            "data_row1",
            "final_min_id_samples",
            "prior_x_grid_0",
            "prior_x_grid_1",
            "prior_exp_measure_0",
            "prior_exp_measure_1",
        }
        missing = required.difference(loaded.files)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Saved simulation arrays are missing: {missing_str}")

        observed_data = [
            np.asarray(loaded["data_row0"], dtype=np.float64),
            np.asarray(loaded["data_row1"], dtype=np.float64),
        ]
        posterior_samples = np.asarray(loaded["final_min_id_samples"], dtype=np.float64)
        prior_survival_curves = [
            (
                np.asarray(loaded["prior_x_grid_0"], dtype=np.float64),
                np.exp(-np.asarray(loaded["prior_exp_measure_0"], dtype=np.float64)),
            ),
            (
                np.asarray(loaded["prior_x_grid_1"], dtype=np.float64),
                np.exp(-np.asarray(loaded["prior_exp_measure_1"], dtype=np.float64)),
            ),
        ]

    return arrays_path, observed_data, posterior_samples, prior_survival_curves


def _add_observation_lines(ax, same_margin, other_margin):
    same_margin = np.asarray(same_margin, dtype=np.float64)
    other_margin = np.asarray(other_margin, dtype=np.float64)

    for idx, value in enumerate(np.sort(other_margin)):
        ax.axvline(value, color="0.65", linewidth=1.0, linestyle="--", alpha=0.12, zorder=1, label=None)

    for idx, value in enumerate(np.sort(same_margin)):
        ax.axvline(value, color="0.15", linewidth=1.1, linestyle="-", alpha=0.16, zorder=2, label=None)


def _masked_ymax(curves, mask, *, quantile=0.995, floor=1.0):
    finite_parts = []
    for curve in curves:
        if curve is None:
            continue
        values = np.asarray(curve, dtype=np.float64)
        if values.ndim == 0:
            continue
        current = values[mask]
        current = current[np.isfinite(current)]
        if current.size:
            finite_parts.append(current)

    if not finite_parts:
        return floor

    joined = np.concatenate(finite_parts)
    return max(float(np.quantile(joined, quantile)), floor)


def _plot_main_text_survival_and_cumulative_hazard(
    diagnostics,
    save_dir: Path,
    *,
    file_prefix: str,
    observed_data,
    save_pdf: bool,
    dpi: int,
    time_quantile: float,
):
    t_grid = np.asarray(diagnostics["t_grid"], dtype=np.float64)
    margin_results = diagnostics["margin_results"]
    posterior_samples = np.asarray(diagnostics["posterior_samples"], dtype=np.float64)

    combined_observations = np.concatenate([
        np.asarray(observed_data[0], dtype=np.float64),
        np.asarray(observed_data[1], dtype=np.float64),
    ])
    x_right = max(float(np.quantile(posterior_samples.reshape(-1), time_quantile)), float(np.max(combined_observations) * 1.1))
    x_right = min(x_right, float(t_grid[-1]))
    x_left = max(0.0, float(t_grid[0]))
    mask = (t_grid >= x_left) & (t_grid <= x_right)

    cumhaz_curves = []
    for res in margin_results:
        cumhaz_curves.extend([
            res["prior_cumulative_hazard"],
            res["posterior_cumulative_hazard"],
            res["prior_cumulative_hazard_upper"],
            res["posterior_cumulative_hazard_upper"],
        ])
    cumhaz_ymax = 1.05 * _masked_ymax(cumhaz_curves, mask, quantile=0.98, floor=1.0)

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), sharex=True)
    for margin, res in enumerate(margin_results):
        survival_ax = axes[0, margin]
        cumhaz_ax = axes[1, margin]

        _add_observation_lines(survival_ax, res["observed"], res["other_observed"])
        _add_observation_lines(cumhaz_ax, res["observed"], res["other_observed"])

        survival_ax.step(t_grid, res["prior_survival"], where="post", color="#1d3557", linestyle="--", linewidth=2.4, label="Prior predictive", zorder=4)
        survival_ax.step(t_grid, res["posterior_survival"], where="post", color="#d97706", linewidth=2.5, label="Posterior predictive", zorder=5)
        survival_ax.fill_between(t_grid, res["posterior_survival_lower"], res["posterior_survival_upper"], step="post", color="#f4a261", alpha=0.23, linewidth=0.0, label=f"Posterior {int(100 * diagnostics['confidence_level'])}% DKW band", zorder=3)
        survival_ax.set_ylim(-0.02, 1.02)
        survival_ax.set_xlim(x_left, x_right)
        _format_axis(survival_ax, xlabel="", ylabel="Survival probability" if margin == 0 else "", title=f"Group {margin + 1}")

        cumhaz_ax.step(t_grid, res["prior_cumulative_hazard"], where="post", color="#1d3557", linestyle="--", linewidth=2.4, label="Prior predictive", zorder=4)
        cumhaz_ax.step(t_grid, res["posterior_cumulative_hazard"], where="post", color="#d97706", linewidth=2.5, label="Posterior predictive", zorder=5)
        cumhaz_ax.fill_between(t_grid, res["posterior_cumulative_hazard_lower"], res["posterior_cumulative_hazard_upper"], step="post", color="#f4a261", alpha=0.23, linewidth=0.0, label=f"Posterior {int(100 * diagnostics['confidence_level'])}% DKW band", zorder=3)
        cumhaz_ax.set_xlim(x_left, x_right)
        cumhaz_ax.set_ylim(-0.02 * cumhaz_ymax, cumhaz_ymax)
        _format_axis(cumhaz_ax, xlabel=f"Time $T_{margin + 1}$", ylabel="Cumulative hazard" if margin == 0 else "")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Posterior Predictive Survival and Cumulative Hazard", fontsize=22, y=1.02)
    fig.text(0.5, 0.02, "The posterior predictive survival function is estimated by the empirical survival function of the posterior predictive draws. The cumulative hazard is the implied transform $H(t)=-\\log S(t)$, and the shaded band is the transformed DKW confidence band.", ha="center", va="bottom", fontsize=10, wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    return _save_figure(fig, save_dir, f"{file_prefix}_maintext_survival_cumhaz", save_pdf=save_pdf, dpi=dpi)


def _plot_group_overlay_survival_and_cumhaz(
    diagnostics,
    save_dir: Path,
    *,
    file_prefix: str,
    observed_data,
    save_pdf: bool,
    dpi: int,
):
    t_grid = np.asarray(diagnostics["t_grid"], dtype=np.float64)
    margin_results = diagnostics["margin_results"]
    posterior_samples = np.asarray(diagnostics["posterior_samples"], dtype=np.float64)
    x_left, x_right, mask = _suggest_time_window(t_grid, posterior_samples, observed_data)

    cumhaz_curves = []
    for res in margin_results:
        cumhaz_curves.extend([
            res["prior_cumulative_hazard"],
            res["posterior_cumulative_hazard"],
            res["prior_cumulative_hazard_upper"],
            res["posterior_cumulative_hazard_upper"],
        ])
    cumhaz_ymax = 1.05 * _masked_ymax(cumhaz_curves, mask, quantile=0.98, floor=1.0)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0))
    shared_prior = margin_results[0]
    axes[0].step(t_grid, shared_prior["prior_survival"], where="post", color="0.05", linestyle="--", linewidth=2.5, label="Prior predictive")
    axes[1].step(t_grid, shared_prior["prior_cumulative_hazard"], where="post", color="0.05", linestyle="--", linewidth=2.5, label="Prior predictive")
    for margin, res in enumerate(margin_results):
        style = GROUP_STYLES[margin]
        label_suffix = f"{style['name']}"

        axes[0].step(t_grid, res["posterior_survival"], where="post", color=style["color"], linewidth=2.5, label=f"{label_suffix} posterior")
        axes[0].fill_between(t_grid, res["posterior_survival_lower"], res["posterior_survival_upper"], step="post", color=style["color"], alpha=0.10, linewidth=0.0)
        _add_mean_reference_lines(
            axes[0],
            res["prior_mean"],
            res["posterior_mean"],
            color=style["color"],
            empirical_mean=res["empirical_mean"],
            label_empirical=False,
            prior_linestyle=(0, (1.0, 2.2)),
            posterior_linestyle=(0, (9.0, 2.6, 1.8, 2.6)),
            empirical_linestyle=(0, (16.0, 3.2)),
        )

        axes[1].step(t_grid, res["posterior_cumulative_hazard"], where="post", color=style["color"], linewidth=2.5, label=f"{label_suffix} posterior")
        axes[1].fill_between(t_grid, res["posterior_cumulative_hazard_lower"], res["posterior_cumulative_hazard_upper"], step="post", color=style["color"], alpha=0.10, linewidth=0.0)

    axes[0].set_xlim(x_left, x_right)
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_xlim(x_left, x_right)
    axes[1].set_ylim(-0.02 * cumhaz_ymax, cumhaz_ymax)
    _format_axis(axes[0], xlabel="Time", ylabel="Survival probability", title="Posterior Predictive Survival")
    _format_axis(axes[1], xlabel="Time", ylabel="Cumulative hazard", title="Posterior Predictive Cumulative Hazard")
    axes[0].legend(frameon=False, fontsize=11, loc="upper right")
    axes[1].legend(frameon=False, fontsize=11, loc="upper left")
    fig.tight_layout()

    return _save_figure(fig, save_dir, f"{file_prefix}_group_overlay_survival_cumhaz", save_pdf=save_pdf, dpi=dpi)


def _plot_appendix_predictive_diagnostics(
    diagnostics,
    save_dir: Path,
    *,
    file_prefix: str,
    save_pdf: bool,
    dpi: int,
):
    r_grid = np.asarray(diagnostics["r_grid"], dtype=np.float64)
    fixed_radius = float(diagnostics["fixed_radius"])
    margin_results = diagnostics["margin_results"]

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), sharex="col", sharey="row")
    for margin, res in enumerate(margin_results):
        local_ax = axes[0, margin]
        nearest_ax = axes[1, margin]

        local_ax.plot(r_grid, res["prior_local_mass"], color="#1d3557", linestyle="--", linewidth=2.4, label="Prior predictive")
        local_ax.plot(r_grid, res["posterior_local_mass"], color="#d97706", linewidth=2.5, label="Posterior predictive")
        local_ax.set_xlim(0.0, 1.0)
        local_ax.set_ylim(-0.02, 1.02)
        _format_axis(local_ax, xlabel="", ylabel="Average local predictive mass" if margin == 0 else "", title=f"Group {margin + 1}")

        nearest_ax.step(r_grid, res["prior_nearest_cdf"], where="post", color="#1d3557", linestyle="--", linewidth=2.4, label="Prior predictive")
        nearest_ax.step(r_grid, res["posterior_nearest_cdf"], where="post", color="#d97706", linewidth=2.5, label="Posterior predictive")
        nearest_ax.set_xlim(0.0, 1.0)
        nearest_ax.set_ylim(-0.02, 1.02)
        _format_axis(nearest_ax, xlabel=r"Radius $r$", ylabel=r"$M_i(r)$" if margin == 0 else "")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Posterior Predictive Neighborhood Diagnostics", fontsize=22, y=1.02)
    fig.text(0.5, 0.02, "Top row: average posterior predictive mass in a radius-$r$ neighborhood around the observed event times. Bottom row: posterior predictive probability that a new draw lies within distance $r$ of at least one observed event. Curves above the prior indicate that conditioning on the data moves predictive mass toward the observations.", ha="center", va="bottom", fontsize=10, wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    return _save_figure(fig, save_dir, f"{file_prefix}_appendix_predictive_diagnostics", save_pdf=save_pdf, dpi=dpi)


def _plot_posterior_predictive_scatter(
    posterior_samples,
    observed_data,
    posterior_means,
    save_dir: Path,
    *,
    file_prefix: str,
    save_pdf: bool,
    dpi: int,
    scatter_quantile: float,
):
    posterior_samples = np.asarray(posterior_samples, dtype=np.float64)
    x_values = posterior_samples[:, 0]
    y_values = posterior_samples[:, 1]

    max_lim = max(
        float(np.quantile(x_values, scatter_quantile)),
        float(np.quantile(y_values, scatter_quantile)),
        float(np.max(observed_data[0]) * 1.1),
        float(np.max(observed_data[1]) * 1.1),
    )

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    ax.scatter(
        x_values,
        y_values,
        s=18,
        color="#1d3557",
        alpha=1.0,
        linewidths=0.25,
        edgecolors="white",
        zorder=2,
    )

    for idx, value in enumerate(np.sort(np.asarray(observed_data[0], dtype=np.float64))):
        ax.axvline(value, color="#1d3557", linewidth=1.2, alpha=0.18, zorder=1)
    for idx, value in enumerate(np.sort(np.asarray(observed_data[1], dtype=np.float64))):
        ax.axhline(value, color="#bc4749", linewidth=1.2, alpha=0.18, zorder=1)

    ax.plot([0.0, max_lim], [0.0, max_lim], color="black", linestyle="--", linewidth=1.8, alpha=0.85)
    ax.scatter([posterior_means[0]], [posterior_means[1]], color="black", s=70, marker="x", linewidths=2.0, zorder=4)
    ax.set_xlim(0.0, max_lim)
    ax.set_ylim(0.0, max_lim)
    ax.set_aspect("equal", adjustable="box")
    _format_axis(ax, xlabel="Posterior predictive sample in group 1", ylabel="Posterior predictive sample in group 2")
    fig.tight_layout()
    return _save_figure(fig, save_dir, f"{file_prefix}_posterior_predictive_scatter", save_pdf=save_pdf, dpi=dpi)


def _write_caption_file(save_dir: Path, captions: dict[str, str]):
    lines = []
    for name, caption in captions.items():
        lines.append(name)
        lines.append(caption)
        lines.append("")

    with open(Path(save_dir) / "simulation_study_plot_captions.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).strip() + "\n")


def create_simulation_study_plot_suite(
    results_dir,
    *,
    save_dir=None,
    file_prefix="toy_simulation",
    confidence_level=0.90,
    fixed_radius=0.5,
    show_dkw_bands=True,
    save_pdf=True,
    dpi=300,
    main_time_quantile=0.995,
    scatter_quantile=0.995,
    derivative_delta_values=(1e-1, 5e-1, 1.0),
    derivative_r_grid=None,
    fixed_anchor_time=1e-1,
    shifted_anchor_fixed_delta=5e-1,
    positive_anchor_shifts=(1e-2, 5e-2, 1e-1),
    negative_anchor_shifts=(-1e-2, -5e-2, -1e-1, -5e-1),
    hazard_delta_values=(1e-1, 2e-1, 5e-1),
    hazard_r_grid=None,
):
    _setup_matplotlib()

    results_dir = Path(results_dir)
    save_dir = Path(results_dir if save_dir is None else save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    arrays_path, observed_data, posterior_samples, prior_survival_curves = _load_simulation_study_inputs(results_dir)
    diagnostics = create_prior_posterior_separate_diagnostics(
        prior_predictive_samples=prior_survival_curves,
        posterior_predictive_samples=posterior_samples,
        observed_event_times_by_group=observed_data,
        save_dir=save_dir,
        file_prefix=file_prefix,
        confidence_level=confidence_level,
        show_dkw_bands=show_dkw_bands,
        save_pdf=save_pdf,
        fixed_radius=fixed_radius,
        dpi=dpi,
    )
    diagnostics["posterior_samples"] = posterior_samples
    if derivative_r_grid is None:
        derivative_r_grid = np.linspace(-1.0, 1.0, 401)
    if hazard_r_grid is None:
        hazard_r_grid = np.asarray(diagnostics["t_grid"], dtype=np.float64)

    saved_paths = dict(diagnostics["saved_paths"])
    saved_paths["group_overlay_survival_cumhaz"] = _plot_group_overlay_survival_and_cumhaz(diagnostics, save_dir, file_prefix=file_prefix, observed_data=observed_data, save_pdf=save_pdf, dpi=dpi)
    saved_paths["posterior_predictive_scatter"] = _plot_posterior_predictive_scatter(posterior_samples, observed_data, diagnostics["posterior_means"], save_dir, file_prefix=file_prefix, save_pdf=save_pdf, dpi=dpi, scatter_quantile=scatter_quantile)
    saved_paths["hazard_rate_estimate"] = _plot_hazard_rate_estimate_from_cumhaz(
        diagnostics,
        save_dir,
        file_prefix=file_prefix,
        hazard_delta_values=hazard_delta_values,
        hazard_r_grid=hazard_r_grid,
        save_pdf=save_pdf,
        dpi=dpi,
    )
    saved_paths["posterior_cumhaz_derivative_approx"] = _plot_posterior_cumhaz_derivative_approximation(
        diagnostics,
        save_dir,
        file_prefix=file_prefix,
        delta_values=derivative_delta_values,
        derivative_r_grid=derivative_r_grid,
        save_pdf=save_pdf,
        dpi=dpi,
    )
    saved_paths["posterior_cumhaz_derivative_fixed_anchor"] = _plot_fixed_anchor_cumhaz_derivative_approximation(
        diagnostics,
        save_dir,
        file_prefix=file_prefix,
        fixed_anchor=float(fixed_anchor_time),
        delta_values=derivative_delta_values,
        derivative_r_grid=derivative_r_grid,
        save_pdf=save_pdf,
        dpi=dpi,
    )
    saved_paths["posterior_cumhaz_derivative_shifted_anchor_positive"] = _plot_shifted_anchor_cumhaz_derivative_collection(
        diagnostics,
        save_dir,
        file_prefix=file_prefix,
        derivative_r_grid=derivative_r_grid,
        anchor_shifts=positive_anchor_shifts,
        fixed_delta=float(shifted_anchor_fixed_delta),
        suffix="posterior_cumhaz_derivative_shifted_anchor_positive",
        xlabel=fr"Scaled offset $r$ with fixed $D={float(shifted_anchor_fixed_delta):g}$",
        save_pdf=save_pdf,
        dpi=dpi,
    )
    saved_paths["posterior_cumhaz_derivative_shifted_anchor_negative"] = _plot_shifted_anchor_cumhaz_derivative_collection(
        diagnostics,
        save_dir,
        file_prefix=file_prefix,
        derivative_r_grid=derivative_r_grid,
        anchor_shifts=negative_anchor_shifts,
        fixed_delta=float(shifted_anchor_fixed_delta),
        suffix="posterior_cumhaz_derivative_shifted_anchor_negative",
        xlabel=fr"Scaled offset $r$ with fixed $D={float(shifted_anchor_fixed_delta):g}$",
        save_pdf=save_pdf,
        dpi=dpi,
    )

    captions = {
        "Survival and cumulative hazard panels": "Prior and posterior predictive survival and cumulative hazard summaries for each group. Colors distinguish the two groups and the vertical lines mark the observed event times. Dotted and dash-dotted vertical lines indicate the prior and posterior predictive means.",
        "Group overlay": "Combined overlay of the group-specific prior and posterior predictive survival and cumulative hazard summaries. The overlay highlights how the two groups deviate from the shared prior baseline after conditioning on the data.",
        "Local diagnostics": "The local predictive mass and nearest-observation probability diagnostics compare prior and posterior predictive concentration near the observed event times in each group.",
        "Directional local diagnostics": "Positive-direction plots restrict attention to predictive draws that fall above observed event times by at most r, while negative-direction plots restrict attention to draws that fall below observed event times by at most r.",
        "Normalized directional local diagnostics": "The normalized one-sided local-mass plots divide the positive-direction summands by P(X_i > t_{i,j}^{obs}) and the negative-direction summands by P(X_i \\le t_{i,j}^{obs}), so each curve is scaled by the corresponding upper- or lower-tail probability at that observed time.",
        "Hazard-scale normalized directional local diagnostics": "These plots show average cumulative-hazard ratios. The positive-direction curve averages H_i(t_{i,j}^{obs}+r)/H_i(t_{i,j}^{obs}) across observed times, and the negative-direction curve averages H_i(max(t_{i,j}^{obs}-r,0))/H_i(t_{i,j}^{obs}).",
        "Hazard-rate estimate from cumulative hazard": "These plots estimate the hazard rate by the centered finite-difference quotient (H_i(r+D)-H_i(max(r-D,0)))/(2D) as r varies on the x-axis, with fixed bandwidths D in {0.1, 0.2, 0.5}. Posterior curves are solid and prior curves are dotted.",
        "Posterior cumulative-hazard derivative approximation": "These plots show the prior and posterior averages of the finite-difference quotient (H_i(x+rD)-H_i(x))/(Dr) over the observed event times x in each group, with three step sizes D and scaled offset r in [-1,1]. For this panel only, the prior curves are computed from 1000 inverse-transform samples from the prior and drawn with dotted lines.",
        "Fixed-anchor cumulative-hazard derivative approximation": "These plots replace the average over observed anchors by the single fixed anchor x=0.1 and show the prior and posterior finite-difference quotient (H_i(x+rD)-H_i(x))/(Dr) for three step sizes D and scaled offset r in [-1,1]. For this panel only, the prior curves are computed from 1000 inverse-transform samples from the prior and drawn with dotted lines.",
        "Shifted-anchor cumulative-hazard derivative approximation (positive U)": "These plots fix D=0.5 and replace each observed anchor x by x+U before evaluating the finite-difference quotient (H_i(x+U+rD)-H_i(x+U))/(Dr). The positive-shift panel uses U in {0.01, 0.05, 0.1}. For this panel only, the prior curves are computed from 1000 inverse-transform samples from the prior and drawn with dotted lines.",
        "Shifted-anchor cumulative-hazard derivative approximation (negative U)": "These plots fix D=0.5 and replace each observed anchor x by max(0, x+U) before evaluating the finite-difference quotient (H_i(x+U+rD)-H_i(x+U))/(Dr). The negative-shift panel uses U in {-0.01, -0.05, -0.1, -0.5}. For this panel only, the prior curves are computed from 1000 inverse-transform samples from the prior and drawn with dotted lines.",
        "Directional nearest-event diagnostics": "The one-sided nearest-event diagnostics report the posterior predictive probability of landing within distance r of at least one observed event time when only upward or only downward deviations are counted.",
        "Posterior scatter": "Joint posterior predictive draws for $(T_1, T_2)$ together with the 45-degree reference line. Stronger concentration around the diagonal indicates stronger positive posterior dependence between the two groups.",
    }
    _write_caption_file(save_dir, captions)

    summary = {
        "arrays_path": arrays_path,
        "save_dir": save_dir,
        "fixed_radius": float(fixed_radius),
        "confidence_level": float(confidence_level),
        "derivative_delta_values": [float(delta) for delta in derivative_delta_values],
        "fixed_anchor_time": float(fixed_anchor_time),
        "shifted_anchor_fixed_delta": float(shifted_anchor_fixed_delta),
        "hazard_delta_values": [float(delta) for delta in hazard_delta_values],
        "positive_anchor_shifts": [float(shift) for shift in positive_anchor_shifts],
        "negative_anchor_shifts": [float(shift) for shift in negative_anchor_shifts],
        "derivative_anchor_points": {
            "group_1": observed_data[0].tolist(),
            "group_2": observed_data[1].tolist(),
        },
        "prior_predictive_means": diagnostics["prior_means"].tolist(),
        "posterior_predictive_means": diagnostics["posterior_means"].tolist(),
        "empirical_group_means": diagnostics["empirical_means"].tolist(),
        "saved_paths": saved_paths,
        "summaries": diagnostics["summaries"],
    }

    with open(save_dir / f"{file_prefix}_plot_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    return summary