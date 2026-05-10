#Set this BEFORE importing jax in your entry script:
import os
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import final_min_id_sampler as min_id_sampler
import final_MCMC_hitting_scenario as MCMC_hitting_scenario
import final_MCMC_extr_seq as MCMC_extr_seq
import jax
import jax.numpy as jnp
import final_partition_class as partition_class
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime
from pathlib import Path
from final_simulation_study_plots import create_simulation_study_plot_suite

def _atomic_write_json(path: Path, obj) -> None:
    """Write JSON atomically (best-effort) to avoid corrupt partial files."""
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp_path, path)


def _load_json_list(path: Path):
    """Load a JSON file that stores a list; return [] if missing."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r") as f:
        data = json.load(f)
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data)}")
    return data


def ensure_hitting_scenario_samples(
    *,
    data,
    n_samples: int,
    tau0=1.0,
    tau1=2 / 3,
    sigma=0.5,
    precision_a=24,
    precision_b=24,
    MCMC_steps_hit_scen=5000,
    hit_verbose=50,
    hit_offset=10,
    seed_jax=42,
    out_dir: Path,
):
    """Ensure exactly n_samples hitting-scenario samples exist on disk.

    If the output JSON already contains some samples, only the missing number
    is simulated and appended. This makes runs resumable and easy to split
    across server allocations.
    """
    if out_dir is None:
        raise ValueError("out_dir must be provided")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    #suffix = f"_steps{MCMC_steps_hit_scen}"
    samples_path = out_dir / f"hitting_scenario_samples.json"

    existing_raw = _load_json_list(samples_path)
    existing_count = len(existing_raw)

    if existing_count >= n_samples:
        raw_needed = existing_raw[:n_samples]
        partitions = []
        for part in raw_needed:
            cleaned = [[tuple(pair) for pair in subset] for subset in part]
            partitions.append(partition_class.partition(cleaned))
        return partitions

    missing = n_samples - existing_count
    print(f"Found {existing_count} existing hitting-scenario samples; simulating {missing} more...")

    # Important: change the seed based on existing_count so that
    # (a) the new chunk differs from previous chunks and
    # (b) repeated resume runs are deterministic.
    new_partitions = simulate_hitting_scenarios(
        data,
        tau0=tau0,
        tau1=tau1,
        sigma=sigma,
        precision_a=precision_a,
        precision_b=precision_b,
        MCMC_steps_hit_scen=MCMC_steps_hit_scen,
        hit_verbose=hit_verbose,
        hit_offset=hit_offset,
        n_samples=missing,
        seed_jax=seed_jax + existing_count,
        out_dir=None,  # avoid overwriting; we handle append
    )

    new_raw = [p.get_partition() for p in new_partitions]
    combined = existing_raw + new_raw
    _atomic_write_json(samples_path, combined)

    return load_partitions_json(samples_path)


def ensure_extremal_samples(
    *,
    data,
    partitions,
    n_samples: int,
    MCMC_steps_extr_seqs=4000,
    tau0=1.0,
    tau1=2 / 3,
    sigma=0.5,
    precision_a=24,
    precision_b=24,
    k_ext=1,
    seed_jax=42,
    out_dir: Path,
):
    """Ensure exactly n_samples extremal samples exist on disk.

    Appends missing samples to `extr_seq_samples_steps{MCMC_steps_extr_seqs}.json`.
    Deterministic chunking: sample i always uses seed `seed_jax + i`.
    """
    if out_dir is None:
        raise ValueError("out_dir must be provided")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(partitions) < n_samples:
        raise ValueError(f"Need at least {n_samples} partitions, got {len(partitions)}")

    #suffix = f"_steps{MCMC_steps_extr_seqs}"
    samples_path = out_dir / f"extr_seq_samples.json"

    existing_raw = _load_json_list(samples_path)
    existing_count = len(existing_raw)

    if existing_count >= n_samples:
        raw_needed = existing_raw[:n_samples]
        return [jnp.array(item) for item in raw_needed]

    missing = n_samples - existing_count
    print(f"Found {existing_count} existing extremal samples; simulating {missing} more...")

    new_partitions = partitions[existing_count:n_samples]
    # seed_jax offset ensures sample i uses PRNGKey(seed_jax + i)
    new_Z_n_list = simulate_extremal_functions_for_partitions(
        data,
        new_partitions,
        MCMC_steps_extr_seqs=MCMC_steps_extr_seqs,
        tau0=tau0,
        tau1=tau1,
        sigma=sigma,
        precision_a=precision_a,
        precision_b=precision_b,
        k_ext=k_ext,
        seed_jax=seed_jax,
        start_index=existing_count,
        out_dir=None,  # avoid overwrite; we handle append
    )

    new_raw_any = _to_jsonable(new_Z_n_list)
    if not isinstance(new_raw_any, list):
        raise TypeError(f"Expected list from _to_jsonable, got {type(new_raw_any)}")
    new_raw = new_raw_any
    combined = existing_raw + new_raw
    _atomic_write_json(samples_path, combined)

    # Return exactly n_samples
    return [jnp.array(item) for item in combined[:n_samples]]


def ensure_min_id_samples(
    *,
    data,
    grid_list,
    n_samples: int,
    tau0=1.0,
    tau1=2 / 3,
    sigma=0.5,
    precision_a=24,
    precision_b=24,
    min_id_steps=4000,
    seed_jax=42,
    out_dir: Path,
):
    """Ensure exactly n_samples min-id samples exist on disk.

    Appends missing samples to `min_id_samples_steps{min_id_steps}.json`.
    Deterministic chunking: sample i always uses seed `seed_jax + i`.
    """
    if out_dir is None:
        raise ValueError("out_dir must be provided")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    #suffix = f"_steps{min_id_steps}"
    samples_path = out_dir / f"min_id_samples.json"

    existing_raw = _load_json_list(samples_path)
    existing_count = len(existing_raw)

    if existing_count >= n_samples:
        raw_needed = existing_raw[:n_samples]
        return [jnp.array(item) for item in raw_needed]

    missing = n_samples - existing_count
    print(f"Found {existing_count} existing min-id samples; simulating {missing} more...")

    new_Y_min_list = simulate_min_id_vectors(
        data,
        grid_list,
        missing,
        tau0=tau0,
        tau1=tau1,
        sigma=sigma,
        precision_a=precision_a,
        precision_b=precision_b,
        min_id_steps=min_id_steps,
        seed_jax=seed_jax,
        start_index=existing_count,
        out_dir=None,  # avoid overwrite; we handle append
    )

    new_raw_any = _to_jsonable(new_Y_min_list)
    if not isinstance(new_raw_any, list):
        raise TypeError(f"Expected list from _to_jsonable, got {type(new_raw_any)}")
    new_raw = new_raw_any
    combined = existing_raw + new_raw
    _atomic_write_json(samples_path, combined)

    return [jnp.array(item) for item in combined[:n_samples]]


def load_partitions_json(path):
    """Load partitions from JSON and return list of partition objects."""
    with open(path, "r") as f:
        data = json.load(f)
    partitions = []
    for part in data:
        cleaned = [[tuple(pair) for pair in subset] for subset in part]
        partitions.append(partition_class.partition(cleaned))
    return partitions


def load_array_list_json(path):
    """Load list-of-arrays stored as JSON and convert to jnp arrays."""
    with open(path, "r") as f:
        data = json.load(f)
    return [jnp.array(item) for item in data]


def load_saved_grids(
    arrays_path: Path,
    *,
    x_min,
    x_max,
    n_points_grid,
    prior_n_points_grid,
    tau0,
    tau1,
    sigma,
    precision_a,
    precision_b,
):
    """Load cached data/prior grids when they match the current configuration."""
    arrays_path = Path(arrays_path)
    if not arrays_path.exists():
        return None, None

    with np.load(arrays_path) as saved:
        config_matches = (
            int(saved.get("n_points_grid", -1)) == int(n_points_grid)
            and np.isclose(float(saved.get("x_min", np.nan)), float(x_min))
            and np.isclose(float(saved.get("x_max", np.nan)), float(x_max))
            and np.isclose(float(saved.get("tau0", np.nan)), float(tau0))
            and np.isclose(float(saved.get("tau1", np.nan)), float(tau1))
            and np.isclose(float(saved.get("sigma", np.nan)), float(sigma))
            and int(saved.get("precision_a", -1)) == int(precision_a)
            and int(saved.get("precision_b", -1)) == int(precision_b)
        )
        if not config_matches:
            return None, None

        grid_keys = {"x_grid_0", "exp_measure_0", "x_grid_1", "exp_measure_1"}
        prior_keys = {
            "prior_x_grid_0",
            "prior_exp_measure_0",
            "prior_x_grid_1",
            "prior_exp_measure_1",
            "prior_n_points_grid",
        }

        grid_list = None
        prior_grid_list = None

        if grid_keys.issubset(saved.files):
            grid_list = [
                (saved["x_grid_0"], saved["exp_measure_0"]),
                (saved["x_grid_1"], saved["exp_measure_1"]),
            ]

        if prior_keys.issubset(saved.files) and int(saved["prior_n_points_grid"]) == int(prior_n_points_grid):
            prior_grid_list = [
                (saved["prior_x_grid_0"], saved["prior_exp_measure_0"]),
                (saved["prior_x_grid_1"], saved["prior_exp_measure_1"]),
            ]

    return grid_list, prior_grid_list

def generate_data(n_obs_per_margin, seed_np=42):
    """Generate artificial 2D data with n observations per margin."""
    np.random.seed(seed_np)
    row_0 = np.random.weibull(a=1.0, size=n_obs_per_margin).tolist()
    row_1 = np.random.weibull(a=2.0, size=n_obs_per_margin).tolist()
    return [row_0, row_1]


def simulate_hitting_scenarios(
    data,
    tau0=1.0,
    tau1=2/3,
    sigma=0.5,
    precision_a=24,
    precision_b=24,
    MCMC_steps_hit_scen=5000,
    hit_verbose=50,
    hit_offset=10,
    n_samples=5,
    seed_jax=42,
    out_dir=None,
):
    """Simulate conditional hitting scenarios and return n_samples partitions."""
    key = jax.random.PRNGKey(seed_jax)
    if MCMC_steps_hit_scen /(n_samples * hit_offset) <= 1:
        raise ValueError("Not enough MCMC steps for sampling partitions.")
    MCMC_partitions, key = MCMC_hitting_scenario.Gibbs_sampler_cond_hit_scen(
        MCMC_steps_hit_scen, data, tau0, tau1, precision_a, precision_b, sigma, verbose=hit_verbose, key=key
    )
    # Partitions are appended once per full sweep, plus the initialization.
    # Use the actual list length to avoid off-by-one issues.
    last_idx = len(MCMC_partitions) - 1
    start = last_idx
    stop = last_idx - hit_offset * n_samples
    indices = range(start, stop, -hit_offset)
    sampled_partitions = [MCMC_partitions[i] for i in indices]
    if out_dir is not None:
        #suffix = f"_steps{MCMC_steps_hit_scen}"
        partitions_list = [MCMC_partitions[i].get_partition() for i in indices]
        with open(out_dir / f"hitting_scenario_samples.json", "w") as f:
            json.dump(partitions_list, f, indent=2)
    return sampled_partitions


def _to_jsonable(obj):
    """Recursively convert JAX/NumPy arrays to Python lists for JSON serialization."""
    # JAX/NumPy arrays
    if hasattr(obj, "tolist"):
        try:
            return np.array(obj).tolist()
        except Exception:
            pass

    # dict / list / tuple
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]

    # scalars (numpy scalars etc.)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()

    return obj


def simulate_extremal_functions_for_partitions(
    data,
    partitions,
    MCMC_steps_extr_seqs=4000,
    tau0=1.0,
    tau1=2/3,
    sigma=0.5,
    precision_a=24,
    precision_b=24,
    k_ext=1,
    seed_jax=42,
    start_index=0,
    out_dir=None,
):
    """Simulate extremal functions for a list of partitions."""
    #extr_sequences_MCMC = []
    Z_n_list = []
    for idx, part in enumerate(partitions):
        global_idx = start_index + idx
        if (idx + 1) % 50 == 0:
            print(f"Simulating extremal sequence {global_idx + 1}/{start_index + len(partitions)}")
        key = jax.random.PRNGKey(seed_jax + global_idx)
        extr_seq = MCMC_extr_seq.compute_parallel_chains(
            MCMC_steps_extr_seqs, data, part, tau0, tau1, sigma,
            precision_a, precision_b, k_ext, key=key
        )
        #extr_sequences_MCMC.append(extr_seq)
        Z_n = MCMC_extr_seq.create_sample_from_extr_seq(extr_seq, MCMC_steps_extr_seqs - 1)
        Z_n_list.append(Z_n)


    # Z_n_list = [
    #     MCMC_extr_seq.create_sample_from_extr_seq(seq, MCMC_steps_extr_seqs - 1)
    #     for seq in extr_sequences_MCMC
    # ]

    if out_dir is not None:
        #suffix = f"_steps{MCMC_steps_extr_seqs}"

        # # Pickle the full sequences (lossless, avoids JSON issues / size blowups)
        # with open(out_dir / f"extr_seq_MCMC.pkl", "wb") as f:
        #     pickle.dump(extr_sequences_MCMC, f)

        # JSON only for the final samples (small)
        Z_n_json = _to_jsonable(Z_n_list)
        with open(out_dir / f"extr_seq_samples.json", "w") as f:
            json.dump(Z_n_json, f, indent=2)

    return Z_n_list


def simulate_min_id_vectors(
    data,
    grid_list,
    n_samples,
    tau0=1.0,
    tau1=2/3,
    sigma=0.5,
    precision_a=24,
    precision_b=24,
    min_id_steps=4000,
    seed_jax=42,
    start_index=0,
    out_dir=None,
):
    """Simulate n_samples min-id random vectors via simulate_exact_min_id."""
    Y_min_list = []
    for idx in range(n_samples):
        global_idx = start_index + idx
        if (idx + 1) % 50 == 0:
            print(f"Simulating min-id vector {global_idx + 1}/{start_index + n_samples}")
        key_m = jax.random.PRNGKey(seed_jax + global_idx)
        Y_min = min_id_sampler.simulate_exact_min_id(
            key_m,
            data,
            grid_list,
            tau0,
            tau1,
            sigma,
            k=1,
            precision_a=precision_a,
            precision_b=precision_b,
            steps=min_id_steps,
        )
        Y_min_list.append(Y_min)

    if out_dir is not None:
        #suffix = f"_steps{min_id_steps}"
        Y_min_json = _to_jsonable(Y_min_list)
        with open(out_dir / f"min_id_samples.json", "w") as f:
            json.dump(Y_min_json, f, indent=2)

    return Y_min_list


def patch_final_samples(Z_n_list, Y_min_list):
    """Patch extremal and min-id samples via componentwise minimum."""
    final_samples = []
    for Z_n, Y_min in zip(Z_n_list, Y_min_list):
        z0 = float(Z_n[0,0])
        z1 = float(Z_n[1,0])
        m0 = float(Y_min[0, 0])
        m1 = float(Y_min[1, 0])
        final_samples.append([float(jnp.minimum(z0, m0)), float(jnp.minimum(z1, m1))])
    return  final_samples



if __name__ == "__main__":
    mp.freeze_support()

    print(datetime.now())

    export_dir = Path(__file__).resolve().parent
    run_name = "final_sim_study"
    out_dir = export_dir / "Results" / run_name
    plots_dir = export_dir / "Plots" / run_name
    hitting_samples_path = out_dir / "hitting_scenario_samples.json"
    extremal_samples_path = out_dir / "extr_seq_samples.json"
    min_id_samples_path = out_dir / "min_id_samples.json"

    # Auto-resume: simulate any missing intermediate outputs, otherwise reuse them.
    RUN_HIT_SCEN = not hitting_samples_path.exists()
    RUN_EXTREMAL = not extremal_samples_path.exists()
    RUN_MIN_ID = not min_id_samples_path.exists()
    VISUALIZE = True

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results will be saved to: {out_dir}")
    print(f"Plots will be saved to: {plots_dir}")

    print("Running simulation study...")
    n_obs_per_margin = 6
    tau0 = 1.0
    tau1 = 2/3
    sigma = 0.5
    precision_a = 24
    precision_b = 24
    MCMC_steps_hit_scen = 12000
    hit_verbose = 50
    hit_offset = 10
    n_samples = 1000
    MCMC_steps_extr_seqs = 5000
    x_min = 0.001
    x_max = 10.0
    n_points_grid = 1000
    prior_n_points_grid = 1000
    min_id_steps = 5000
    seed_np = 42
    seed_jax = 42


    data = generate_data(n_obs_per_margin=n_obs_per_margin, seed_np=seed_np)
    print(data)
    # suffix_hit = f"_steps{MCMC_steps_hit_scen}"
    # suffix_ext = f"_steps{MCMC_steps_extr_seqs}"
    # suffix_min = f"_steps{min_id_steps}"

    if RUN_HIT_SCEN:
        print("Ensuring (append/resume) hitting-scenario samples...")
        conditional_hitting_scenarios = ensure_hitting_scenario_samples(
            data=data,
            n_samples=n_samples,
            tau0=tau0,
            tau1=tau1,
            sigma=sigma,
            precision_a=precision_a,
            precision_b=precision_b,
            MCMC_steps_hit_scen=MCMC_steps_hit_scen,
            hit_verbose=hit_verbose,
            hit_offset=hit_offset,
            seed_jax=seed_jax,
            out_dir=out_dir,
        )
    else:
        print("Loading hitting scenarios from disk...")
        conditional_hitting_scenarios = load_partitions_json(
            out_dir / f"hitting_scenario_samples.json"
        )

        # If you accidentally request more samples than exist on disk while RUN_HIT_SCEN=False,
        # fail early with a clear error.
        if len(conditional_hitting_scenarios) < n_samples:
            raise ValueError(
                f"Requested n_samples={n_samples} but only found {len(conditional_hitting_scenarios)} "
                f"in {out_dir / f'hitting_scenario_samples.json'}. "
                "Set RUN_HIT_SCEN=True to append missing samples."
            )

        conditional_hitting_scenarios = conditional_hitting_scenarios[:n_samples]

    if RUN_EXTREMAL:
        print("Ensuring (append/resume) extremal samples...")
        Z_n_list = ensure_extremal_samples(
            data=data,
            partitions=conditional_hitting_scenarios,
            n_samples=n_samples,
            tau0=tau0,
            tau1=tau1,
            sigma=sigma,
            precision_a=precision_a,
            precision_b=precision_b,
            MCMC_steps_extr_seqs=MCMC_steps_extr_seqs,
            k_ext=1,
            seed_jax=seed_jax,
            out_dir=out_dir,
        )
    else:
        print("Loading extremal samples from disk...")
        Z_n_list = load_array_list_json(
            out_dir / f"extr_seq_samples.json"
        )

        if len(Z_n_list) < n_samples:
            raise ValueError(
                f"Requested n_samples={n_samples} but only found {len(Z_n_list)} "
                f"in {out_dir / f'extr_seq_samples.json'}. "
                "Set RUN_EXTREMAL=True to append missing samples."
            )

        Z_n_list = Z_n_list[:n_samples]

    arrays_path = out_dir / "data_study_arrays.npz"
    grid_list, prior_grid_list = load_saved_grids(
        arrays_path,
        x_min=x_min,
        x_max=x_max,
        n_points_grid=n_points_grid,
        prior_n_points_grid=prior_n_points_grid,
        tau0=tau0,
        tau1=tau1,
        sigma=sigma,
        precision_a=precision_a,
        precision_b=precision_b,
    )

    if grid_list is not None:
        print("Loaded saved grids")
    else:
        print("No compatible saved grids found. Generating new grids...")
        grid_list = [
            min_id_sampler.compute_exp_measure_margin_grid(
                i, data, tau0, tau1, sigma,
                precision_a, precision_b, x_min, x_max, n_points_grid
            )
            for i in range(len(data))
        ]
        print("Generated new grids")

    if prior_grid_list is not None:
        print("Loaded saved prior grids")
    else:
        print("No compatible saved prior grids found. Generating new prior grids...")
        empty_data = [[] for _ in range(len(data))]
        prior_grid_list = [
            min_id_sampler.compute_exp_measure_margin_grid(
                i,
                empty_data,
                tau0,
                tau1,
                sigma,
                precision_a,
                precision_b,
                x_min,
                x_max,
                prior_n_points_grid,
            )
            for i in range(len(data))
        ]
        print("Generated new prior grids")

    if RUN_MIN_ID:
        print("Ensuring (append/resume) min-id samples...")
        Y_min_list = ensure_min_id_samples(
            data=data,
            grid_list=grid_list,
            n_samples=n_samples,
            tau0=tau0,
            tau1=tau1,
            sigma=sigma,
            precision_a=precision_a,
            precision_b=precision_b,
            min_id_steps=min_id_steps,
            seed_jax=seed_jax,
            out_dir=out_dir,
        )
    else:
        print("Loading min-id samples from disk...")
        Y_min_list = load_array_list_json(
            out_dir / f"min_id_samples.json"
        )

        if len(Y_min_list) < n_samples:
            raise ValueError(
                f"Requested n_samples={n_samples} but only found {len(Y_min_list)} "
                f"in {out_dir / f'min_id_samples.json'}. "
                "Set RUN_MIN_ID=True to append missing samples."
            )

        Y_min_list = Y_min_list[:n_samples]

    final_samples = patch_final_samples(Z_n_list, Y_min_list)
    prior_survival_curves = [
        (
            np.asarray(prior_grid_list[margin][0], dtype=np.float64),
            np.exp(-np.asarray(prior_grid_list[margin][1], dtype=np.float64)),
        )
        for margin in range(len(prior_grid_list))
    ]

    np.savez(
        arrays_path,
        data_row0=np.array(data[0], dtype=np.float64),
        data_row1=np.array(data[1], dtype=np.float64),
        n_points_grid=n_points_grid,
        x_min=x_min,
        x_max=x_max,
        x_grid_0=np.array(grid_list[0][0]),
        exp_measure_0=np.array(grid_list[0][1]),
        x_grid_1=np.array(grid_list[1][0]),
        exp_measure_1=np.array(grid_list[1][1]),
        prior_n_points_grid=prior_n_points_grid,
        prior_x_grid_0=np.array(prior_grid_list[0][0]),
        prior_exp_measure_0=np.array(prior_grid_list[0][1]),
        prior_x_grid_1=np.array(prior_grid_list[1][0]),
        prior_exp_measure_1=np.array(prior_grid_list[1][1]),
        final_min_id_samples=np.array(final_samples, dtype=np.float64),
        tau0=tau0,
        tau1=tau1,
        sigma=sigma,
        precision_a=precision_a,
        precision_b=precision_b,
        min_id_steps=min_id_steps,
        MCMC_steps_hit_scen=MCMC_steps_hit_scen,
        offset_hit_scen=hit_offset,
        MCMC_steps_extr_seqs=MCMC_steps_extr_seqs,
        n_samples=n_samples,
    )

    meta = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "tau0": tau0,
        "tau1": tau1,
        "sigma": sigma,
        "precision_a": precision_a,
        "precision_b": precision_b,
        "n_obs_per_margin": n_obs_per_margin,
        "prior_n_points_grid": prior_n_points_grid,
        "MCMC_steps_hit_scen": MCMC_steps_hit_scen,
        "hit_offset": hit_offset,
        "n_samples": n_samples,
        "MCMC_steps_extr_seqs": MCMC_steps_extr_seqs,
        "final_min_id_samples": final_samples,
        "data": {
            "row0": list(map(float, data[0])),
            "row1": list(map(float, data[1])),
        },
    }

    # Write as JSON (keeps nested lists intact)
    with open(out_dir / f"data_study_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print ("Simulation done. Starting visualization")

    if VISUALIZE:


        out = create_simulation_study_plot_suite(
            out_dir,
            save_dir=plots_dir,
            file_prefix="toy_simulation",
            confidence_level=0.90,
            fixed_radius=0.5,
            show_dkw_bands=True,
            save_pdf=True,
        )
   

        print(out["saved_paths"])
        print(out["summaries"])

    print("All done:", datetime.now())



