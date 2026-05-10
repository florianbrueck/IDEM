# Final Simulation Study Export

This folder contains the standalone code and bundled result data needed to reproduce the simulation study from the paper *Infinitely divisible priors for multivariate survival
functions* which can be found [here](https://arxiv.org/abs/2502.09162).

## Contents

- `environment.yml`: Conda environment `IDEMpriors` which contains all necessary packages.
- `Results/final_sim_study`: Bundled simulation-study outputs used to reproduce the exported plots.
- `final_Final_simulation.py`: Main entrypoint for the full exported simulation study.
- Remaining `final_*.py` files: helper files to reproduce the simulation and the plots.


## Default study settings

The simulation study runs with these defaults:

- `n_obs_per_margin = 6`
- `tau0 = 1.0`
- `tau1 = 2/3`
- `sigma = 0.5`
- `precision_a = 24`
- `precision_b = 24`
- `n_samples = 1000`
- `MCMC_steps_hit_scen = 12000`
- `MCMC_steps_extr_seqs = 5000`
- `min_id_steps = 5000`

If you want different settings, edit `final_Final_simulation.py` directly.
