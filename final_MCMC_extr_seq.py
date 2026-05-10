#Set this BEFORE importing jax in your entry script:
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=true") # --xla_cpu_multi_thread_eigen_threads=1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import final_densities as densities
import numpy as np

import jax
import jax.numpy as jnp
from final_derivatives_laplace_trafo import (gauss_laguerre_gamma_crm_jax, gauss_legendre_gamma_crm_jax,make_F_grid_b0_dep)




def create_sample_from_extr_seq(extr_seq,index):
    ''' This function creates a sample from the extremal sequences. 
    Arguments:  
    extr_seq: list of lists, each list corresponds to a list of samples from Z^{(n,l)}_k
    index: int, the index of the entry of the list of samples to be used for the sample creation
    '''
    sample = jnp.full(extr_seq[0][0].shape, jnp.inf)
    for l in range(len(extr_seq)):
        sample = jnp.minimum(sample, extr_seq[l][index])
    return sample


####################################################################################

def MH_step_Z_l_cached(l,data,partition,locations_current,tau_0,tau_1,sigma,a0,wa,b0_base,wb_base,data_max,precision_b=12,key=None,make_deriv_matrix=None,sd_norm=0.1):
    ''' This function conduct one step in the Metropolis-Hastings MCMC algorithm for the simulation of Z^{(n,l)}_k
    The arguments are indentical to the arguments of the function MH_sampler_Z_l.
    '''
    if key is None:
        key = jax.random.PRNGKey(0)
    
    key1, key2, key3, key4 = jax.random.split(key, 4)
    
    locations_current = jnp.asarray(locations_current)
    locations_proposal = jnp.full(locations_current.shape, jnp.inf)
    N = locations_current.shape[0] * locations_current.shape[1]
    part = partition.get_element(l)
    row_indices = jnp.array([i[0] for i in part])
    
    if jnp.all(row_indices == row_indices[0]):#if Theta_l contains only one row index i, then all other rows of location might be infinite with positive probability.
        B = jax.random.bernoulli(key1, 0.5)
        if B == 1:#all proposed locations finite
            if jnp.all(jnp.isfinite(locations_current)): #all current locations are finite
                eps = jax.random.normal(key2, shape=(N,)).reshape(locations_current.shape) * sd_norm
                log_locations_current = jnp.log(locations_current)
                log_locations_prop = log_locations_current + eps
                locations_proposal = jnp.exp(log_locations_prop)
            else:#only current locations row row_indices[0] is finite
                eps = jax.random.normal(key2, shape=(N,)).reshape(locations_current.shape) * sd_norm
                log_locations_current = jnp.zeros(locations_current.shape)
                log_locations_current = log_locations_current.at[row_indices[0],:].set(jnp.log(locations_current[row_indices[0],:]))
                log_locations_prop = log_locations_current + eps
                locations_proposal = jnp.exp(log_locations_prop) 

        else:#only the row row_indices[0] is finite
            eps = jax.random.normal(key2, shape=(locations_current.shape[1],)) * sd_norm
            log_locations_current = jnp.log(locations_current[row_indices[0],:])
            log_locations_prop = log_locations_current + eps
            locations_proposal = locations_proposal.at[row_indices[0],:].set(jnp.exp(log_locations_prop)) 
        
        log_trans_dens_current_to_prop = densities.log_transition_density_MH_Z_l_jax(locations_proposal, locations_current, l, partition, sd_norm)
        log_trans_dens_prop_to_current = densities.log_transition_density_MH_Z_l_jax(locations_current, locations_proposal, l, partition, sd_norm)
        
        m = jnp.max(jnp.array([data_max, jnp.max(jnp.where(jnp.isfinite(locations_proposal), locations_proposal, -jnp.inf)),
                                jnp.max(jnp.where(jnp.isfinite(locations_current), locations_current, -jnp.inf))]))
        m=jnp.min(jnp.array([jnp.array(10.0),m]))
        b0,wb = (m+tau_0)*b0_base, (m+tau_0)*wb_base #gauss_legendre_gamma_crm_jax(precision_b, 0.0, m + tau_0)
        
        log_p_current = jnp.log(densities.dens_Z_l_jax_cached(l,locations_current,data,partition,tau_0,tau_1,sigma,a0,wa,b0,wb,make_deriv_matrix))
        log_p_prop = jnp.log(densities.dens_Z_l_jax_cached(l,locations_proposal,data,partition,tau_0,tau_1,sigma,a0,wa,b0,wb,make_deriv_matrix))

        log_accept_ratio = log_trans_dens_prop_to_current + log_p_prop - log_p_current - log_trans_dens_current_to_prop

        u = jax.random.uniform(key3)
        if jnp.log(u) < log_accept_ratio:
            locations = locations_proposal
        else:
            locations = locations_current
    
    else: #if Theta_l contains distinct row indices we only need to simulate a real-valued random vector
        eps = jax.random.normal(key2, shape=(N,)).reshape(locations_current.shape) * sd_norm
        log_locations_current = jnp.log(locations_current)
        log_locations_prop = log_locations_current + eps
        locations_proposal = jnp.exp(log_locations_prop)
            
        log_trans_dens_current_to_prop = densities.log_transition_density_MH_Z_l_jax(locations_proposal, locations_current, l, partition, sd_norm)
        log_trans_dens_prop_to_current = densities.log_transition_density_MH_Z_l_jax(locations_current, locations_proposal, l, partition, sd_norm)   
        
        m = jnp.max(jnp.array([data_max, jnp.max(locations_proposal), jnp.max(locations_current)]))
        m=jnp.min(jnp.array([jnp.array(10.0),m]))
        b0,wb = (m+tau_0)*b0_base, (m+tau_0)*wb_base #gauss_legendre_gamma_crm_jax(precision_b, 0.0, m + tau_0)
        
        log_p_current = jnp.log(densities.dens_Z_l_jax_cached(l,locations_current,data,partition,tau_0,tau_1,sigma,a0,wa,b0,wb,make_deriv_matrix))
        log_p_prop = jnp.log(densities.dens_Z_l_jax_cached(l,locations_proposal,data,partition,tau_0,tau_1,sigma,a0,wa,b0,wb,make_deriv_matrix))

        log_accept_ratio = log_trans_dens_prop_to_current + log_p_prop - log_p_current - log_trans_dens_current_to_prop

        u = jax.random.uniform(key3)
        if jnp.log(u) < log_accept_ratio:
            locations = locations_proposal
        else:
            locations = locations_current
            
    return locations, key4       



def MH_sampler_Z_l_cached(l,steps,data,partition,tau_0,tau_1,sigma,precision_a=12,precision_b=12,k=1,key=None):
    ''' This function runs the Markov Chain for the extremal sequences.  
    Arguments:  
    k: int, the number of random vectors we want to sample, corresponding to the 1-to-k-steps posterior predicitive distribution
    steps: int, the number of steps in the Markov Chain
    data: list of lists, the list contains the rows if X_IJ with the i-th row given by the i-th list                       
    tau_0,tau_i: are the parameters of the rectangular and Dykstra-Laudt kernel   
    sigma,theta: float>0, parameters of the Gamma Lévy process
    '''
    if key is None:
        key = jax.random.PRNGKey(42)
    
    key1, key2 = jax.random.split(key, 2)
    
    # Draw the initialization   
    locations_current = jnp.exp(jax.random.normal(key1, shape=(len(data),k)))  # lognormal equivalent
    # Initialize the list to store the samples
    locations = [locations_current]
    a0, wa = gauss_laguerre_gamma_crm_jax(precision_a, -sigma)
    b0_base, wb_base = gauss_legendre_gamma_crm_jax(precision_b, 0.0, 1.0)  # weights for [0,1] which are scaled to data dependent interval later
    data_max = jnp.max(jnp.array([val for row in data for val in row]))
    F_grid = make_F_grid_b0_dep(a0, tau_0, tau_1, sigma)

    current_key = key2
    for step in range(steps):
        locations_current, current_key = MH_step_Z_l_cached(l,data,partition,locations_current,tau_0,tau_1,sigma,a0,wa,b0_base,wb_base,data_max,precision_b,current_key,F_grid)
        locations.append(locations_current)
    return locations



def MCMC_ext_seq_cached(steps,data,partition,tau_0,tau_1,sigma,precision_a=12,precision_b=12,k=1,key=None):
    ''' This function runs the MCMC algorithm for the extremal sequences given sample from the conditional hitting scenario.
    Arguments:  
    k: int,  the number of random vectors we want to sample, corresponding to the 1-to-k-steps posterior predicitive distribution
    steps: int, the number of steps in the Gibbs sampler
    data: list of lists, the list contains the rows if X_IJ with the i-th row given by the i-th list                       
    tau_0,tau_i: are the parameters of the rectangular and Dykstra-Laudt kernel
    sigma: float>0, parameters of the Gamma Lévy process
    '''
    if key is None:
        key = jax.random.PRNGKey(123)
    
    # Draw the initial partition    
    L = partition.get_length()
    extr_seq = []
    
    keys = jax.random.split(key, L)
    
    for l in range(L):
        #current_partition=partition.get_element(l)
        locations_l = MH_sampler_Z_l_cached(l,steps,data,partition,tau_0,tau_1,sigma,precision_a,precision_b,k,keys[l])
        extr_seq.append(locations_l)
    return extr_seq


###################################################################################




def _serialize_key(key):
    """Convert a JAX PRNGKey (uint32[2]) to a picklable Python tuple."""
    arr = np.array(key, dtype=np.uint32)
    return (int(arr[0]), int(arr[1]))

def _deserialize_key(pair):
    """Rebuild a JAX PRNGKey from a (u32,u32) tuple."""
    return jnp.array(pair, dtype=jnp.uint32)

def _run_single_l(args):
    # l, steps, data, partition, tau_0, tau_1, sigma, precision_a, precision_b, k, key_pair, seed
    l, steps, data, partition, tau_0, tau_1, sigma, precision_a, precision_b, k, key_pair, seed = args
    if key_pair is not None:
        key = _deserialize_key(key_pair)
    else:
        key = jax.random.PRNGKey(seed)
    # Use the cached sampler (not the sequential wrapper) to compare speed fairly
    out = MH_sampler_Z_l_cached(l, steps, data, partition, tau_0, tau_1, sigma,
                                precision_a, precision_b, k, key)
    return l, out

def run_all_l_parallel(steps, data, partition, tau_0, tau_1, sigma,
                       precision_a=12, precision_b=12, k=1,
                       max_workers=None, base_seed=12345, key=None):
    """
    Parallelize the loop over l using multiple CPU processes (Windows-safe).
    If 'key' is provided, split it into L subkeys and pass them to workers
    so results match a sequential run with the same key.
    """
    L = partition.get_length()
    if max_workers is None:
        # Respect SLURM allocation when available; fallback to os.cpu_count()
        try:
            max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", ""))
        except ValueError:
            max_workers = None
        if not max_workers:
            max_workers = os.cpu_count() or 1

    if key is not None and isinstance(key, jnp.ndarray):
        subkeys = jax.random.split(key, L)
        key_pairs = [_serialize_key(sk) for sk in subkeys]
        tasks = [(l, steps, data, partition, tau_0, tau_1, sigma,
                  precision_a, precision_b, k, key_pairs[l], None) for l in range(L)]
    else:
        # deterministic fallback by seed
        tasks = [(l, steps, data, partition, tau_0, tau_1, sigma,
                  precision_a, precision_b, k, None, base_seed + l) for l in range(L)]

    results = [None] * L
    ctx = mp.get_context("spawn")
    ex = ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)

    futures = []
    try:
        futures = [ex.submit(_run_single_l, t) for t in tasks]
        for f in as_completed(futures):
            l_idx, out = f.result()
            results[l_idx] = out
        return results
    except KeyboardInterrupt:
        for f in futures:
            f.cancel()
        raise
    finally:
        ex.shutdown(wait=True, cancel_futures=True)

def compute_parallel_chains(steps, data, partition, tau_0, tau_1, sigma,
                            precision_a=12, precision_b=12, k=1,
                            max_workers=None, base_seed=12345, key=None):
    """
    Starts and computes the parallel chains for all l using the cached sampler.
    Returns extr_seq as a list of lists, one per l.
    """
    return run_all_l_parallel(steps, data, partition, tau_0, tau_1, sigma,
                              precision_a, precision_b, k, max_workers, base_seed, key)

# # Usage example that only runs when this script is executed directly, not when it is imported
# if __name__ == "__main__":
#     from time import perf_counter
#     mp.freeze_support()
#     obs = [[2.0,3.0,5.0,1.01], [1.5,4.0]]
#     steps = 200
#     from MCMC_hitting_scenario import draw_initialization

#     par, _ = draw_initialization(obs, key=jax.random.PRNGKey(1))
#     print(par.get_partition())
#     key = jax.random.PRNGKey(0)
#     print(par)
#     # Parallel (cached)
#     start = perf_counter()
#     extr_seq_par = compute_parallel_chains(
#         steps,
#         data=obs,
#         partition=par,
#         tau_0=1.0,
#         tau_1=1.0,
#         sigma=0.5,
#         precision_a=12,
#         precision_b=12,
#         k=1,
#         key=key
#         # max_workers=2
#     )
#     elapsed_par = perf_counter() - start
#     #print(extr_seq_par)
#     print(f"Parallel (cached): {elapsed_par:.4f}s total, {elapsed_par/(steps*par.get_length()):.6f}s per step per chain")

#     # Non-parallel (cached)
#     start = perf_counter()
#     extr_seq_seq_cached = MCMC_ext_seq_cached(
#         steps,
#         data=obs,
#         partition=par,
#         tau_0=1.0,
#         tau_1=1.0,
#         sigma=0.5,
#         precision_a=12,
#         precision_b=12,
#         k=1,
#         key=key
#     )
#     elapsed_seq_cached = perf_counter() - start
#     #print(extr_seq_seq_cached)
#     print(f"Sequential (cached): {elapsed_seq_cached:.4f}s total, {elapsed_seq_cached/(steps*par.get_length()):.6f}s per step per chain")

#     # Non-parallel (non-cached)
#     start = perf_counter()
#     extr_seq_seq = MCMC_ext_seq(
#         steps,
#         data=obs,
#         partition=par,
#         tau_0=1.0,
#         tau_1=1.0,
#         sigma=0.5,
#         precision_a=12,
#         precision_b=12,
#         k=1,
#         key=key
#     )
#     elapsed_seq = perf_counter() - start
#     print(f"Sequential (non-cached): {elapsed_seq:.4f}s total, {elapsed_seq/(steps*par.get_length()):.6f}s per step per chain")

#     for i in range(len(extr_seq_par)):
#         print(jnp.allclose(jnp.asarray(extr_seq_par[i]),jnp.asarray( extr_seq_seq_cached[i])))
#         print(jnp.allclose(jnp.asarray(extr_seq_par[i]),jnp.asarray( extr_seq_seq[i])))
#     print(extr_seq_seq_cached,extr_seq_seq)
#     # Speedups
#     if elapsed_par > 0:
#         print(f"Speedup: parallel(cached) vs sequential(cached): {elapsed_seq_cached/elapsed_par:.2f}x")
#         print(f"Speedup: parallel(cached) vs sequential(non-cached): {elapsed_seq/elapsed_par:.2f}x")

#### cached is much faster than non-cached and parallel seems to be faster starting from >100 steps