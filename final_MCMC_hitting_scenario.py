import final_partition_class as partition_class
import final_densities as densities
import copy
import jax  # add
import jax.numpy as jnp
from final_derivatives_laplace_trafo import (gauss_laguerre_gamma_crm_jax, gauss_legendre_gamma_crm_jax, make_F_grid)





def Gibbs_sampler_cond_hit_scen(steps,data,tau_0,tau_1,precision_a=12,precision_b=12,sigma=0.5,verbose=50, key=None):
    ''' This function runs the Gibbs sampler for the conditional hitting scenario.  
    Returns (partitions, key)
    '''
    if key is None:
        key = jax.random.PRNGKey(0)

    # Draw the initial partition    
    current_partition, key = draw_initialization(data, key)
    # Initialize the list to store the partitions
    partitions = [current_partition]
    index_set=[]
    
    data_max=get_data_max(data)

    # locations and weights for numerical integration only depend on precision and data
    a0,wa = gauss_laguerre_gamma_crm_jax(precision_a, -sigma, dtype=jnp.float64)
    b0,wb = gauss_legendre_gamma_crm_jax(precision_b, 0.0, data_max+tau_0, dtype=jnp.float64)
    F_grid = make_F_grid(a0, b0, tau_0, tau_1, sigma)  # build once, reuse

    for i in range(len(data)):
            for j in range(len(data[i])):
                index_set.append((i,j))
    L_IJ=len(index_set)
    for step in range(steps*L_IJ):
        # Go thorugh the index set IJ by row and delete the current index
        del_index = index_set[step % L_IJ]
        # Update the current partition taking one step in the Gibbs sampler
        key, sub = jax.random.split(key)
        current_partition, key = Gibbs_partition_step(current_partition,del_index,data,tau_0,tau_1,sigma,a0, wa, b0, wb, F_grid, key=sub)
        if (step+1) % (verbose*L_IJ) == 0:
            print(f"Completed step {(step+1)/L_IJ} out of {steps} steps.")
        if step % (L_IJ) == 0:        # Every componente has been updated once. Append the new partition to the list
            partitions.append(copy.deepcopy(current_partition))
    return partitions, key


def Gibbs_partition_step(current_partition,del_index,data,tau_0,tau_1,sigma,a0, wa, b0, wb, F_grid, key):
    ''' This function updates current_partition to new_partition by taking one step in the Gibbs sampler.
    Arguments:
    del_index (tuple): the index of the element to be deleted from the current partition
    current_partition (partition): the current partition object
    Returns: (random_partition, new_key)
    '''
    # creating all possible new partitions by adding del_index to its own subset or to an existing subset
    # take into account that staying at the current partition has proportionality constant 1
    L=current_partition.get_length()
    par=current_partition.get_partition()
    singleton=False
    copy_par=copy.deepcopy(par) #this is a list, not a partition object
    for i in range(L):
        if not set([del_index]).isdisjoint(set(par[i])):
            idx=i
            if len(par[i])==1:
                singleton=True
    if singleton:
        copy_par.pop(idx) 
        par_list=[]
        for i in range(len(copy_par)):
            new_partition=partition_class.partition(copy.deepcopy(copy_par))
            new_partition.get_element(i).append(del_index)
            par_list.append(new_partition)
        # add "stay" as last candidate
        par_list.append(copy.deepcopy(current_partition))
        # compute weights
        probs = [densities.prob_new_partition(current_partition,cand,data,sigma,tau_0,tau_1,a0, wa, b0, wb, make_deriv_matrix=F_grid) for cand in par_list[:-1]]
        probs.append(1.0)  # stay weight
        w = jnp.asarray(probs, dtype=jnp.float64)
        w = w / jnp.clip(jnp.sum(w), a_min=1e-300)  # normalize
        # sample index using JAX key
        key, sub = jax.random.split(key)
        idx_draw = int(jax.random.categorical(sub, jnp.log(w)))
        random_partition=par_list[idx_draw]

    else:
        par_list = [None] * (L+1)  # Pre-allocate with correct size
         #remove del_index from its old subset, since its not a singleton the returned list/set has a least one more entry
        for i in range(len(copy_par)):###check that copy par has length L #The same partition as current partition should enter with probab 1
            if i == idx:
                continue  # Skip this iteration
            new_par=copy.deepcopy(copy_par)
            new_par[idx].remove(del_index)
            new_partition=partition_class.partition(new_par)
            new_partition.get_element(i).append(del_index)
            par_list[i]= new_partition
        # Staying at the current partition
        par_list[L]=copy.deepcopy(current_partition)
        # The partition where del_index goes into a new subset
        new_par=copy.deepcopy(copy_par)
        new_par[idx].remove(del_index)
        new_par.append([del_index])
        par_list[idx]=partition_class.partition(new_par)

        probs = [0.0] * (L+1)
        for i, cand in enumerate(par_list):
            if cand is None:
                probs[i] = 0.0
            elif i == L:
                probs[i] = 1.0  # stay weight
            else:
                probs[i] = float(densities.prob_new_partition(current_partition,cand,data,sigma,tau_0,tau_1,a0, wa, b0, wb, make_deriv_matrix=F_grid))
        w = jnp.asarray(probs, dtype=jnp.float64)
        w = w / jnp.clip(jnp.sum(w), a_min=1e-300)
        key, sub = jax.random.split(key)
        idx_draw = int(jax.random.categorical(sub, jnp.log(w)))
        random_partition=par_list[idx_draw]
    return random_partition, key


################ Helpers ####################
def draw_initialization(data, key=None):
    ''' Draw a uniform initialization for the conditional hitting scenario using JAX RNG.
    Returns (partition, key)
    '''
    if key is None:
        key = jax.random.PRNGKey(0)
    total = sum(len(row) for row in data)
    # sample an assignment label in {0,...,total-1} for each observation
    key, sub = jax.random.split(key)
    encoding = jax.random.randint(sub, shape=(total,), minval=0, maxval=total)
    partition_encoding=[]
    for s in range(total):
        subset=[]
        counter=0
        for i in range(len(data)):
            for j in range(len(data[i])):
                if int(encoding[counter])==s:
                    subset.append((i,j)) 
                counter+=1
        if len(subset)>0:
            partition_encoding.append(subset)
    return partition_class.partition(partition_encoding), key

def get_data_max(data):
    """Compact one-liner version"""
    all_values = [val for row in data for val in row]
    return max(all_values) if all_values else 1.0


# ############ Test non-parallel ###################
# steps=300
# tau_0=1.0   
# tau_1=0.5
# precision_a=12
# precision_b=12
# sigma=0.5

# np.random.seed(123)  # Fix NumPy seed
# data = [np.random.gamma(2.0, 2.0, size=6), np.random.gamma(1.5, 2.3, size=5)]

# key = jax.random.PRNGKey(42)  # Fix JAX seed

# from time import perf_counter

# # Non-parallel
# start = perf_counter()
# result, key_out = Gibbs_sampler_cond_hit_scen(steps, data, tau_0, tau_1, precision_a, precision_b, sigma, key=key)
# elapsed_nonparallel = perf_counter() - start
# print(f"Non-parallel: {elapsed_nonparallel:.4f} seconds total, {elapsed_nonparallel/steps:.4f} seconds per MCMC step")
# for i in range(len(result)):
#     print(result[i].get_partition())

# # Parallel
# key_parallel = jax.random.PRNGKey(42)  # Same JAX seed
# start = perf_counter()
# result_parallel, key_out_parallel = Gibbs_sampler_cond_hit_scen_parallel(steps, data, tau_0, tau_1, precision_a, precision_b, sigma, n_threads=4, key=key_parallel)
# elapsed_parallel = perf_counter() - start
# print(f"Parallel: {elapsed_parallel:.4f} seconds total, {elapsed_parallel/steps:.4f} seconds per MCMC step")
# for i, p in enumerate(result_parallel):
#     print(i, p.get_partition())

# ###parellelization does not seem to be faster even for 300 steps