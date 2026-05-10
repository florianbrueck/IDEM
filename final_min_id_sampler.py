#from typing import Sequence
#from collections.abc import Sequence
import jax
#import time
import jax.numpy as jnp
import numpy as np
import final_MCMC_extr_seq as MCMC_extr_seq
import final_partition_class as partition_class
import copy
#from scipy.stats import lognorm

from final_derivatives_laplace_trafo import (
    gauss_legendre_gamma_crm_jax,
    gauss_laguerre_gamma_crm_jax,
    log_laplace_trafo_margin_sorted_jit,
    mixed_each_coord_value_jvp,
    pack_sort_and_indices,
    make_F_grid
)

from final_densities import evaluate_density_mar
#################### Distribution of Y^(n) for the posterior ####################

def exponent_measure_min_id_post(locations,data, tau0, tau1, sigma, a0, wa,b0,wb):
    ''' This function evaluates the exponent measure of the prior or posterior.
    Arguments:
    locations: |IJ|-dimensional jnp.array of values where the exponent measure is evaluated (0s in locations do not change the output so they are the neutral element)
    data: list of lists, the list contains the rows of X_IJ with the i-th row given by the i-th list
    tau_0, tau_1: are the parameters of the rectangular and Dykstra-Laudt kernel
    precision_a: int, precision for Gauss-Laguerre quadrature
    precision_b: int, precision for Gauss-Legendre quadrature
    sigma: float>0, parameter of the Gamma Lévy process
    '''
    d=locations.shape[0]
    x_sorted_list=[]
    perm_list=[]
    X_IJ_sorted_list=[]
    perm_IJ_list=[]
    all_data_ind=[]
    mar_contr = jnp.asarray(0.0, dtype=jnp.float64)
    for i in range(d):
        for j in range(len(data[i])):
            all_data_ind.append((i,j))
    for row in range(d):
        # Collect X_{i,j} for which i= row
        X_IJ_row = jnp.array([data[i][j] for i, j in all_data_ind if i == row])
        x1=jnp.array([])

        X_IJ_sorted, perm_IJ=pack_sort_and_indices(x1,X_IJ_row)   
        X_IJ_sorted_list.append(X_IJ_sorted)
        perm_IJ_list.append(perm_IJ)

        # Collect all locations values for which i= row
        location_values = locations[row,:]
        X_row = jnp.concatenate([X_IJ_row, location_values])
        x_sorted, perm=pack_sort_and_indices(x1,X_row)   
        x_sorted_list.append(x_sorted) #must be X_row sorted
        perm_list.append(perm)
        #calculate marginal contribution
        #perm must be empty! Then evaluate_density_mar just returns the logarithm of the laplace trafo
        #evaluate_density_mar of empty or 0 argument returns 0
        # the difference below is then \int (1-prod_{(row,j) in I'J'}exp(-a 1_{locations(row,j)>=b}) [ prod_{(row,j) in IJ } exp(a-1_{X_{row,j} >= b}) ] ) rho(a) (da)\alpha_o^{\kappa_0}(db)
        mar_contr += - evaluate_density_mar(x_sorted, perm, tau0, tau1, sigma)+evaluate_density_mar(X_IJ_sorted, perm, tau0, tau1, sigma) #since evaluate_density_mar returns - \int ...
    dep_result = evaluate_density_dep_exp_meas_min_id(locations,x_sorted_list,perm_list,X_IJ_sorted_list,perm_IJ_list,a0, b0, wa, wb, tau0, tau1, sigma)
    dep_contr = jnp.asarray(dep_result, dtype=jnp.float64)

    return dep_contr+mar_contr

def survival_func_min_id(locations,data, tau0, tau1, sigma, a0, wa,b0,wb):
    ''' This function evaluates the survival functon of the min-id prior or posterior.
    Arguments:
    locations: |I'J'|-dimensional jnp.array of values where the exponent measure is evaluated (0s in locations do not change the output so they are the neutral element)
    data: list of lists, the list contains the rows of X_IJ with the i-th row given by the i-th list
    tau_0, tau_1: are the parameters of the rectangular and Dykstra-Laudt kernel
    precision_a: int, precision for Gauss-Laguerre quadrature
    precision_b: int, precision for Gauss-Legendre quadrature
    sigma: float>0, parameter of the Gamma Lévy process
    '''
    return jnp.exp(- exponent_measure_min_id_post(locations,data, tau0, tau1, sigma, a0, wa,b0,wb) )



# Evaluate integral using vectorized operations
def evaluate_density_dep_exp_meas_min_id(locations, x_sorted_list,perm_list,X_IJ_sorted_list,perm_IJ_list,a0, b0, w_a, w_b, tau0, tau1, sigma):
    Nb, Na = w_b.shape[0], w_a.shape[0]
    w_a_t = jnp.asarray(w_a/a0, dtype=jnp.float64).reshape(-1, 1)   # (Na, 1)
    w_b = jnp.asarray(w_b, dtype=jnp.float64).reshape(1, -1)     # (1, Nb)
    make_deriv_matrix = make_F_grid(a0,b0,tau0,tau1,sigma)   # function returning (Nb,Na) array
    #numerically stable product in log domain
    logM = jnp.zeros((Nb, Na), dtype=jnp.float64)
    #Clip log for numerical purposes at eps
    eps = jnp.asarray(1e-14, dtype=jnp.float64)
    for i, (x_sorted, perm, X_IJ_sorted, perm_IJ) in enumerate(zip(x_sorted_list, perm_list, X_IJ_sorted_list, perm_IJ_list)):
        #mixed_each_coord_value_jvp of empty argument for x or for x=0 returns 1
        if np.sum(locations[i,:])<=1e-14: #(numercally catch only zeros in locsations)
            #X_IJ_sorted=x_sorted (up to additional zeros which are neutral elements in the computation)
            #mixed_each_coord_value_jvp just returns the laplace trafo as perm is empty
            G = mixed_each_coord_value_jvp(make_deriv_matrix, X_IJ_sorted, perm_IJ)  # >= 0
            G_safe = jnp.clip(G, a_min=eps)
            logM = logM +  jnp.log(G_safe)
        else:
            #mixed_each_coord_value_jvp just returns the laplace trafo as perm is empty
            G = mixed_each_coord_value_jvp(make_deriv_matrix, X_IJ_sorted, perm_IJ)-mixed_each_coord_value_jvp(make_deriv_matrix, x_sorted, perm)  # >= 0
            G_safe = jnp.clip(G, a_min=eps)
            logM = logM +  jnp.log(G_safe)
    M = jnp.exp(logM)  # any zero factor → -inf → 0
    # M is now the (Nb, Na) array of products of derivatives
    const_gamma = jax.scipy.special.gamma(1 - sigma).astype(jnp.float64)

    return (w_b @ (M @ w_a_t) / const_gamma) [0,0] # scalar




def compute_survival_grid(margin,data, tau0, tau1, sigma, precision_a=36,precision_b=36, x_min= 0.001, x_max = 10.0, n_points= 1000, save_path= None) :
    """
    Compute survival function on a grid and optionally save results.
    
    Parameters:
    -----------
    margin : int
        Index of the margin to compute survival function for
    data : list of lists
        The data structure as used in survival_func_min_id
    tau0, tau1, sigma : float
        Model parameters
    a0, wa, b0, wb : arrays
        Quadrature nodes and weights
    x_min, x_max : float
        Grid boundaries
    n_points : int
        Number of grid points
    save_path : str, optional
        Path to save the grid and survival values as .npz file
        
    Returns:
    --------
    x_grid : jnp.ndarray
        Grid points where survival function was evaluated
    survival_values : jnp.ndarray
        Survival function values S(x) = P(X > x)
    """
    x_grid, exp_measure_values = compute_exp_measure_margin_grid(margin, data, tau0, tau1, sigma, precision_a, precision_b, x_min, x_max, n_points, save_path)
    return x_grid , jnp.exp(-exp_measure_values)

def compute_exp_measure_margin_grid(margin,data, tau0, tau1, sigma, precision_a=36,precision_b=36, x_min= 0.001, x_max = 10.0, n_points= 1000, save_path= None) :
    """
    Computes the exponent measure on a grid and optionally saves results.
    
    Parameters:
    -----------
    margin : int
        Index of the margin to compute survival function for
    data : list of lists
        The list contains the rows of X_IJ with the i-th row given by the i-th list
    tau0, tau1, sigma : float
        Model parameters
    a0, wa, b0, wb : arrays
        Quadrature nodes and weights
    x_min, x_max : float
        Grid boundaries
    n_points : int
        Number of grid points
    save_path : str, optional
        Path to save the grid and survival values as .npz file
        
    Returns:
    --------
    x_grid : jnp.ndarray
        Grid points where survival function was evaluated
    survival_values : jnp.ndarray
        Exponent measure values S(x) = -log(P(X > x))
    """
    # create quadrature nodes and weights
    a0, wa = gauss_laguerre_gamma_crm_jax(precision_a, -sigma)

    # Create grid
    x_grid = jnp.linspace(x_min, x_max, n_points)
    
    # Determine dimensions from data
    d = len(data)

    # Pre-allocate survival values array
    survival_values = jnp.zeros(n_points)
    
    print(f"Computing exponent measure on {n_points} grid points...")
    
    # Compute survival function for each grid point
    for i, x in enumerate(x_grid):
        if i % 100 == 0:
            print(f"Progress: {i}/{n_points}")
            
        # Create locations array with x on diagonal, 0 elsewhere
        # This assumes you want to evaluate \mu((0,x]) for the univariate case
        # Modify this part based on your specific needs
        locations = jnp.zeros((d, 1))
        locations = locations.at[margin, 0].set(x)  # Set margin-th row to x

        # Create b0, wb for current x
        b0, wb = gauss_legendre_gamma_crm_jax(precision_b,0,x+tau0)
    
        # Compute exponent measure at this point
        survival_val = exponent_measure_min_id_post(locations, data, tau0, tau1, sigma, a0, wa, b0, wb)
        survival_values = survival_values.at[i].set(survival_val)
    # Enforce strictly increasing exponent measure along the grid
    # 1) Make it non-decreasing to remove small numerical dips
    survival_values = jax.lax.associative_scan(jnp.maximum, survival_values)

    # # 2) Replace zero/negative increments with a tiny epsilon to ensure strict increase
    # eps = jnp.asarray(1e-12, dtype=jnp.float64)
    # diffs = survival_values[1:] - survival_values[:-1]
    # diffs_fixed = jnp.maximum(diffs, eps)
    # survival_values = jnp.concatenate([
    #     survival_values[:1],
    #     survival_values[:1] + jnp.cumsum(diffs_fixed)
    # ])

    print("Exponent measure computation completed.")
    
    # Save results if path provided
    if save_path:
        np.savez(save_path, 
                x_grid=np.array(x_grid), 
                survival_values=np.array(survival_values),
                tau0=tau0, tau1=tau1, sigma=sigma)
        print(f"Results saved to {save_path}")
    
    return x_grid, survival_values


################### Sampling functions ##################



import jax.random as jr

def _interp_linear_scalar(x, xp, fp):
    """
    Scalar linear interpolation: returns f(x) given monotone xp and fp.
    Assumes xp is strictly increasing and fp aligned with xp.
    """
    n = xp.shape[0]
    # Find interval i s.t. xp[i] <= x < xp[i+1]
    i = jnp.clip(jnp.searchsorted(xp, x, side='right') - 1, 0, n - 2)
    x0, x1 = xp[i], xp[i + 1]
    f0, f1 = fp[i], fp[i + 1]
    w = (x - x0) / jnp.maximum(x1 - x0, 1e-12)
    return f0 + w * (f1 - f0)

def F_inv_from_grid(s_vals, t_grid, F_grid):
    """
    Vectorized inverse of F using piecewise-linear interpolation of (F -> t).
    s_vals: shape (k,)
    Returns t such that F(t) ~= s (linear inverse on each grid segment).
    """
    n = F_grid.shape[0]
    # For each s, find i s.t. F[i] <= s < F[i+1]
    idx = jnp.clip(jnp.searchsorted(F_grid, s_vals, side='right') - 1, 0, n - 2)
    F0 = F_grid[idx]
    F1 = F_grid[idx + 1]
    t0 = t_grid[idx]
    t1 = t_grid[idx + 1]
    w = (s_vals - F0) / jnp.maximum(F1 - F0, 1e-12)
    return t0 + w * (t1 - t0)

def sample_nhpp_inverse(key, t_grid, F_grid, T=None):
    """
    Sample event times of a nonhomogeneous Poisson process (NHPP) on [0, T]
    by inverting a tabulated cumulative intensity F(t). 
    In our case this will be the marginal exponent measure of Y in the min-id posterior

    Args:
        key: jax.random.PRNGKey
        t_grid: (n,) strictly increasing times covering at least up to T
        F_grid: (n,) nondecreasing values F(t_grid)
        T: horizon (scalar). If None, uses T = t_grid[-1].

    Returns:
        event_times: (N,) sorted array of event times in [0, T]
    """
    if T is None:
        T = t_grid[-1]

    # Interpolate F(T) from the grid
    F_T = _interp_linear_scalar(T, t_grid, F_grid)
    F_T = jnp.clip(F_T, a_min=1e-13)  # guard

    # Draw number of events
    key_N, key_U = jr.split(key, 2)
    N = jr.poisson(key_N, lam=F_T, shape=()).astype(jnp.int32)

    if N == 0:
        return jnp.empty((0,), dtype=t_grid.dtype)
    else:
        # Sample order stats in transformed space: sort Uniform(0, F_T)
        s = jnp.sort(jr.uniform(key_U, (int(N),), minval=0.0, maxval=F_T))
        # Invert to times via piecewise-linear F^{-1}
        t = F_inv_from_grid(s, t_grid, F_grid)
        # Numerical guard to keep within [t_grid[0], T]
        t = jnp.clip(t, t_grid[0], T)
        return t



######################### Conditionaldistribution of min-id distribution ###############

######################### Algorithm Skeleton #########################

def inv_lambda_piecewise_linear(y, x_grid, lam_grid):
    """
    Invert Λ(x) given samples (x_grid, lam_grid) with lam_grid increasing.

    y: (...,) nonnegative
    returns x: (...,)
    """
    # Find right interval: lam[i] <= y < lam[i+1]
    # searchsorted returns index in [0..m]
    idx = jnp.searchsorted(lam_grid, y, side="right") - 1
    idx = jnp.clip(idx, 0, lam_grid.size - 2)

    lam0 = lam_grid[idx]
    lam1 = lam_grid[idx + 1]
    x0 = x_grid[idx]
    x1 = x_grid[idx + 1]

    # Linear interpolation parameter
    t = (y - lam0) / (lam1 - lam0 + 1e-13)
    return x0 + t * (x1 - x0)


def simulate_exact_min_id(key, data, grid_list , tau0, tau1, sigma, k=1, precision_a=36,precision_b=36, x_min= 0.001, x_max = 10.0, n_points= 500,steps=2500) :
    """ 
    Exact simulation of min-id process Y^(n) at locations (d x k), conditionally on data
    
    Proceeds iteratively over all locations to be simulated in three steps:
    1) Simulate points from the Poisson point process for each location
    2) For each point of the Poisson process simulate the corresponding extremal function
    3) Check if the extremal function is accepted or not
    Stop if the extremal function is accepted
    
    If data is empty we simulate from the prior.

    Args:
        key: jax.random.PRNGKey
        k: number of vectors to simulate
        data : list of lists, the data to condition on, if empty we simulate from the prior
        tau0, tau1, sigma: model parameters

    Returns:
        Y^{(n)}_k: jnp.ndarray of shape (d,k) of simulated min-id random vector
    """
    d= len(data)
    
    #grid_list = [ compute_exp_measure_margin_grid(i, data, tau0, tau1, sigma, precision_a, precision_b, x_min, x_max, n_points) for i in range(d) ]
    
    margin_lengths = [ len(data[i]) for i in range(d) ]
    # verify that we are actually looking at an increasing grid

    extr_fct_list = []
    # JAX-safe running min over accepted extremal functions
    Y_min = jnp.full((d, k), jnp.inf, dtype=jnp.float64)

    # Step 1: simulate initial point for margin 1 (index 0)
    key, kE = jr.split(key, 2)
    E0 = jr.exponential(kE, shape=())
    z = inv_lambda_piecewise_linear(E0, grid_list[0][0], grid_list[0][1])

    # Step 2: initial draw of extremal function Y ~ P_{s_1}(z, ·)
    key, key_sample = jr.split(key, 2)
    par = partition_class.partition(remove_empty_lists( [ [ (0,margin_lengths[0]) ], [(i, j) for i in range(d) for j in range(len(data[i]))]  ] )) 
    data_temp = copy.deepcopy(data)
    data_temp[0].append(float(z))  # set first location to z

    ###### Step 3
    extr_fct_MCMC = MCMC_extr_seq.MH_sampler_Z_l_cached(0,steps,data_temp,par,tau0,tau1,sigma,precision_a,precision_b,k,key_sample) # shape (d,k)
    extr_fct = extr_fct_MCMC[steps-1]
    extr_fct = extr_fct.at[0,0].set(z) # set first location to z
    extr_fct_list.append(extr_fct)
    Y_min = jnp.minimum(Y_min, extr_fct)  # update running min


    # Loop for margin 1 (index 0)
    for j in range(1,k):
        # start with a fresh E0 and z
        key, kE0 = jr.split(key)
        E0 = jr.exponential(kE0, shape=())
        z = inv_lambda_piecewise_linear(E0, grid_list[0][0], grid_list[0][1])

        current_min_at_j = Y_min[0, j]

        while z <= current_min_at_j:
            data_temp = copy.deepcopy(data)
            data_temp[0].append(float(z))  # set first location to z
            
            key, key_sample = jr.split(key)
            extr_fct_MCMC = MCMC_extr_seq.MH_sampler_Z_l_cached(0,steps,data_temp,par,tau0,tau1,sigma,precision_a,precision_b,k,key_sample) # shape (d,k)
            extr_fct_cand = extr_fct_MCMC[steps-1]
            if check_extr_fct(extr_fct_cand, Y_min, j, 0):
                extr_fct_cand = extr_fct_cand.at[0,j].set(z)# set location (i,j) to z (and implicitly drop the previous value at location (i,j))
                extr_fct_list.append(extr_fct_cand)
                Y_min = jnp.minimum(Y_min, extr_fct_cand)  # update running min
                break
            else:
                # update new z
                print("Rejected extremal function at margin 0, location ", j, ", resampling z...")
                key, kE0 = jr.split(key)
                E0 = E0 + jr.exponential(kE0, shape=())
                if E0 <= grid_list[0][1][-1]:
                    z = inv_lambda_piecewise_linear(E0, grid_list[0][0], grid_list[0][1])
                else:
                    x_max_temp=x_max
                    r_max = grid_list[0][1][-1]
                    while E0 > r_max:
                        x_max_temp=x_max_temp +x_max/2 
                        grid,values = compute_exp_measure_margin_grid(0, data, tau0, tau1, sigma, precision_a, precision_b, x_min,  x_max_temp, n_points)
                        r_max = values[-1]
                    z = inv_lambda_piecewise_linear(E0, grid, values)

    # Loop for margins 2,...,d    
    for i in range(1,d):
        par = partition_class.partition( remove_empty_lists( [ [(i,margin_lengths[i]) ] , [ (l, m) for l in range(d) for m in range(len(data[l])) ] ] )) 
        for j in range(0,k):
            # start with a fresh E0 and z
            key, kE0 = jr.split(key)
            E0 = jr.exponential(kE0, shape=())
            z = inv_lambda_piecewise_linear(E0, grid_list[i][0], grid_list[i][1])

            current_min_at_j = Y_min[i, j]  # read current min for margin i, column j
            while z <= current_min_at_j:
                data_temp = copy.deepcopy(data)
                data_temp[i].append(float(z))  # set first location to z
                
                key, key_sample = jr.split(key)
                extr_fct_MCMC = MCMC_extr_seq.MH_sampler_Z_l_cached(0,steps,data_temp,par,tau0,tau1,sigma,precision_a,precision_b,k,key_sample) # shape (d,k) ,,, debug into and check that only derivative w.r.t to z location is taken
                extr_fct_cand = extr_fct_MCMC[steps-1]
                if check_extr_fct(extr_fct_cand, Y_min, j, i):
                    extr_fct_cand = extr_fct_cand.at[i,j].set(z) # set location (i,j) to z (and implicitly drop the previous value at location (i,j))
                    extr_fct_list.append(extr_fct_cand)
                    Y_min = jnp.minimum(Y_min, extr_fct_cand)  # update running min
                    break
                else:
                    # update new z
                    #print("Rejected extremal function at margin ", i, ", location ", j, ", resampling z...")
                    key, kE0 = jr.split(key)
                    E0 = E0 + jr.exponential(kE0, shape=())
                    if E0 <= grid_list[i][1][-1]:
                        z = inv_lambda_piecewise_linear(E0, grid_list[i][0], grid_list[i][1])
                    else:
                        x_max_temp=x_max
                        r_max = grid_list[i][1][-1]
                        while E0 > r_max:
                            x_max_temp=x_max_temp +x_max/2 
                            grid,values = compute_exp_measure_margin_grid(i, data, tau0, tau1, sigma, precision_a, precision_b, x_min,  x_max_temp, n_points)
                            r_max = values[-1]
                        z = inv_lambda_piecewise_linear(E0, grid, values)
   
    return Y_min



def check_extr_fct(extr_fct_cand, Y_min, j , margin):
    """
    Check if the candidate extremal function is accepted or not.
    An extremal function is accepted if it is larger than all previously accepted extremal functions at locations k<j in the margin-th row and in all locations of the rows < margin-th row
    
    Args:
        extr_fct_cand: jnp.ndarray of shape (d,k) candidate extremal function
        extr_fct_list: list of previously accepted extremal functions
        j: index of the location to check
    """
    d = extr_fct_cand.shape[0]
    n = extr_fct_cand.shape[1]
    for i in range(0,d):
        if i < margin:
            for k in range(0,n):
                if extr_fct_cand[i,k] <= Y_min[i,k]:
                    return False
        if i == margin:
            for k in range(0,j):
                if extr_fct_cand[i,k] <= Y_min[i,k]:
                    return False
    return True




def remove_empty_lists(list_of_lists):
    """
    Return a new list-of-lists with all empty inner lists removed.

    Example:
        [[(0, 1)], [], [(2, 3), (2, 4)]] -> [[(0, 1)], [(2, 3), (2, 4)]]
    """
    return [subset for subset in list_of_lists if len(subset) > 0]


#### Simulate from Prior


# #Assume you have your data and parameters ready
# data = [[],[]]  # Example data
# tau0, tau1, sigma = 1.01, 0.8, 0.5
# a0, wa = gauss_laguerre_gamma_crm_jax(20, -sigma)
# b0, wb = gauss_legendre_gamma_crm_jax(20, 0.0, 10.0)

# #Step 1: Compute survival function on grid
# x_grid_0, survival_vals_0 = compute_survival_grid(0,
#     data, tau0, tau1, sigma, precision_a=12, precision_b=12,
#     x_min=0.1, x_max=10.0, n_points=500,
#     save_path='marginal_prior_survival_function_0.npz'
# )

# #Step 1: Compute survival function on grid
# x_grid_1, survival_vals_1 = compute_survival_grid(1,
#     data, tau0, tau1, sigma, precision_a=12, precision_b=12,
#     x_min=0.1, x_max=10.0, n_points=500,
#     save_path='marginal_prior_survival_function_1.npz'
# )

# #Step 2: Generate samples
# key = jax.random.PRNGKey(123)
# # samples = sample_from_survival_function(x_grid_1, survival_vals_1, 1000, key)

# # #Or load and sample in one step:
# # samples = load_and_sample('marginal_prior_survival_function_1.npz', 1000, key)

# # print(f"Generated {len(samples)} samples")
# # print(f"Sample statistics: mean={jnp.mean(samples):.3f}, std={jnp.std(samples):.3f}")
    

# # # Plot your computed survival functions
# # print("Plotting survival function for margin 0:")
# # plot_survival_function(x_grid_0, survival_vals_0, margin=0, save_path='marginal_prior_survival_function_0.png')

# # print("Plotting survival function for margin 1:")
# # plot_survival_function(x_grid_1, survival_vals_1, margin=1, save_path='marginal_prior_survival_function_1.png')

# # # Compare both margins on the same plot
# # plt.figure(figsize=(10, 6))
# # plt.plot(np.array(x_grid_0), np.array(survival_vals_0), 'b-', linewidth=2, label='Margin 0')
# # plt.plot(np.array(x_grid_1), np.array(survival_vals_1), 'r--', linewidth=2, label='Margin 1')
# # plt.grid(True, alpha=0.3)
# # plt.xlabel('x', fontsize=12)
# # plt.ylabel('Survival Probability', fontsize=12)
# # plt.title('Survival Functions - Both Margins', fontsize=14)
# # plt.legend()
# # plt.ylim(0, 1)
# # plt.tight_layout()
# # plt.savefig('prior_survival_comparison.png', dpi=300, bbox_inches='tight')
# # plt.show()

# # # Plot samples with survival function
# # print("Plotting samples with survival function:")
# # plot_survival_and_samples(x_grid_1, survival_vals_1, samples, margin=1, 
# #                          save_path='survival_with_samples.png')

# # # Validate the survival function
# # print("\nSurvival function validation:")
# # print(f"Margin 0 - Min: {np.min(survival_vals_0):.6f}, Max: {np.max(survival_vals_0):.6f}")
# # print(f"Margin 1 - Min: {np.min(survival_vals_1):.6f}, Max: {np.max(survival_vals_1):.6f}")
# # print(f"Is decreasing (Margin 0): {np.all(np.diff(survival_vals_0) <= 1e-10)}")
# # print(f"Is decreasing (Margin 1): {np.all(np.diff(survival_vals_1) <= 1e-10)}")

# # # Check if both margins give similar results (as expected from your comment)
# # difference = np.abs(np.array(survival_vals_0) - np.array(survival_vals_1))
# # print(f"Max difference between margins: {np.max(difference):.6f}")
# # print(f"Are margins nearly identical: {np.allclose(survival_vals_0, survival_vals_1, rtol=1e-6)}")


# #Step 1: Compute survival function on grid
# x_grid_e_0, exp_meas_vals_0 = compute_exp_measure_margin_grid(0,
#     data, tau0, tau1, sigma, precision_a=12, precision_b=12,
#     x_min=0.1, x_max=10.0, n_points=500,
#     save_path='exp_measure_function_0.npz'
# )

# #Step 1: Compute survival function on grid
# x_grid_e_1, exp_meas_vals_1 = compute_exp_measure_margin_grid(1,
#     data, tau0, tau1, sigma, precision_a=12, precision_b=12,
#     x_min=0.1, x_max=10.0, n_points=500,
#     save_path='exp_measure_function_1.npz'
# )

# n_samples=1000
# sample_0=[]
# sample_1=[]
# for i in range(n_samples):
#     key, subkey = jax.random.split(key)
#     sample_0.append(jnp.min(sample_nhpp_inverse(subkey, x_grid_e_0, exp_meas_vals_0, T=None)))
#     sample_1.append(jnp.min(sample_nhpp_inverse(subkey, x_grid_e_1, exp_meas_vals_1, T=None)))

# sample_0=jnp.array(sample_0)
# sample_1=jnp.array(sample_1)

# plot_survival_and_samples(x_grid_0, survival_vals_0, sample_0, margin=1, 
#                          save_path='survival_with_exp_measure_sample_0.png')

# plot_survival_and_samples(x_grid_1, survival_vals_1, sample_1, margin=1, 
#                          save_path='survival_with_exp_measure_sample_1.png')


# ##### Posterior test


# #Assume you have your data and parameters ready
# data = [[6.0, 5.0], [2.0, 1.0]]  # Example data
# tau0, tau1, sigma = 1.01, 0.8, 0.5
# a0, wa = gauss_laguerre_gamma_crm_jax(20, -sigma)
# b0, wb = gauss_legendre_gamma_crm_jax(20, 0.0, 10.0)

# #Step 1: Compute survival function on grid
# x_grid_0, survival_vals_0 = compute_survival_grid(0,
#     data, tau0, tau1, sigma, precision_a=12, precision_b=12,
#     x_min=0.1, x_max=10.0, n_points=1000,
#     save_path='survival_function_0.npz'
# )

# #Step 1: Compute survival function on grid
# x_grid_1, survival_vals_1 = compute_survival_grid(1,
#     data, tau0, tau1, sigma, precision_a=12, precision_b=12,
#     x_min=0.1, x_max=10.0, n_points=1000,
#     save_path='survival_function_1.npz'
# )



# ##should be identical to survival_vals_0

# #Step 2: Generate samples
# key = jax.random.PRNGKey(123)
# samples = sample_from_survival_function(x_grid_1, survival_vals_1, 2000, key)

# #Or load and sample in one step:
# samples = load_and_sample('survival_function_1.npz', 2000, key)

# print(f"Generated {len(samples)} samples")
# print(f"Sample statistics: mean={jnp.mean(samples):.3f}, std={jnp.std(samples):.3f}")
    


# #  Plot your computed survival functions
# print("Plotting survival function for margin 0:")
# plot_survival_function(x_grid_0, survival_vals_0, margin=0, save_path='survival_margin_0.png')

# print("Plotting survival function for margin 1:")
# plot_survival_function(x_grid_1, survival_vals_1, margin=1, save_path='survival_margin_1.png')

# # Compare both margins on the same plot
# plt.figure(figsize=(10, 6))
# plt.plot(np.array(x_grid_0), np.array(survival_vals_0), 'b-', linewidth=2, label='Margin 0')
# plt.plot(np.array(x_grid_1), np.array(survival_vals_1), 'r--', linewidth=2, label='Margin 1')
# plt.grid(True, alpha=0.3)
# plt.xlabel('x', fontsize=12)
# plt.ylabel('Survival Probability', fontsize=12)
# plt.title('Survival Functions - Both Margins', fontsize=14)
# plt.legend()
# plt.ylim(0, 1)
# plt.tight_layout()
# plt.savefig('survival_comparison.png', dpi=300, bbox_inches='tight')
# plt.show()

# # Plot samples with survival function
# print("Plotting samples with survival function:")
# plot_survival_and_samples(x_grid_1, survival_vals_1, samples, margin=1, 
#                          save_path='survival_with_samples.png')

# # Validate the survival function
# print("\nSurvival function validation:")
# print(f"Margin 0 - Min: {np.min(survival_vals_0):.6f}, Max: {np.max(survival_vals_0):.6f}")
# print(f"Margin 1 - Min: {np.min(survival_vals_1):.6f}, Max: {np.max(survival_vals_1):.6f}")
# print(f"Is decreasing (Margin 0): {np.all(np.diff(survival_vals_0) <= 1e-10)}")
# print(f"Is decreasing (Margin 1): {np.all(np.diff(survival_vals_1) <= 1e-10)}")

# # Check if both margins give similar results (as expected from your comment)
# difference = np.abs(np.array(survival_vals_0) - np.array(survival_vals_1))
# print(f"Max difference between margins: {np.max(difference):.6f}")
# print(f"Are margins nearly identical: {np.allclose(survival_vals_0, survival_vals_1, rtol=1e-6)}")






# # Plot hazard rates for each margin
# print("Plotting hazard rate for margin 0:")
# plot_hazard_rate(x_grid_0, survival_vals_0, margin=0, save_path='hazard_margin_0.png')

# print("Plotting hazard rate for margin 1:")
# plot_hazard_rate(x_grid_1, survival_vals_1, margin=1, save_path='hazard_margin_1.png')

# # Plot survival and hazard together
# print("Plotting survival and hazard together:")
# plot_survival_and_hazard(x_grid_0, survival_vals_0, margin=0, save_path='survival_hazard_margin_0.png')
# plot_survival_and_hazard(x_grid_1, survival_vals_1, margin=1, save_path='survival_hazard_margin_1.png')

# # Compare hazard rates between margins
# print("Comparing hazard rates between margins:")
# plot_hazard_comparison(x_grid_0, survival_vals_0, x_grid_1, survival_vals_1, 
#                       save_path='hazard_comparison.png')

# # Validate hazard rates
# hazard_0 = compute_hazard_rate(x_grid_0, survival_vals_0)
# hazard_1 = compute_hazard_rate(x_grid_1, survival_vals_1)

# print("\nHazard rate validation:")
# print(f"Margin 0 - Mean: {np.mean(hazard_0):.3f}, Max: {np.max(hazard_0):.3f}")
# print(f"Margin 1 - Mean: {np.mean(hazard_1):.3f}, Max: {np.max(hazard_1):.3f}")

# if len(hazard_0) == len(hazard_1):
#     hazard_diff = np.abs(hazard_0 - hazard_1)
#     print(f"Max hazard difference between margins: {np.max(hazard_diff):.6f}")
#     print(f"Are hazards nearly identical: {np.allclose(hazard_0, hazard_1, rtol=1e-3)}")


