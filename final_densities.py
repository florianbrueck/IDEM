from functools import lru_cache
import jax
import jax.numpy as jnp
import numpy as np


from final_derivatives_laplace_trafo import (
    log_laplace_trafo_margin_sorted_jit,
    mixed_each_coord_value_jvp,
    mixed_each_coord_value_jvp_wrt_x,
    pack_sort_and_indices,
    make_F_grid,
    make_F_grid_b0_dep
)


################### Density evaluations ####################


# Evaluate integral using vectorized operations
def evaluate_density_dep_b0_dep(x_sorted_list,perm_list,a0, b0, w_a, w_b, tau0, tau1, sigma,make_deriv_matrix=None):
    Nb, Na = w_b.shape[0], w_a.shape[0]
    w_a_t = jnp.asarray(w_a/a0, dtype=jnp.float64).reshape(-1, 1)   # (Na, 1)
    w_b = jnp.asarray(w_b, dtype=jnp.float64).reshape(1, -1)     # (1, Nb)
    if make_deriv_matrix is None:
        make_deriv_matrix = make_F_grid_b0_dep(a0, tau0, tau1, sigma)  # jitted; build once if not provided,function returning (Nb,Na) array
    # M=jnp.ones((w_b.shape[1],w_a_t.shape[0]),dtype=jnp.float64)
    # for (x_sorted,perm) in zip(x_sorted_list,perm_list):
    #     G = mixed_each_coord_value_jvp(make_deriv_matrix, x_sorted, perm)
    #     M=M*G #entry wise multiplication
    #numerically stable product in log domain
    logM = jnp.zeros((Nb, Na), dtype=jnp.float64)
    for (x_sorted,perm) in zip(x_sorted_list,perm_list):
        G = mixed_each_coord_value_jvp_wrt_x(make_deriv_matrix, x_sorted, perm, b0)  # >= 0
        logM = logM +  jnp.log(G)
    M = jnp.exp(logM)  # any zero factor → -inf → 0
    # M is now the (Nb, Na) array of products of derivatives
    const_gamma = jax.scipy.special.gamma(1 - sigma).astype(jnp.float64)

    return (w_b @ (M @ w_a_t) / const_gamma) [0,0] # scalar


# Evaluate integral using vectorized operations
def evaluate_density_dep(x_sorted_list,perm_list,a0, b0, w_a, w_b, tau0, tau1, sigma,make_deriv_matrix=None):
    Nb, Na = w_b.shape[0], w_a.shape[0]
    w_a_t = jnp.asarray(w_a/a0, dtype=jnp.float64).reshape(-1, 1)   # (Na, 1)
    w_b = jnp.asarray(w_b, dtype=jnp.float64).reshape(1, -1)     # (1, Nb)
    if make_deriv_matrix is None:
        make_deriv_matrix = make_F_grid(a0, b0, tau0, tau1, sigma)  # jitted; build once if not provided,function returning (Nb,Na) array
    # M=jnp.ones((w_b.shape[1],w_a_t.shape[0]),dtype=jnp.float64)
    # for (x_sorted,perm) in zip(x_sorted_list,perm_list):
    #     G = mixed_each_coord_value_jvp(make_deriv_matrix, x_sorted, perm)
    #     M=M*G #entry wise multiplication
    #numerically stable product in log domain
    logM = jnp.zeros((Nb, Na), dtype=jnp.float64)
    for (x_sorted,perm) in zip(x_sorted_list,perm_list):
        G = mixed_each_coord_value_jvp(make_deriv_matrix, x_sorted, perm)  # >= 0
        logM = logM +  jnp.log(G)
    M = jnp.exp(logM)  # any zero factor → -inf → 0
    # M is now the (Nb, Na) array of products of derivatives
    const_gamma = jax.scipy.special.gamma(1 - sigma).astype(jnp.float64)

    return (w_b @ (M @ w_a_t) / const_gamma) [0,0] # scalar



# def evaluate_density_mar(x_sorted,perm, tau0, tau1, sigma):
#     """Evaluate density using cached JIT functions"""
#     # returns the evalution of one density integral for given x_sorted, perm
#     # x_sorted: jnp array of sorted x values
#     # prem encodes the indices w.r.t. which the derivative is taken

#     f_mar_jit = log_laplace_trafo_margin_sorted_jit(tau0, tau1, sigma)
#     I = mixed_each_coord_value_jvp(f_mar_jit, x_sorted, perm)
#     return I 


@lru_cache(maxsize=None)
def _cached_f_mar(tau0: float, tau1: float, sigma: float):
    # build once per parameter triple
    return log_laplace_trafo_margin_sorted_jit(tau0, tau1, sigma)

def evaluate_density_mar(x_sorted, perm, tau0, tau1, sigma):
    # use cached jitted function
    f_mar_jit = _cached_f_mar(float(tau0), float(tau1), float(sigma))
    return mixed_each_coord_value_jvp(f_mar_jit, x_sorted, perm)

############# Proposal densities for conditional hitting functions ##############
def prob_new_partition(current_partition, new_partition, data, sigma, tau0, tau1, a0, wa, b0, wb, make_deriv_matrix=None):
    """
    Calculate the probability ratio between new and current partition by evaluating
    Laplace transform derivatives at the difference sets.
    
    Parameters:
    -----------
    current_partition, new_partition : partition objects
    data : data as list of lists
    tau0, tau1 : float - kernel parameters
    a0,wa,b0,wa: weights and nodes for the numerical integration to evalute the dependence integral
    sigma, theta : float - Gamma Lévy process parameters
    
    Returns:
    --------
    float : probability ratio (up to proportionality constant)
    """
    
    # Calculate the at most 4 difference sets between current and new partition
    idx_enum, enum_par=new_partition.partition_intersect_diff(current_partition)
    idx_denom, denom_par =current_partition.partition_intersect_diff(new_partition)

    d=len(data)
    # Accumulate in log-domain for numerical stability
    log_enumerator = jnp.asarray(0.0, dtype=jnp.float64)
    log_denominator = jnp.asarray(0.0, dtype=jnp.float64)
    eps = jnp.asarray(1e-13, dtype=jnp.float64)

    #### Calculate the enumerator #### 
    for idx in range(len(idx_enum)):
        x_sorted_list = []
        perm_list = []
        mar_cont = jnp.asarray(0.0, dtype=jnp.float64)  # JAX scalar
        for row in range(d):
            diffset=new_partition.get_element(idx_enum[idx])
            L_diffset=len(diffset)
            X_Theta_l_row = [data[i][j] for i, j in diffset if i==row]
            diffset_comp=new_partition.get_complement(idx_enum[idx])
            X_Theta_l_row_comp=[data[i][j] for i, j in [tuples for tuples_list in diffset_comp for tuples in tuples_list] if i==row]
            x1=jnp.array(X_Theta_l_row)
            z=jnp.array(X_Theta_l_row_comp)
            x_sorted, perm=pack_sort_and_indices(x1,z)   
            x_sorted_list.append(x_sorted)
            perm_list.append(perm)
            # if diffset subset i x n then we need the marginal contribution, otherwise we dont
            if len(X_Theta_l_row) == L_diffset:
                # Keep as JAX array operation
                mar_result = evaluate_density_mar(x_sorted, perm, tau0, tau1, sigma)
                mar_cont = jnp.asarray(mar_result, dtype=jnp.float64)
        
        dep_result = evaluate_density_dep(x_sorted_list,perm_list,a0, b0, wa, wb, tau0, tau1, sigma, make_deriv_matrix=make_deriv_matrix)
        dep_contr = jnp.asarray(dep_result, dtype=jnp.float64)
        
        # Stable accumulation in log-domain
        term = dep_contr + mar_cont
        term = jnp.clip(term, a_min=eps)
        log_enumerator = log_enumerator + jnp.log(term)
    
    #### Calculate the denominator #### 
    for idx in range(len(idx_denom)):
        x_sorted_list=[]
        perm_list=[]
        mar_cont = jnp.asarray(0.0, dtype=jnp.float64) 
        for row in range(d): ### Check that range is correct here
            diffset=current_partition.get_element(idx_denom[idx])
            L_diffset=len(diffset)
            # collect x_{i,j} for which (i,j) in theta_l
            x1= jnp.array([data[i][j] for i, j in diffset if i==row])
            diffset_comp=current_partition.get_complement(idx_denom[idx])
            # collect x_{i,j} for which (i,j) in theta_l^c
            z=jnp.array([data[i][j] for i, j in [tuples for tuples_list in diffset_comp for tuples in tuples_list] if i==row])
            x_sorted, perm=pack_sort_and_indices(x1,z)   
            x_sorted_list.append(x_sorted)
            perm_list.append(perm)
            # if theta_l is a subset if (row x n) then we need the marginal contribution, otherwise we dont
            if len(x1)==L_diffset:# theta_l is subset of (row x n)
                mar_result = evaluate_density_mar(x_sorted, perm, tau0, tau1, sigma)
                mar_cont = jnp.asarray(mar_result, dtype=jnp.float64)
        dep_result = evaluate_density_dep(x_sorted_list,perm_list,a0, b0, wa, wb, tau0, tau1, sigma, make_deriv_matrix=make_deriv_matrix)
        dep_contr = jnp.asarray(dep_result, dtype=jnp.float64)
        # Stable accumulation in log-domain
        term = dep_contr + mar_cont
        term = jnp.clip(term, a_min=eps)
        log_denominator = log_denominator + jnp.log(term)
              
    # Return ratio
    return jnp.exp(log_enumerator - log_denominator)




############# Proposal densities for conditional extremal functions ##############
def lognorm_logpdf_jax_vectorized(x, s, scale):
    """Vectorized lognormal log-pdf with underlying Normal std = s."""
    # jnp needed imports assumed at module level
    return (-jnp.log(x) - jnp.log(s) - 0.5 * jnp.log(2 * jnp.pi)
            - 0.5 * ((jnp.log(x) - jnp.log(scale)) / s) ** 2)



def log_transition_density_MH_Z_l_jax(locations_prop, locations_current, l, partition, sd_norm=1.0):
    ''' Calculates the transition density from the current state to the proposed state in the simulation of Z_l. When Theta_l contains only one row index the return log-density if correct.
    When Theta_l contains more than one row index the returned log-density is only correct up to a constant (jnp.log(1/2) should be 0). The factor then cancels out when the density ratio is calculated in the MH step.
    The model we use is throwing a Bernoulli random variable to decide whether we have all finite values, called state F, 
    and otherwise only the row corresponding to Theta_l is finite, called state I.
    Given the state we propose new values based on i.i.d. lognormal distributions with scale parameter location_current (using the scipy.stats.lognorm function notation).
    If the current value at a sight is infinite we choose scale=1 as a default, which corresponds to centering of the normal distribution at 0.
    sd_norm is the standard deviation of the underlying normal distribution used in the lognormal proposal.
    Currently we always add log(1/2) even though when Theta_l contains more than one row index this is wrong as there is no Bernoulli draw (all values are finite). This is still okay, since the factor cancels in the MH ratio.'''
    
    # Convert to JAX arrays
    locations_current = jnp.asarray(locations_current, dtype=jnp.float64) # True if we are currently in state F
    locations_prop = jnp.asarray(locations_prop, dtype=jnp.float64) # True if proposal in state F
    s_val = jnp.asarray(sd_norm, dtype=jnp.float64)

    current_state = jnp.all(jnp.isfinite(locations_current))
    prop_state = jnp.all(jnp.isfinite(locations_prop))
    row_index = partition.get_element(l)[0][0]

    if current_state:
        if prop_state:
            array = lognorm_logpdf_jax_vectorized(locations_prop, s_val, locations_current)
            dens = jnp.log(1/2) + jnp.sum(array)
        else:
            array_row = lognorm_logpdf_jax_vectorized(locations_prop[row_index, :], s_val, locations_current[row_index, :])
            dens = jnp.log(1/2) + jnp.sum(array_row)
    else:
        if prop_state:
            log_sum = jnp.asarray(0.0, dtype=jnp.float64)
            for i in range(locations_current.shape[0]):
                if i == row_index:
                    # Use current values as scale for finite row
                    array_row = lognorm_logpdf_jax_vectorized(locations_prop[i, :], s_val, locations_current[i, :])
                else:
                    # Use scale=1 for other rows
                    array_row = lognorm_logpdf_jax_vectorized(locations_prop[i, :], s_val, 1.0)
                log_sum = log_sum + jnp.sum(array_row)
            dens = jnp.log(1/2) + log_sum
        else:
            array_row = lognorm_logpdf_jax_vectorized(locations_prop[row_index, :], s_val, locations_current[row_index, :])
            dens = jnp.log(1/2) + jnp.sum(array_row)
    return dens




def dens_Z_l_jax(l, locations, data, Theta, tau_0, tau_1, sigma,a0,wa,b0,wb):
    ''' This function evaluates the density of Z^{(n,l)}(x_k) at the locations x_k.
    Arguments:
    locations: 2-dimensional jnp.array of values where the density is evaluated 
    data: list of lists, the list contains the rows of X_IJ with the i-th row given by the i-th list
    Theta: partition object, an object of the partition class representing the conditional hitting scenario
    tau_0, tau_1: are the parameters of the rectangular and Dykstra-Laudt kernel
    precision_a: int, precision for Gauss-Laguerre quadrature
    precision_b: int, precision for Gauss-Legendre quadrature
    sigma: float>0, parameter of the Gamma Lévy process
    '''
    
    # Convert to JAX arrays
    locations = jnp.asarray(locations, dtype=jnp.float64)

    current_state = jnp.all(jnp.isfinite(locations))  # True if we are currently in state F
    row_index = Theta.get_element(l)[0][0] 
    

    # Get the subset Theta_l for this specific l
    Theta_l = Theta.get_element(l)
    d = len(data)
    
    # Prepare data structures for integral evaluation
    x_sorted_list = []
    perm_list = []
    marginal_contr = jnp.asarray(0.0, dtype=jnp.float64)
    

    if current_state:
        # Process each row (margin) of the data
        for row in range(d):
            L_diffset = len(Theta_l)
            
            # Collect x_{i,j} for which (i,j) in Theta_l for this row
            X_Theta_l_row = jnp.array([data[i][j] for i, j in Theta_l if i == row])
            location_values = locations[row,:]
            X_Theta_l_row_combined = jnp.concatenate([X_Theta_l_row, location_values])
            
            # Collect x_{i,j} for which (i,j) in Theta_l^c (complement) for this row
            diffset_comp = Theta.get_complement(l)
            X_Theta_l_row_comp = [data[i][j] for i, j in [tuples for tuples_list in diffset_comp for tuples in tuples_list] if i == row]
            
            # Convert to JAX arrays and sort
            x1 = X_Theta_l_row_combined
            z = jnp.array(X_Theta_l_row_comp)
            x_sorted, perm = pack_sort_and_indices(x1, z) #check pack and sort for empty x1 ( even though it must be non-empty here)
            
            x_sorted_list.append(x_sorted)
            perm_list.append(perm)
            
            # Check if Theta_l is a subset of {row} × {1,...,n}
            # If so, we need the marginal contribution
            if len(X_Theta_l_row) == L_diffset:
                mar_result = evaluate_density_mar(x_sorted, perm, tau_0, tau_1, sigma)
                marginal_contr = jnp.asarray(mar_result, dtype=jnp.float64)
        

        # Evaluate dependence integral
        dep_contr = evaluate_density_dep(x_sorted_list, perm_list, a0, b0, wa, wb, tau_0, tau_1, sigma)
        dep_contr = jnp.asarray(dep_contr, dtype=jnp.float64)
        
        # Total density is sum of marginal and dependence contributions
        density = marginal_contr + dep_contr
    
    else: #Only row row_index contains finite values, the rest is infinite
            
        # Collect x_{i,j} for which (i,j) in Theta_l for this row
        X_Theta_l_row = jnp.array([data[i][j] for i, j in Theta_l if i == row_index])
        location_values = locations[row_index,:]
        X_Theta_l_row_combined = jnp.concatenate([X_Theta_l_row, location_values])
        
        # Collect x_{i,j} for which (i,j) in Theta_l^c (complement) for this row
        diffset_comp = Theta.get_complement(l)
        X_Theta_l_row_comp = [data[i][j] for i, j in [tuples for tuples_list in diffset_comp for tuples in tuples_list] if i == row_index]
            
        # Convert to JAX arrays and sort
        x1 = X_Theta_l_row_combined
        z = jnp.array(X_Theta_l_row_comp)
        x_sorted, perm = pack_sort_and_indices(x1, z)

        # Use vectorized operation for the row
        mar_result = evaluate_density_mar(x_sorted, perm, tau_0, tau_1, sigma)
        marginal_contr = jnp.asarray(mar_result, dtype=jnp.float64)

        density=marginal_contr

    
    return density


def dens_Z_l_jax_cached(l, locations, data, Theta, tau_0, tau_1, sigma, a0, wa, b0, wb, make_deriv_matrix=None):
    ''' This function evaluates the density of Z^{(n,l)}(x_k) at the locations x_k.
    Arguments:
    locations: 2-dimensional jnp.array of values where the density is evaluated 
    data: list of lists, the list contains the rows of X_IJ with the i-th row given by the i-th list
    Theta: partition object, an object of the partition class representing the conditional hitting scenario
    tau_0, tau_1: are the parameters of the rectangular and Dykstra-Laudt kernel
    precision_a: int, precision for Gauss-Laguerre quadrature
    precision_b: int, precision for Gauss-Legendre quadrature
    sigma: float>0, parameter of the Gamma Lévy process
    '''
    
    # Convert to JAX arrays
    locations = jnp.asarray(locations, dtype=jnp.float64)

    current_state = jnp.all(jnp.isfinite(locations))  # True if we are currently in state F
    row_index = Theta.get_element(l)[0][0] 
    

    # Get the subset Theta_l for this specific l
    Theta_l = Theta.get_element(l)
    d = len(data)
    
    # Prepare data structures for integral evaluation
    x_sorted_list = []
    perm_list = []
    marginal_contr = jnp.asarray(0.0, dtype=jnp.float64)
    

    if current_state:
        # Process each row (margin) of the data
        for row in range(d):
            L_diffset = len(Theta_l)
            
            # Collect x_{i,j} for which (i,j) in Theta_l for this row
            X_Theta_l_row = jnp.array([data[i][j] for i, j in Theta_l if i == row])
            location_values = locations[row,:]
            X_Theta_l_row_combined = jnp.concatenate([X_Theta_l_row, location_values])
            
            # Collect x_{i,j} for which (i,j) in Theta_l^c (complement) for this row
            diffset_comp = Theta.get_complement(l)
            X_Theta_l_row_comp = [data[i][j] for i, j in [tuples for tuples_list in diffset_comp for tuples in tuples_list] if i == row]
            
            # Convert to JAX arrays and sort
            x1 = X_Theta_l_row_combined
            z = jnp.array(X_Theta_l_row_comp)
            x_sorted, perm = pack_sort_and_indices(x1, z) #check pack and sort for empty x1 ( even though it must be non-empty here)
            
            x_sorted_list.append(x_sorted)
            perm_list.append(perm)
            
            # Check if Theta_l is a subset of {row} × {1,...,n}
            # If so, we need the marginal contribution
            if len(X_Theta_l_row) == L_diffset:
                mar_result = evaluate_density_mar(x_sorted, perm, tau_0, tau_1, sigma)
                marginal_contr = jnp.asarray(mar_result, dtype=jnp.float64)
        

        # Evaluate dependence integral
        dep_contr = evaluate_density_dep_b0_dep(x_sorted_list, perm_list, a0, b0, wa, wb, tau_0, tau_1, sigma,make_deriv_matrix)
        dep_contr = jnp.asarray(dep_contr, dtype=jnp.float64)
        
        # Total density is sum of marginal and dependence contributions
        density = marginal_contr + dep_contr
    
    else: #Only row row_index contains finite values, the rest is infinite
            
        # Collect x_{i,j} for which (i,j) in Theta_l for this row
        X_Theta_l_row = jnp.array([data[i][j] for i, j in Theta_l if i == row_index])
        location_values = locations[row_index,:]
        X_Theta_l_row_combined = jnp.concatenate([X_Theta_l_row, location_values])
        
        # Collect x_{i,j} for which (i,j) in Theta_l^c (complement) for this row
        diffset_comp = Theta.get_complement(l)
        X_Theta_l_row_comp = [data[i][j] for i, j in [tuples for tuples_list in diffset_comp for tuples in tuples_list] if i == row_index]
            
        # Convert to JAX arrays and sort
        x1 = X_Theta_l_row_combined
        z = jnp.array(X_Theta_l_row_comp)
        x_sorted, perm = pack_sort_and_indices(x1, z)

        # Use vectorized operation for the row
        mar_result = evaluate_density_mar(x_sorted, perm, tau_0, tau_1, sigma)
        marginal_contr = jnp.asarray(mar_result, dtype=jnp.float64)

        density=marginal_contr

    
    return density







##################### Non-vectorized version for verification #####################


# def log_transition_density_MH_Z_l(locations_prop,locations_current,l,partition):
#     ''' Calculates the transition density from the current state to the proposed state in the simulation of Z_l. When Theta_l contains only one row index the return log-density if correct.
#     When Theta_l contains more than one row index the returned log-density is only correct up to a constant (np.log(1/2) should be 0). The factor then cancels out when the density ratio is calculated in the MH step.
#     The model we use is throwing a Bernoulli random variable to decide whether we have all finite values, called state F, 
#     and otherwise only the row corresponding to Theta_l is finite, called state I.
#     Given the state we propse new values based on i.i.d. lognormal distributions with scale parameter location_current (using the scipy.stats.lognorm function notation).
#     If the current value at a sight is infinite we choose scale=1 as a default, which corresponds to centering of the normal distribution at 0.'''
#     current_state=np.all(np.isfinite(locations_current))# True if we are currently in state F
#     prop_state=np.all(np.isfinite(locations_prop))# True if proposoal in state F
#     row_index=partition.get_element(l)[0][0] 
#     if current_state:
#         if prop_state:
#             array=copy.deepcopy(locations_current)
#             for i in range(array.shape[0]):
#                 for j in range(array.shape[1]):
#                     array[i,j]=lognorm.pdf(locations_prop[i,j],s=1,scale=locations_current[i,j])
#             dens=np.log(1/2) + np.sum(np.log(array))
#         else:###check that also proposal is infinite everywhere except in row row_index
#             array=copy.deepcopy(locations_current)
#             for j in range(array.shape[1]):
#                 array[row_index,j]=lognorm.pdf(locations_prop[row_index,j],s=1,scale=locations_current[row_index,j])
#             dens=np.log(1/2) + np.sum(np.log(array[row_index,:]))
#     else:
#         if prop_state:
#             array=copy.deepcopy(locations_current)
#             for i in range(array.shape[0]):
#                 if i==row_index:
#                     for j in range(array.shape[1]):
#                         array[i,j]=lognorm.pdf(locations_prop[i,j],s=1,scale=locations_current[i,j])
#                 else:
#                     array[i,:]=lognorm.pdf(locations_prop[i,:],s=1,scale=1)
#             dens=np.log(1/2) + np.sum(np.log(array))
#         else:###check that also proposal is infinite everywhere exect in row row_index
#             array=copy.deepcopy(locations_current)
#             for j in range(array.shape[1]):
#                 array[row_index,j]=lognorm.pdf(locations_prop[row_index,j],s=1,scale=locations_current[row_index,j])
#             dens=np.log(1/2) + np.sum(np.log(array[row_index,:]))
#     return dens


# def evaluate_density_dep_loop(x_sorted_list, perm_list, tau0, tau1, sigma, precision_a=32, precision_b=32, use_endpoint_correction=False):
#     """Evaluate density using cached JIT functions
#      returns the evalution of one density integral for given list of x_sorted, perm
#      x_sorted_list: a list of jnp arrays of sorted x values for margin 1<=i<=d
#      perm_list: a list of permutations which encode the indices w.r.t. which the derivative are taken in margins 1<=i<=d
#     """
    
#     # Ensure all parameters are float64
#     tau0 = jnp.asarray(tau0, dtype=jnp.float64)
#     tau1 = jnp.asarray(tau1, dtype=jnp.float64)
#     sigma = jnp.asarray(sigma, dtype=jnp.float64)
#     # Use Gauss Laguerre nodes x and weights w to integrate a function of the form f(a)=g(a) * a^(alpha) exp(-a), requires alpha > -1
#     #here f(a)=E[h(a)]l(a,b)=E[h(a)]a^(-1-sigma)exp(-a) /Gamma(1-sigma) 
#     #Therefore we need to choose g(a)=E[h(a)]/{a*Gamma(1-sigma)}, alpha=-sigma
#     a, w_a = gauss_laguerre_gamma_crm_jax(precision_a, -sigma, dtype=jnp.float64)
    
#     # The inner integral g(b) is a function of b and we will integrate g(b)q(b) w.r.t. b
#     # This is an integral of the form int_0^(X_max+tau0) g(b)q(b) db, where X_max is the largest value in x_sorted
#     upper = jnp.max(jnp.concatenate(x_sorted_list)) + tau0
#     upper = jnp.asarray(upper, dtype=jnp.float64)  # Ensure float64
#     lower = jnp.asarray(0.0, dtype=jnp.float64)
    
#     b, w_b = gauss_legendre_gamma_crm_jax(precision_b, lower, upper, dtype=jnp.float64)
    
#     # Rest of function with consistent float64 operations...
#     int_b = []
#     const_gamma = jax.scipy.special.gamma(1 - sigma).astype(jnp.float64)

#     for j in range(len(b)):
#         int_a = []
#         for k in range(len(a)):
#             jit_funcs_dep = get_jit_functions_dep(a[k], b[j], tau0, tau1, sigma)
#             f_dep_jit = jit_funcs_dep['f_dep_jit']
#             dep_derivative = jnp.asarray(1.0, dtype=jnp.float64)
            
#             for x_sorted, perm in zip(x_sorted_list, perm_list):
#                 result = mixed_each_coord_value_jvp(f_dep_jit, x_sorted, coord=perm)
#                 dep_derivative *= jnp.asarray(result, dtype=jnp.float64)
            
#             ga = dep_derivative / (a[k] * const_gamma)
#             int_a.append(w_a[k] * ga)
        
#         int_a_sum = jnp.sum(jnp.array(int_a, dtype=jnp.float64))
#         int_b.append(w_b[j] * int_a_sum)
    
#     I = jnp.sum(jnp.array(int_b, dtype=jnp.float64))
#     return I   

# def prob_new_partition_loop(current_partition, new_partition, data, tau0, tau1,
#                       sigma, precision_a=32, precision_b=32):
#     """
#     Calculate the probability ratio between new and current partition by evaluating
#     Laplace transform derivatives at the difference sets.
    
#     Parameters:
#     -----------
#     current_partition, new_partition : partition objects
#     data : list of lists
#     tau0, tau1 : float - kernel parameters
#     l_fct : function - evaluates l(a,b)=rho(a)q(b)
#     alpha_0 : function - evaluates alpha_0([x,y])
#     precision_a, precision_b : int - precision parameters
#     sigma, theta : float - Gamma Lévy process parameters
    
#     Returns:
#     --------
#     float : probability ratio (up to proportionality constant)
#     """
    
#     # Calculate the 4 difference sets between current and new partition
#     idx_enum, enum_par=new_partition.partition_intersect_diff(current_partition)
#     idx_denom, denom_par =current_partition.partition_intersect_diff(new_partition)
    
#     L_enum=current_partition.get_length()
#     L_denom=new_partition.get_length()
#     d=len(data)
#     enumerator = jnp.asarray(1.0, dtype=jnp.float64)
#     denominator = jnp.asarray(1.0, dtype=jnp.float64)

#     #### Calculate the enumerator #### 
#     for idx in range(len(idx_enum)):
#         x_sorted_list = []
#         perm_list = []
#         mar_cont = jnp.asarray(0.0, dtype=jnp.float64)  # JAX scalar
#         for row in range(d):
#             diffset=new_partition.get_element(idx_enum[idx])
#             L_diffset=len(diffset)
#             X_Theta_l_row = [data[i][j] for i, j in diffset if i==row]
#             diffset_comp=new_partition.get_complement(idx_enum[idx])
#             X_Theta_l_row_comp=[data[i][j] for i, j in [tuples for tuples_list in diffset_comp for tuples in tuples_list] if i==row]
#             x1=jnp.array(X_Theta_l_row)
#             z=jnp.array(X_Theta_l_row_comp)
#             x_sorted, perm=pack_sort_and_indices(x1,z)   
#             x_sorted_list.append(x_sorted)
#             perm_list.append(perm)
#             # if diffset subset i x n then we need the marginal contribution, otherwise we dont
#             if len(X_Theta_l_row) == L_diffset:
#                 # Keep as JAX array operation
#                 mar_result = evaluate_density_mar(x_sorted, perm, tau0, tau1, sigma)
#                 mar_cont = jnp.asarray(mar_result, dtype=jnp.float64)
        
#         dep_result = evaluate_density_dep_loop(x_sorted_list, perm_list, tau0, tau1, sigma, precision_a, precision_b)
#         dep_contr = jnp.asarray(dep_result, dtype=jnp.float64)
        
#         # JAX array operations
#         enumerator *= (dep_contr + mar_cont)
    
#     #### Calculate the denominator #### 
#     for idx in range(len(idx_denom)):
#         x_sorted_list=[]
#         perm_list=[]
#         mar_cont = jnp.asarray(0.0, dtype=jnp.float64) 
#         for row in range(d): ### Check that range is correct here
#             diffset=current_partition.get_element(idx_denom[idx])
#             L_diffset=len(diffset)
#             # collect x_{i,j} for which (i,j) in theta_l
#             x1= jnp.array([data[i][j] for i, j in diffset if i==row])
#             diffset_comp=current_partition.get_complement(idx_denom[idx])
#             # collect x_{i,j} for which (i,j) in theta_l^c
#             z=jnp.array([data[i][j] for i, j in [tuples for tuples_list in diffset_comp for tuples in tuples_list] if i==row])
#             x_sorted, perm=pack_sort_and_indices(x1,z)   
#             x_sorted_list.append(x_sorted)
#             perm_list.append(perm)
#             # if theta_l is a subset if (row x n) then we need the marginal contribution, otherwise we dont
#             if len(x1)==L_diffset:# theta_l is subset of (row x n)
#                 mar_result = evaluate_density_mar(x_sorted, perm, tau0, tau1, sigma)
#                 mar_cont = jnp.asarray(mar_result, dtype=jnp.float64)
#         dep_result = evaluate_density_dep_loop(x_sorted_list, perm_list, tau0, tau1, sigma, precision_a, precision_b)
#         dep_contr = jnp.asarray(dep_result, dtype=jnp.float64)
#         denominator *= (dep_contr + mar_cont)
              

#     return enumerator/denominator



