######## this script contains the computation of the laplace transform and its derivatives ##########
from jax import config
from time import time
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax.tree_util import Partial

############### (log) Laplace transforms ################



# Vectorized version of dependence integral for grid of (a0,b0) values
def laplace_dep_value(x_sorted, a0, b0, tau0, tau1, sigma):
    x = jnp.asarray(x_sorted, jnp.float64)
    a0 = jnp.asarray(a0, jnp.float64)
    b0 = jnp.asarray(b0, jnp.float64)
    tau0 = jnp.asarray(tau0, jnp.float64)
    tau1 = jnp.asarray(tau1, jnp.float64)
    sig  = jnp.asarray(sigma, jnp.float64)

    tiny = jnp.finfo(x.dtype).tiny
    coef = -1.0 / (sig * (sig + 1.0))

    bounds = jnp.concatenate([jnp.zeros((1,), x.dtype), x])
    L = bounds[:-1]
    R = bounds[1:]

    suffix   = jnp.cumsum(x[::-1])[::-1]
    m_counts = jnp.arange(x.shape[0], 0, -1, dtype=x.dtype)
    C = tau1 * suffix + 1.0
    D = tau1 * m_counts

    # clamp around b0±tau0
    L = jnp.maximum(L, b0 - tau0)
    U = jnp.minimum(R, b0 + tau0)
    valid = U > L

    baseU = jnp.maximum(C - D * U, tiny)
    baseL = jnp.maximum(C - D * L, tiny)

    def Gpos(base, b):
        return coef * base**(sig + 1.0) / D - b / sig

    contrib = jnp.where(valid, Gpos(baseU, U) - Gpos(baseL, L), 0.0)
    total = jnp.sum(contrib)
    return jnp.exp(-a0 * total / (2.0 * tau0))

#constructor for vectorized evaluation over grid of (a0,b0) values
def make_F_grid(a0s, b0s, tau0, tau1, sigma):
    """ Vectorized evaluation of laplace_trafo_dep over grid of (a0,b0) values """
    a0s = jnp.asarray(a0s, jnp.float64)   # (Na,)
    b0s = jnp.asarray(b0s, jnp.float64)   # (Nb,)

    def F(x_sorted):  # returns (Nb, Na)
        # Create all combinations of (b0, a0)
        def single_laplace(b0, a0):
            return laplace_dep_value(x_sorted, a0, b0, tau0, tau1, sigma)
        
        # Vectorize over a0 first (inner dimension)
        vec_over_a0 = jax.vmap(single_laplace, in_axes=(None, 0))
        
        # Then vectorize over b0 (outer dimension) 
        vec_over_both = jax.vmap(vec_over_a0, in_axes=(0, None))
        
        return vec_over_both(b0s, a0s)  # shape (Nb, Na)

    return jax.jit(F)

#constructor for vectorized evaluation over grid of (a0,b0) values
def make_F_grid_b0_dep(a0s, tau0, tau1, sigma):
    """ Vectorized evaluation of laplace_trafo_dep over grid of a0 values """
    a0s = jnp.asarray(a0s, jnp.float64)   # (Na,)

    def F(x_sorted,b0s):  # returns (Nb, Na)
        b0s = jnp.asarray(b0s, jnp.float64)   # (Nb,)
        # Create all combinations of (b0, a0)
        def single_laplace(b0, a0):
            return laplace_dep_value(x_sorted, a0, b0, tau0, tau1, sigma)
        
        # Vectorize over a0 first (inner dimension)
        vec_over_a0 = jax.vmap(single_laplace, in_axes=(None, 0))
        
        # Then vectorize over b0 (outer dimension) 
        vec_over_both = jax.vmap(vec_over_a0, in_axes=(0, None))
        
        return vec_over_both(b0s, a0s)  # shape (Nb, Na)

    return jax.jit(F)


# a 1_{x\geq b} exP(-\int_0^x a 1_{s\geq b} ds) =a exp(-a(x-b) ds)   der: -a^2 exp(-a(x-b)). Thus derivative is permutation invariant in the arguemnts

#jitted version of marginal integral
def log_laplace_trafo_margin_sorted_jit(tau0, tau1, sigma, dtype=jnp.float64):
    """
    Returns f(x_sorted) that computes
      I = - ∫₀^∞ [ ((τ₁ Σ_j (x_j - b)_+ + 1)^σ - 1)/σ ] * (2τ₀)⁻¹ * min{b+τ₀, 2τ₀} db
    Assumes: τ₀>0, τ₁>0, σ∈(0,1), and x_sorted is ascending.
    """
    tau0 = jnp.asarray(tau0, dtype)
    tau1 = jnp.asarray(tau1, dtype)
    sig  = jnp.asarray(sigma, dtype)
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype)

    @jax.jit
    def f(x_sorted):
        x = jnp.asarray(x_sorted, dtype)
        m = x.shape[0]
        bounds = jnp.concatenate([jnp.zeros((1,), dtype), x])
        L = bounds[:-1]     # [m]
        R = bounds[1:]      # [m]

        # Active count/sum on [x_{i-1}, x_i]: k = m-i, A = sum_{j>=i} x_j
        k = jnp.arange(m, 0, -1, dtype=dtype)         # m, m-1, ..., 1
        A = jnp.cumsum(x[::-1])[::-1]                 # suffix sums
        c1 = jnp.maximum(tau1 * k, tiny)
        c0 = tau1 * A + 1.0

        # Split each segment by weight regions using clamping (no sort/searchsorted):
        L_lin, R_lin   = L, jnp.minimum(R, tau0)
        valid_lin      = R_lin > L_lin
        L_unit, R_unit = jnp.maximum(L, tau0), R
        valid_unit     = R_unit > L_unit

        # ---- primitives (FIXED F_unit SIGN) ----
        def F_unit(b, c0, c1):
            # integrates  - (1/(2τ₀σ)) * ( (c0 - c1 b)^σ - 1 ) 
            base = jnp.maximum(c0 - c1 * b, tiny)
            term1 = base**(sig + 1.0) / (c1 * (sig + 1.0))
            term2 = b
            return (term1 + term2) / (2.0 * tau0 * sig)

        def F_lin(b, c0, c1):
            # integrates  - (1/(2τ₀σ)) * (b+τ₀) * ( (c0 - c1 b)^σ - 1 )
            base = jnp.maximum(c0 - c1 * b, tiny)
            t1 = -(c0 * base**(sig + 1.0) / (sig + 1.0) - base**(sig + 2.0) / (sig + 2.0)) / (c1**2)
            t2 = -b**2 / 2.0
            t3 = tau0 * (-base**(sig + 1.0) / (c1 * (sig + 1.0)) - b)
            return -(t1 + t2 + t3) / (2.0 * tau0 * sig)

        lin_contrib  = jnp.where(valid_lin,  F_lin(R_lin,  c0, c1) - F_lin(L_lin,  c0, c1), 0.0)
        unit_contrib = jnp.where(valid_unit, F_unit(R_unit, c0, c1) - F_unit(L_unit, c0, c1), 0.0)
        total = jnp.sum(lin_contrib + unit_contrib)

        # Optional safety check: integral must be ≤ 0 (numerically allow tiny eps)
        # total = jnp.where(total > 1e-12, jnp.nan, total)

        return total

    return f

######################################################


#################### Differentiation ####################
###### works much faster for large d, go-to method for k<10 derivatives
def mixed_each_coord_value_jvp(f, x, coord):
    """
    Compute ∂_0 ∂_1 ... ∂_{d-1} f(x) for scalar f: R^d->R,
    using a pure forward-mode JVP chain. Returns a 0-d JAX array (a number).
    """
    x = jnp.asarray(x)
    d = x.shape[0]
    J = coord.shape[0]
    E = jnp.eye(d, dtype=x.dtype)  # basis vectors

    def dir_derivative(prev_g, e):
        # returns h(y) = directional derivative of prev_g at y along e
        def h(y):
            _, tang = jax.jvp(prev_g, (y,), (e,))
            return tang  # scalar
        return h

    g = f
    for i in coord: # if coord is empty then we return f(x) as intended
        g = dir_derivative(g, E[i])  # g stays a unary scalar function

    out = ((-1.0)**J)*g(x)              # <-- VALUE (0-d array), not a function
    return out


def mixed_each_coord_value_jvp_wrt_x(f, x, coord, *args):
    """
    Compute the mixed partial ∂_{coord[0]} ... ∂_{coord[J-1]} f(x, *args)
    where differentiation is ONLY w.r.t. x. Additional args (e.g. b0s)
    are treated as constants (tangent = 0).

    f : callable (x, *args) -> array_like
    x : (d,) array
    coord : (J,) integer array (indices in [0..d-1])
    *args : extra positional args to f (e.g., b0s)

    Returns:
        array with the same shape as f(x, *args)
    """
    x = jnp.asarray(x)
    d = x.shape[0]
    J = jnp.asarray(coord).shape[0]
    E = jnp.eye(d, dtype=x.dtype)  # basis vectors for directions

    # Zero tangents for all non-differentiated args
    zeros_args = jax.tree_util.tree_map(jnp.zeros_like, args)

    def dir_derivative(prev_g, e):
        # returns h(y, *a) = directional derivative of prev_g wrt y along e
        def h(y, *a):
            _, tang = jax.jvp(prev_g,
                              (y, *a),
                              (e, *zeros_args))
            return tang
        return h

    g = f  # g will stay a unary function in y, but keeps *args as extra params
    for i in coord:   # if coord is empty we just return f(x, *args)
        g = dir_derivative(g, E[i])

    out = ((-1.0)**J) * g(x, *args)
    return out


####################################################
####### Helpers ####################################

def pack_sort_and_indices(z, x3):
    x = jnp.concatenate([z, x3])
    perm = jax.lax.stop_gradient(jnp.argsort(x))
    inv  = jnp.empty_like(perm).at[perm].set(jnp.arange(perm.size))
    pos_of_z = inv[: z.shape[0]]
    x_sorted = x[perm]
    return x_sorted, pos_of_z


def gauss_laguerre_gamma_crm_jax(n: int, alpha: float, dtype=jnp.float64):
    # nodes/weights for weight x^alpha * exp(-x), requires alpha > -1
    #l(a,b)=rho(a)q(b)=a^(-1-sigma)exp(-a) /Gamma(1-sigma) 
    # we need to choose f(a)=E[h(a)]/{a*Gamma(1-sigma)}, alpha=-sigma
    k = jnp.arange(1, n, dtype=dtype)
    diag = (2 * jnp.arange(1, n + 1, dtype=dtype) - 1 + alpha)
    off  = jnp.sqrt(k * (k + alpha))
    J = jnp.diag(diag) + jnp.diag(off, 1) + jnp.diag(off, -1)
    evals, evecs = jnp.linalg.eigh(J)
    x = evals
    v0 = evecs[0, :]
    w = jax.scipy.special.gamma(alpha + 1.0).astype(dtype) * (v0 ** 2)
    return x, w ### they should only be computed once and then reused


def gauss_legendre_gamma_crm_jax(n: int, L, U, dtype=jnp.float64):
    """
    n-point Gauss–Legendre nodes/weights on [L, U], weight = 1.
    Note that the Gauss-Legendre weights on [0,M] are just the M-times the nodes and weight for [0,1]
    Returns:
        t: shape (n,) nodes in [-1, 1]
        w: shape (n,) weights (sum to 2)
    """
    k = jnp.arange(1, n, dtype=dtype)
    off = k / jnp.sqrt(4 * k * k - 1)         # off-diagonals
    J = jnp.diag(off, 1) + jnp.diag(off, -1)  # diagonal is zero
    evals, evecs = jnp.linalg.eigh(J)
    t = evals                                  # nodes
    w = 2.0 * (evecs[0, :] ** 2)
    b = (U + L)/2 + (U - L)/2 * t              # mapped nodes
    wb = (U - L)/2 * w                         # mapped weights
    return b, wb      ### they should only be computed once and then reused


#####################################################
#####################################################




# ####### Test timing of padded version ######


# key = jax.random.PRNGKey(0)
# tau0=2
# b0=1
# tau1=0.8
# a0=1.2
# sigma=2/3
# n01, n2, n3 = 8, 1, 10



# # 1. Pre-generate all test data to avoid recompilation
# keys = jax.random.split(jax.random.PRNGKey(0), 500)
# test_data = []
# z_data = []
# x_3_data = []
# test_data_padded = []
# total_size = n01 + n2 + n3
# for i in range(100):
#     # z = jax.random.uniform(keys[i], (n1+n2,), minval=1.0, maxval=1.5)
#     # x3 = jax.random.uniform(keys[i], (n3,), minval=1.0, maxval=1.5)
#     # x_sorted, idxs = pack_sort_and_indices(z, x3)
#     N = int(jax.random.randint(keys[i], (), 1, n01+n2))
#     z = jax.random.uniform(keys[i], (N,), minval=1.0, maxval=1.5)
#     z_data.append(z)
#     x3_part = jax.random.uniform(keys[i], (n3,), minval=1.0, maxval=1.5) 
#     x_3_data.append(x3_part)
#     #x3 = jnp.pad(x3_part, (0, total_size - N - n3), constant_values=0.0)  # pad x3 with zeros to max size    x_padded = jnp.pad(x_combined, (0, total_size - x_combined.shape[0]), constant_values=0.0)
#     x_sorted, idxs = pack_sort_and_indices(z, x3_part)
#     test_data.append((x_sorted,idxs))

# for i in range(100):
#     x_sorted, idxs = pack_sort_and_indices_and_pad(z_data[i],x_3_data[i],max_size=n01+n2+n3)
#     test_data_padded.append((x_sorted,idxs))


# # Pre-generate array to store derivatives
# derivatives_var_length = jnp.zeros(100)

# derivatives_var_length_padded = jnp.zeros(100)


# # Build jitted base f once
# f_jit = laplace_trafo_dep_sorted_jit(a0, b0, tau0, tau1, sigma, dtype=jnp.float64)


# start = time()
# # 2. Warm up the function with the exact shape you'll use
# sample_x = test_data[0][0]
# idxs = test_data[0][1] 
# _ = mixed_each_coord_value_jvp(f_jit, sample_x,coord=idxs)
# jax.block_until_ready(_)
# stop = time()
# print(f"Compilation time: {(stop-start)*1e3:.2f} ms")



# # 3. Time only the computation, not the compilation
# start = time()
# for i, (x_sorted, idxs) in enumerate(test_data):
#     val = mixed_each_coord_value_jvp(f_jit, x_sorted, coord=idxs)
#     jax.block_until_ready(val)
#     derivatives_var_length = derivatives_var_length.at[i].set(val)  # JAX immutable array update
# stop = time()
# print(f"Average time per variable-order derivative, non-padded, without sampling: {(stop-start)*1e3/100:.2f} ms")



# start = time()
# # 2. Warm up the function with the exact shape you'll use
# sample_x = test_data_padded[0][0]
# idxs = test_data_padded[0][1] 
# _ = mixed_each_coord_value_jvp_padded(f_jit, sample_x,coord=idxs,max_size=n01+n2+n3)
# jax.block_until_ready(_)
# stop = time()
# print(f"Compilation time: {(stop-start)*1e3:.2f} ms")


# # 3. Time only the computation, not the compilation
# start = time()
# for i, (x_sorted, idxs) in enumerate(test_data_padded):
#     val = mixed_each_coord_value_jvp_padded(f_jit, x_sorted, coord=idxs,max_size=n01+n2+n3)
#     jax.block_until_ready(val)
#     derivatives_var_length_padded = derivatives_var_length_padded.at[i].set(val)  # JAX immutable array update
# stop = time()
# print(f"Average time per variable-order derivative, padded, without sampling: {(stop-start)*1e3/100:.2f} ms")

# print("Derivatives:")
# print(derivatives_var_length)
# print("Derivatives padded:")
# print(derivatives_var_length_padded)   

# print("Difference:")
# print(derivatives_var_length_padded - derivatives_var_length)  # should be zero



# # Pre-generate array to store derivatives
# derivatives_var_length = jnp.zeros(100)

# a0_array = jnp.array([a0])  # Shape (1,)
# b0_array = jnp.array([b0])  # Shape (1,)

# # Now F_grid will return shape (1, 1) instead of (32, 32)
# F_grid_single = make_F_grid(a0_array, b0_array, tau0, tau1, sigma)

# start = time()
# x_sorted = test_data[0][0]
# coord = test_data[0][1] 
# result = mixed_each_coord_value_jvp(F_grid_single, x_sorted, coord)  # Shape (1, 1)
# jax.block_until_ready(result)
# stop = time()
# print(f"Compilation time for single (a0,b0): {(stop-start)*1e3:.2f} ms")

# # 3. Time only the computation, not the compilation
# start = time()
# for i, (x_sorted, idxs) in enumerate(test_data):
#     val = mixed_each_coord_value_jvp(F_grid_single, x_sorted, idxs)
#     jax.block_until_ready(val)
#     derivatives_var_length = derivatives_var_length.at[i].set(val[0,0])  
# stop = time()
# print(f"Average time per variable-order derivative, non-padded, without sampling: {(stop-start)*1e3/100:.2f} ms")


# print("Difference:")
# print(jnp.sum(derivatives_var_length_padded - derivatives_var_length))  # should be zero




# ################ Compare with 2nd order derivative in closed form
# # fixed hypers
# key = jax.random.PRNGKey(0)
# tau0=1
# b0=1
# tau1=0.8
# a0=1.2
# sigma=2/3
# n1, n2, n3 = 6, 1, 10


# f_val = laplace_trafo_dep_sorted_jit(a0, b0, tau0, tau1, sigma, dtype=jnp.float64)

# x_sorted, idxs= pack_sort_and_indices_and_pad(jnp.asarray([0.5,2.0]),jnp.asarray([],dtype=jnp.float64),max_size=n1+n2+n3)
# der=mixed_each_coord_value_jvp_padded(f_val, x_sorted,idxs,max_size=n1+n2+n3)
# der_v2=mixed_each_coord_value_jvp(f_val, jnp.asarray([0.5,2.0]),jnp.asarray([0 ,1],dtype=jnp.int32))
# from Old_Code.laplace_trafo import evaluate_laplace_trafo_jax 
# print(f_val(x_sorted))
# print(evaluate_laplace_trafo_jax(x_sorted, a0=a0, tau0=tau0, tau1=tau1, sigma=sigma, b0=b0))
# ##manual check
# def f_1(x,tau0,tau1,a0,b0,sigma):
#     # fac=a0/(2*tau0)
#     # sum= -(tau1*(x[0]+x[1])+1)**(sigma+1) -(tau1*(x[1]-x[0])+1)**(sigma+1) +2*(tau1*(x[1]-(b0+tau0))+1)**(sigma+1)
#     # sum1=(b0+tau0)/sigma
#     # div=(2*tau1*(sigma+1)*sigma)
#     # return math.exp (fac*(sum/div+sum1)
#     T1= f_val(x_sorted)                 
#     A= tau1*(x[0]+x[1])+1
#     B= tau1*(x[1]-x[0])+1
#     fac= -a0/(2*tau0)
#     return T1*fac* ( ( (A**sigma-B**sigma)*(fac)*(A**sigma+B**sigma-2) )/(4*sigma**2) + tau1*(A**(sigma-1) - B**(sigma-1))/2 )
# x=[0.5,2.0]
# t1=f_1(x,tau0,tau1,a0,b0,sigma) 
# t2=der
# print(t1)
# print(t2)
# print(t1-t2)
# #### shows that second order der is correct




# x=jnp.asarray([2.0,0.5],dtype=jnp.float64)
# x_sorted, coord = pack_sort_and_indices_and_pad( x,jnp.asarray([],dtype=jnp.float64),max_size=10)


# der=mixed_each_coord_value_jvp_padded(f_val, x_sorted,coord)



# x_sorted_2, coord_2 = pack_sort_and_indices_and_pad( x,jnp.asarray([0.0,0.0],dtype=jnp.float64),max_size=10)

# der_0_added=mixed_each_coord_value_jvp_padded(f_val, x_sorted_2,coord_2)

# from Old_Code.laplace_trafo import evaluate_laplace_trafo_jax 
# print(f_val(x_sorted))
# print(evaluate_laplace_trafo_jax(x_sorted, a0=a0, tau0=tau0, tau1=tau1, sigma=sigma, b0=b0))
# print(evaluate_laplace_trafo_jax(x_sorted_2, a0=a0, tau0=tau0, tau1=tau1, sigma=sigma, b0=b0))
# ##manual check
# def f_1(x,tau0,tau1,a0,b0,sigma):
#     # fac=a0/(2*tau0)
#     # sum= -(tau1*(x[0]+x[1])+1)**(sigma+1) -(tau1*(x[1]-x[0])+1)**(sigma+1) +2*(tau1*(x[1]-(b0+tau0))+1)**(sigma+1)
#     # sum1=(b0+tau0)/sigma
#     # div=(2*tau1*(sigma+1)*sigma)
#     # return math.exp (fac*(sum/div+sum1)
#     T1= f_val(x_sorted)                 
#     A= tau1*(x[0]+x[1])+1
#     B= tau1*(x[1]-x[0])+1
#     fac= -a0/(2*tau0)
#     return T1*fac* ( ( (A**sigma-B**sigma)*(fac)*(A**sigma+B**sigma-2) )/(4*sigma**2) + tau1*(A**(sigma-1) - B**(sigma-1))/2 )
# x=[0.5,2.0]
# t1=f_1(x,tau0,tau1,a0,b0,sigma) 
# t2=der
# t3=der_0_added
# print(t1)
# print(t2)
# print(t3)
# print(t1-t2)
# print(t2-t3)
# #shows that adding zeros does not change the derivative



