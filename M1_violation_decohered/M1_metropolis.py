from M1_tensor_contraction import *
from M1_prob import *
from jax.random import PRNGKey, split, uniform, randint, bernoulli
from jax import vmap

def metropolis_bulk_step(x, y, p, num_samples, key, strs_ini, strs_column):
    """
    Perform num_samples Metropolis updates on plaq_dict.

    Args:
      x, y           : system size
      p             : model parameter
      batch         : batch_size
      num_samples   : how many proposed flips to attempt
    """
    w_list = []
    key, sub = split(key)
    config_init = bernoulli(sub, p, (x, y)).astype(jnp.int32)
    # 2) build the two “templates” for m=0 or m=1
    T0 = full_tensor(0, p)  # shape (2,2,2,2)
    T1 = full_tensor(1, p)  # shape (2,2,2,2)
    # 3) stack them along a new “m” axis:
    templates = jnp.stack([T0, T1], axis=0)
    # 4) index into that template with your random mask:
    tensor_init = templates[config_init]
    # → shape (x, y, 2,2,2,2)

    Pr_ini = Pr_bulk_func(x, y, p, tensor_init, strs_ini, strs_column)
    w_list.append(jnp.log(Pr_ini))

    def scan_step(carry, _):
        config, tensor, key, Pr0 = carry

        # split RNG for independent draws
        key, k1, k2, k3 = split(key, 4)

        # 1) propose a random flip at (x_,y_)
        x_ = randint(k1, (), 0, x)
        y_ = randint(k2, (), 0, y-1)

        # flip that bit in the config and tensor
        m_new    = 1 - config[x_, y_]
        config_p = config.at[x_, y_].set(m_new)
        tensor_p = tensor.at[x_, y_].set(full_tensor(m_new, p))

        # 2) evaluate the new probability
        Pr = Pr_bulk_func(x, y, p, tensor_p, strs_ini, strs_column)

        # 3) accept/reject
        accept = uniform(k3, ()) < (Pr / Pr0)
        config = jnp.where(accept, config_p, config)
        tensor = jnp.where(accept, tensor_p, tensor)
        Pr_ = jnp.where(accept, Pr, Pr0)
        # emit the same Pr0 every step (or you could emit jnp.log(Pr) if you prefer)
        return (config, tensor, key, Pr_), jnp.log(Pr0)

    # dummy xs with length=num_samples
    xs = jnp.arange(num_samples)

    # initial carry
    init_carry = (config_init, tensor_init, key, Pr_ini)

    # scan over num_samples steps
    (config_final, tensor_final, key_final, Pr_final), w_list = lax.scan(
        scan_step,
        init_carry,
        xs
    )
    return w_list, key
 # str_ini, str_col, str_shrink, str_AB_col_ini, str_AB_col, str_AB_col_end, str_expand
def metropolis_AB_step(n,
                       p,
                       num_samples,
                       key,
                       str_ini,
                       str_col,
                       str_shrink,
                       str_AB_col_ini,
                       str_AB_col,
                       str_AB_col_end,
                       str_expand):
    # total number of “main” plaquettes
    num_plaq = 15*n**2 - 5*n - n*(n+2) + 2
    key, sub = split(key)
    config_init = bernoulli(sub, p, (num_plaq+1, )).astype(jnp.int32)

    # 2) build tensor_init by indexing full_tensor(0,p) vs full_tensor(1,p)
    T0 = full_tensor(0, p)
    T1 = full_tensor(1, p)
    main_templates = jnp.stack([T0, T1], axis=0)  # shape (2, 2,2,2,2)
    tensor_init = main_templates[config_init[:-1]]  # shape (num_plaq, 2,2,2,2)

    # 3) build t_middle_init so only its [0] slice is random
    IE0 = inner_edge_tensor(0, p)
    IE1 = inner_edge_tensor(1, p)
    mid_templates = jnp.stack([IE0, IE1], axis=0)  # shape (2, 2,2,2)

    # pick the first slice
    first = mid_templates[config_init[-1]]  # shape (2,2,2)
    L = 3 * (n - 2)
    # the other L-1 slices stay at m=0
    rest = jnp.tile(IE0, (L - 1, 1, 1, 1))  # shape (L-1, 2,2,2)
    t_middle_init = jnp.concatenate([first[None], rest], axis=0)
    t_corner     = jnp.tile(inner_edge_corner_tensor(0, p), (4,1,1,1,1))

    Pr0 = Pr_AB_func(n, p,
                     tensor_init,
                     t_middle_init,
                     t_corner,
                     str_ini, str_col,
                     str_shrink,
                     str_AB_col_ini,
                     str_AB_col,
                     str_AB_col_end,
                     str_expand)

    # 2) one scan‐step of Metropolis
    def scan_step(carry, _):
        config, tensor, t_middle, Pr_prev, key = carry

        # split RNG
        key, k1, k2, k3 = split(key, 4)
        # pick a plaquette index in [0, num_plaq]
        x = randint(k1, (), 0, num_plaq+1)

        # branch A: flip main tensor[x]
        def flip_main(_):
            m = 1 - config[x]
            return (
                config.at[x].set(m),
                tensor.at[x].set(full_tensor(m, p)),
                t_middle
            )
        # branch B: flip t_middle[0]
        def flip_middle(_):
            m = 1 - config[x]
            return (
                config.at[x].set(m),
                tensor,
                t_middle.at[0].set(inner_edge_tensor(m, p))
            )

        config_p, tensor_p, middle_p = lax.cond(
            x < num_plaq,
            flip_main,
            flip_middle,
            operand=None
        )

        # recompute the proposed probability
        Pr_prop = Pr_AB_func(n, p,
                             tensor_p,
                             middle_p,
                             t_corner,
                             str_ini, str_col,
                             str_shrink,
                             str_AB_col_ini,
                             str_AB_col,
                             str_AB_col_end,
                             str_expand)

        # Metropolis accept/reject
        accept = uniform(k2, ()) < (Pr_prop / Pr_prev)
        config_new  = jnp.where(accept, config_p,  config)
        tensor_new  = jnp.where(accept, tensor_p,  tensor)
        middle_new  = jnp.where(accept, middle_p,  t_middle)
        Pr_new      = jnp.where(accept, Pr_prop, Pr_prev)

        # emit log-prob of the *old* state (or you could use Pr_new)
        return (config_new, tensor_new, middle_new, Pr_new, key), jnp.log(Pr_new)

    # dummy xs to run exactly num_samples steps
    xs = jnp.arange(num_samples)

    # run the scan
    init_carry = (config_init, tensor_init, t_middle_init, Pr0, key)
    (config_final,
     tensor_final,
     t_middle_final,
     Pr_final,
     key_final), w_list = lax.scan(scan_step, init_carry, xs)

    return w_list, key_final


def metropolis_B_step(n,
                      p,
                      num_samples,
                      key,
                      str_ini,
                      str_col,
                      str_shrink_B,
                      str_B_col_ini,
                      str_B_col,
                      str_B_col_end,
                      str_expand_B):
    # total number of plaquettes
    key, sub = split(key)
    num_plaq = 2*(2*n-1)*(n-1) + (n+2)*(n-1) + 2
    config_init = bernoulli(sub, p, (num_plaq, )).astype(jnp.int32)
    # 2) build the two “templates” for m=0 or m=1
    T0 = full_tensor(0, p)  # shape (2,2,2,2)
    T1 = full_tensor(1, p)  # shape (2,2,2,2)
    # 3) stack them along a new “m” axis:
    templates = jnp.stack([T0, T1], axis=0)
    # 4) index into that template with your random mask:
    tensor_init = templates[config_init]

    Pr0 = Pr_B_func(n, p,
                    tensor_init,
                    str_ini, str_col,
                    str_shrink_B,
                    str_B_col_ini, str_B_col,
                    str_B_col_end, str_expand_B)

    # 2) one Metropolis step
    def scan_step(carry, _):
        config, tensor, Pr_prev, key = carry

        # split RNG for index + accept
        key, k1, k2 = split(key, 3)
        x = randint(k1, (), 0, num_plaq)

        # propose a flip at x
        m_new     = 1 - config[x]
        config_p  = config.at[x].set(m_new)
        tensor_p  = tensor.at[x].set(full_tensor(m_new, p))

        # compute proposed weight
        Pr_prop = Pr_B_func(n, p,
                            tensor_p,
                            str_ini, str_col,
                            str_shrink_B,
                            str_B_col_ini, str_B_col,
                            str_B_col_end, str_expand_B)

        # accept/reject
        accept      = uniform(k2, ()) < (Pr_prop / Pr_prev)
        config_new  = jnp.where(accept, config_p,  config)
        tensor_new  = jnp.where(accept, tensor_p,  tensor)
        Pr_new      = jnp.where(accept, Pr_prop,  Pr_prev)

        # emit log Pr_prev
        return (config_new, tensor_new, Pr_new, key), jnp.log(Pr_prev)

    # dummy scan‐axis of length num_samples
    xs = jnp.arange(num_samples)

    # run the scan
    init_carry = (config_init, tensor_init, Pr0, key)
    (_, _, _, key_final), w_list = lax.scan(scan_step, init_carry, xs)

    return w_list, key_final


# 1) batch Metropolis‐bulk
batch_metro_bulk = jax.jit(
    vmap(
        metropolis_bulk_step,
        in_axes=(None,   # x
                 None,   # y
                 None,   # p
                 None,   # num_samples
                 0,      # key    ← we vectorize over the key
                 None,   # strs_ini
                 None),  # strs_column
        out_axes=(0, 0)
    ),
    static_argnums=(0,1,3,5,6)
)

# 2) batch Metropolis‐AB
batch_metro_AB = jax.jit(
    vmap(
        metropolis_AB_step,
        in_axes=(None,   # n
                 None,   # p
                 None,   # num_samples
                 0,      # key
                 None,   # str_ini
                 None,   # str_col
                 None,   # str_shrink
                 None,   # str_AB_col_ini
                 None,   # str_AB_col
                 None,   # str_AB_col_end
                 None),  # str_expand
        out_axes=(0, 0)
    ),
    static_argnums=(0,2,4,5,6,7,8,9,10)
)


# 3) batch Metropolis‐B
batch_metro_B = jax.jit(
    vmap(
        metropolis_B_step,
        in_axes=(None,   # n
                 None,   # p
                 None,   # num_samples
                 0,      # key
                 None,   # str_ini
                 None,   # str_col
                 None,   # str_shrink_B
                 None,   # str_B_col_ini
                 None,   # str_B_col
                 None,   # str_B_col_end
                 None),  # str_expand_B
        out_axes=(0, 0)
    ),
    static_argnums=(0, 2, 4, 5, 6, 7, 8, 9, 10)
)
