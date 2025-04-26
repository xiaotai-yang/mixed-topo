from M1_tensor_contraction import *
from jax.random import PRNGKey, split, uniform, randint

@partial(jax.jit, static_argnums=(0, 1, 4, 5))
def Pr_bulk_func(x, y, p, tensor_arr, str_ini, str_col):

    tensor = process_initial(p, tensor_arr[0] ,str_ini)
    for i in range(1, x):
        t_ini = jnp.einsum("abcd, d -> abc", tensor_arr[i][0], boundary_T_tensor(p))
        t_end = jnp.einsum("abcd, d -> abc", tensor_arr[i][-1], boundary_T_tensor(p))
        t_arr = (t_ini, ) + tuple(tensor_arr[i][1:-1]) + (t_end, )
        tensor = process_col(tensor, t_arr, str_col)
    revision_vec = n_tensor_product_shape_2(y, p)
    Pr = jnp.sum(tensor*revision_vec)
    return Pr

@partial(jax.jit, static_argnums=(0, 5, 6, 7, 8, 9, 10, 11))
def Pr_AB_func(n, p, tensor_, t_middle, t_corner, str_ini, str_col, str_shrink, str_AB_col_ini, str_AB_col, str_AB_col_end, str_expand):
    '''
    t_1: first (2n-1)*(3n-1) tensor
    t_col1: following (2n) tensor
    t_2: following n*(2n-1) tensor
    t_col2: following (2n) tensor
    t_3: following (2n-1)*(3n-1) tensor
    '''
    t_1 = tensor_[:(2*n-1)*(3*n-1)].reshape((2*n-1, 3*n-1, 2, 2, 2, 2))
    t_col1 = tensor_[(2*n-1)*(3*n-1):(2*n-1)*(3*n-1)+2*n]
    t_2 = tensor_[(2*n-1)*(3*n-1)+2*n:(2*n-1)*(3*n-1)+2*n+n*(2*n-1)].reshape((n, 2*n-1, 2, 2, 2, 2))
    t_col2 = tensor_[(2*n-1)*(3*n-1)+2*n+n*(2*n-1):(2*n-1)*(3*n-1)+2*n+n*(2*n-1)+2*n]
    t_3 = tensor_[(2*n-1)*(3*n-1)+2*n+n*(2*n-1)+2*n:].reshape((2*n-1, 3*n-1, 2, 2, 2, 2))
    #Initial column
    tensor = process_initial(p, t_1[0] ,str_ini)

    # 2n-2 column contraction
    for i in range(1, 2*n - 1):
        t_ini = jnp.einsum("abcd, d -> abc", t_1[i][0], boundary_T_tensor(p))
        t_end = jnp.einsum("abcd, d -> abc", t_1[i][-1], boundary_T_tensor(p))
        t_arr = (t_ini, ) + tuple(t_1[i][1:-1]) + (t_end, )
        tensor = process_col(tensor, t_arr, str_col)

    # shrinking column
    t_end = jnp.einsum("abcd, d -> abc", t_col1[-1], boundary_T_tensor(p))
    t_arr = tuple(t_middle[:n-2]) + (t_corner[0],) + tuple(t_col1[:-1]) + (t_end, )
    tensor = process_col(tensor, t_arr, str_shrink)

    t_end = jnp.einsum("abcd, d -> abc", t_2[0][-1], boundary_T_tensor(p))
    t_arr = (t_corner[1], ) + tuple(t_2[0][:-1]) + (t_end, )
    tensor = process_col(tensor, t_arr, str_AB_col_ini)

    # n column on AB only
    for i in range(1, n-1):
        t_end = jnp.einsum("abcd, d -> abc", t_2[i][-1], boundary_T_tensor(p))
        t_arr = (t_middle[n-3+i], ) + tuple(t_2[i][:-1]) + (t_end, )
        tensor = process_col(tensor, t_arr, str_AB_col)

    t_end = jnp.einsum("abcd, d -> abc", t_2[-1][-1], boundary_T_tensor(p))
    t_arr = (t_corner[2], ) + tuple(t_2[-1][:-1]) + (t_end, )
    tensor = process_col(tensor, t_arr, str_AB_col_end)

    # expanding column
    t_ini = jnp.einsum("abcd, d -> abc", t_col2[0], boundary_T_tensor(p))
    t_arr = (t_ini, ) + tuple(t_col2[1:]) + (t_corner[3], ) + tuple(t_middle[2*n-4:])
    tensor = process_col(tensor, t_arr, str_expand)

    # Final 2n-1 column
    for i in range(2*n-1):
        t_ini = jnp.einsum("abcd, d -> abc", t_3[i][0], boundary_T_tensor(p))
        t_end = jnp.einsum("abcd, d -> abc", t_3[i][-1], boundary_T_tensor(p))
        t_arr = (t_ini, ) + tuple(t_3[i][1:-1]) + (t_end, )
        tensor = process_col(tensor, t_arr, str_col)

    revision_vec_end = n_tensor_product_shape_2(3*n-1, p)
    Pr = jnp.sum(tensor*revision_vec_end)
    return Pr

@partial(jax.jit, static_argnums=(0, 3, 4, 5, 6, 7, 8, 9))
def Pr_B_func(n, p, tensor, str_ini, str_col, str_shrink_B, str_B_col_ini, str_B_col,  str_B_col_end, str_expand_B):
    t_1 = tensor[:(n-2)*(2*n-1)].reshape((n-2, 2*n-1, 2, 2, 2, 2))
    t_shrink = tensor[(n-2)*(2*n-1):(n-2)*(2*n-1)+2*n-1].reshape((2*n-1, 2, 2, 2, 2))
    t_B_ini = tensor[(n-2)*(2*n-1)+(2*n-1):(n-2)*(2*n-1)+(2*n-1)+n].reshape((n, 2, 2, 2, 2))
    t_2 = tensor[(n-2)*(2*n-1)+(2*n-1)+n:(n-2)*(2*n-1)+(2*n-1)+n+n*(n-1)].reshape((n, n-1, 2, 2, 2, 2))
    t_B_end = tensor[(n-2)*(2*n-1)+(2*n-1)+n+n*(n-1):(n-2)*(2*n-1)+(2*n-1)+n+n*(n-1)+n].reshape((n, 2, 2, 2, 2))
    t_expand = tensor[(n-2)*(2*n-1)+(2*n-1)+n+n*(n-1)+n:(n-2)*(2*n-1)+(2*n-1)+n+n*(n-1)+n+(2*n-1)].reshape((2*n-1, 2, 2, 2, 2))
    t_3 = tensor[(n-2)*(2*n-1)+(2*n-1)+n+n*(n-1)+n+(2*n-1):].reshape((n-2, 2*n-1, 2, 2, 2, 2))

    tensor_init = process_initial(p, t_1[0] ,str_ini)
    for i in range(1, n-2):
        t_ini = jnp.einsum("abcd, d -> abc", t_1[i][0], boundary_T_tensor(p))
        t_end = jnp.einsum("abcd, d -> abc", t_1[i][-1], boundary_T_tensor(p))
        t_arr = (t_ini, ) + tuple(t_1[i][1:-1]) + (t_end, )
        tensor = process_col(tensor_init, t_arr, str_col)

    # shrink B
    t_ini = jnp.einsum("abcd, c, d -> ab", t_shrink[0], boundary_T_tensor(p), boundary_T_tensor(p))
    t_shr = jnp.einsum("abcde, e -> abcd", t_shrink[1:n-1], boundary_T_tensor(p))
    t_end = jnp.einsum("abcd, d -> abc", t_shrink[-1], boundary_T_tensor(p))
    t_arr = (t_ini,) + tuple(t_shr) + tuple(t_shrink[n-1:2*n-2]) +(t_end,)
    tensor = process_col(tensor, t_arr, str_shrink_B)

    # t_B_ini
    t_ini = jnp.einsum("abcd, c, d -> ab", t_B_ini[0], boundary_T_tensor(p), boundary_T_tensor(p))
    t_end = jnp.einsum("abcd, d -> abc", t_B_ini[-1], boundary_T_tensor(p))
    t_arr = (t_ini,) + tuple(t_B_ini[1:n-1]) + (t_end,)
    tensor = process_col(tensor, t_arr, str_B_col_ini)


    #t_2
    for i in range(n):
        t_ini = jnp.einsum("abcd, d -> abc", t_2[i][0], boundary_T_tensor(p))
        t_end = jnp.einsum("abcd, d -> abc", t_2[i][-1], boundary_T_tensor(p))
        t_arr = (t_ini, ) + tuple(t_2[i][1:-1]) + (t_end, )
        tensor = process_col(tensor, t_arr, str_B_col)

    #t_2_end
    t_ini = jnp.einsum("abcd, d -> abc", t_B_end[0], boundary_T_tensor(p))
    t_end = jnp.einsum("abcd, c, d -> ab", t_B_end[-1], boundary_T_tensor(p), boundary_T_tensor(p))
    t_arr = (t_ini, ) + tuple(t_B_end[1:n-1]) + (t_end,)
    tensor = process_col(tensor, t_arr, str_B_col_end)

    #Expand B
    t_ini = jnp.einsum("abcd, d -> abc", t_expand[0], boundary_T_tensor(p))
    t_epd = jnp.einsum("abcde, e -> abcd", t_expand[n:2*n-2], boundary_T_tensor(p))
    t_end = jnp.einsum("abcd, c, d -> ab", t_expand[-1], boundary_T_tensor(p), boundary_T_tensor(p))
    t_arr = (t_ini,) + tuple(t_expand[1:n]) + tuple(t_epd)  + (t_end,)
    tensor = process_col(tensor, t_arr, str_expand_B)

    #t3
    for i in range(n-2):
        t_ini = jnp.einsum("abcd, d -> abc", t_3[i][0], boundary_T_tensor(p))
        t_end = jnp.einsum("abcd, d -> abc", t_3[i][-1], boundary_T_tensor(p))
        t_arr = (t_ini, ) + tuple(t_3[i][1:-1]) + (t_end, )
        tensor = process_col(tensor, t_arr, str_col)

    revision_vec_end = n_tensor_product_shape_2(2*n-1, p)
    Pr = jnp.sum(tensor*revision_vec_end)
    return Pr


