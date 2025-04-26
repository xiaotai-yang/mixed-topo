from tqdm.auto import tqdm
from util_M1 import *
from M1_tensor_contraction import *
from M1_prob import *
from M1_metropolis import *
from jax import vmap
import argparse

def main():
    parser = argparse.ArgumentParser(description="Compute domain-wall M1 CMI via batched Metropolis")
    parser.add_argument("--n",           type=int,   default=4,
                        help="Lattice parameter n")
    parser.add_argument("--batch-size",  type=int,   default=100,
                        help="Number of parallel chains")
    parser.add_argument("--num-samples", type=int,   default=10000,
                        help="Length of each Metropolis chain")
    parser.add_argument("--p-arr",       type=float, nargs="+",
                        default=[0.02, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14,0.15, 0.16, 0.18, 0.2, 0.3, 0.4],
                        help="List of p values to sweep over")
    parser.add_argument("--array-number", type=int,  default=5,
                        help="SLURM array task ID (for seeding & filenames)")
    args = parser.parse_args()

    n            = args.n
    batch_size   = args.batch_size
    num_samples  = args.num_samples
    p_arr        = jnp.array(args.p_arr)
    array_number = args.array_number

    # precompute all the shapes & string-tables once
    xc, yc       = n,     n-1
    xbc, ybc     = 3*n,   2*n-1
    xabc, yabc   = 5*n,   3*n-1

    strs_ini_c,    strs_column_c    = einsum_strs_initial(yc),    einsum_strs_column(yc)
    strs_ini_bc,   strs_column_bc   = einsum_strs_initial(ybc),   einsum_strs_column(ybc)
    strs_ini_abc,  strs_column_abc  = einsum_strs_initial(yabc),  einsum_strs_column(yabc)

    # AB-specific tables
    strs_col_ab_ini = einsum_strs_AB_column_ini(2*n)
    strs_col_ab     = einsum_strs_AB_column(2*n)
    strs_col_ab_end = einsum_strs_AB_column_end(2*n)
    str_shrink      = einsum_strs_shrink(n)
    str_expand      = einsum_strs_expand(n)

    # B-specific tables
    str_shrink_B    = einsum_shrink_B(n)
    str_B_col_ini   = einsum_B_col_ini(n)
    str_B_col       = einsum_strs_column(n-1)
    str_B_col_end   = einsum_B_col_end(n)
    str_expand_B    = einsum_expand_B(n)

    # storage for each p
    w_b   = []
    w_bc  = []
    w_ab  = []
    w_abc = []

    # loop over p values
    for i, p in enumerate(p_arr):
        # derive four base keys from (array_number, i)
        base_seed = array_number * len(p_arr) * 4
        kB_, kBC_, kAB_, kABC_ = (
            PRNGKey(base_seed + 4*i + 0),
            PRNGKey(base_seed + 4*i + 1),
            PRNGKey(base_seed + 4*i + 2),
            PRNGKey(base_seed + 4*i + 3),
        )
        for i in range(array_number):
            kB_ = split(kB_)[0]; kBC_ = split(kBC_)[0]; kAB_ = split(kAB_)[0]; kABC_ = split(kABC_)[0]
        keyB   = split(kB_,   batch_size)
        keyBC  = split(kBC_,  batch_size)
        keyAB  = split(kAB_,  batch_size)
        keyABC = split(kABC_, batch_size)

        # run all four batched samplers
        wb,   keyB   = batch_metro_B   (n,     p, num_samples, keyB,   strs_ini_bc, strs_column_bc, str_shrink_B, str_B_col_ini, str_B_col, str_B_col_end, str_expand_B)
        wbc,  keyBC  = batch_metro_bulk(xbc,  ybc, p, num_samples, keyBC,  strs_ini_bc, strs_column_bc)
        wabc, keyABC = batch_metro_bulk(xabc, yabc,p, num_samples, keyABC, strs_ini_abc, strs_column_abc)
        wab,  keyAB  = batch_metro_AB  (n,     p, num_samples, keyAB,  strs_ini_abc, strs_column_abc, str_shrink,    strs_col_ab_ini, strs_col_ab,    strs_col_ab_end, str_expand)

        # collect
        w_b.append(wb)
        w_bc.append(wbc)
        w_ab.append(wab)
        w_abc.append(wabc)

    # build a filename prefix with all the parameters + array_number
    prefix = (f"domain_wall_m1/array{array_number}"
              f"_n{n}"
              f"_batch{batch_size}"
              f"_numsamp{num_samples}")

    # save them
    jnp.save(f"{prefix}_w_b.npy",   jnp.array(w_b))
    jnp.save(f"{prefix}_w_bc.npy",  jnp.array(w_bc))
    jnp.save(f"{prefix}_w_ab.npy",  jnp.array(w_ab))
    jnp.save(f"{prefix}_w_abc.npy", jnp.array(w_abc))
    jnp.save(f"{prefix}_p_arr.npy", p_arr)

if __name__ == "__main__":
    main()