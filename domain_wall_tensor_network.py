import jax
import jax.numpy as jnp
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import random
import math
from itertools import chain
import string
from functools import partial
from jax import config
config.update("jax_enable_x64", True)
def get_edges(n):
    '''
    left-top corner is (0, 0)
    The plaquette is made of two side rectangles (2n * n) and one middle square (n*n) with boundary removed except for the top boundary.
    '''
    edges = []
    # Horizontal edges for the top-left and top-right block
    for r in range(n):
        for c in range(n):
            edge1 = ((r, c), 'H')
            edge2 = ((r, c + 2 * n), 'H')
            edges.append(edge1)
            edges.append(edge2)

    # Vertical edges for the top-left and top-right block
    for r in range(n):
        for c in range(n):
            edge1 = ((r, c), 'V')
            if c == 0:
                continue
            edges.append(edge1)
    for r in range(n):
        for c in range(n):
            edge2 = ((r, c + 2 * n), 'V')
            if c == 0 and r < n:
                continue
            edges.append(edge2)
    # left-bottom corner
    for r in range(n):
        for c in range(n - r - 1, -1, -1):
            edge = ((n + r, r + c), 'H')
            edges.append(edge)

    for r in range(n):
        for c in range(n - r):
            edge = ((n + r, n - c), 'V')
            edges.append(edge)
    # right-bottom corner
    for r in range(n):
        for c in range(n - r, 0, -1):
            edge = ((n + r, 3 * n - (r + c)), 'H')
            edges.append(edge)
    for r in range(n):
        for c in range(n - r):
            edge = ((n + r, 2 * n + c), 'V')
            edges.append(edge)

    # Vertical edges for the left for the middle square
    for i in range(n):
        edge = ((n + i, n), 'V')
        edges.append(edge)
    # Horizontal edges for the middle square
    for r in range(n - 1):
        for c in range(n):
            edge = ((r + n + 1, c + n), 'H')
            edges.append(edge)

            # Vertical edges for the middle square
    for r in range(n):
        for c in range(n - 1):
            edge = ((r + n, c + n + 1), 'V')
            edges.append(edge)
    return edges


def edges_of_plaquette(r, c):
    """
    Return the 4 canonical edges of the plaquette with top-left corner (r,c).
    """
    return [((r, c), 'H'), ((r + 1, c), 'H'), ((r, c), 'V'), ((r, c + 1), 'V')]


def define_B_and_C(n):
    all_edges = set(get_edges(n))
    setC = set()
    for r in range(n):
        for c in range(n):
            for e in edges_of_plaquette(r, c + 2 * n):
                if e not in setC:
                    setC.add(e)
    C_vertex_boundary = [(n, 2 * n + i) for i in range(n + 1)]

    def left_below_edges(r, c):
        res = []
        if r > 0: res.append(((r, c - 1), 'H'))
        res.append(((r, c), 'V'))
        return res

    setB = set()
    # Extending the grid distance by n steps.
    for i in range(n):
        for (r, c) in C_vertex_boundary:
            for e in left_below_edges(r, c):
                if e not in setB and e not in setC:
                    setB.add(e)
        new_boundary = []
        for (r, c) in C_vertex_boundary:
            new_boundary.append((r + 1, c))
        for (r, c) in C_vertex_boundary:
            if (r, c - 1) not in C_vertex_boundary:
                new_boundary.append((r, c - 1))
        C_vertex_boundary = new_boundary

    setA = all_edges - setB - setC

    return setA, setB, setC


def plot_regions(n):
    """
    Plot the lattice showing regions A, B, and C in different colors.
    Region A: edges in A (blue)
    Region B: edges in B (red)
    Region C: edges in C (green)
    """
    allEdges = set(get_edges(n))
    setA, setB, setC = define_B_and_C(n)
    # Ensure each region only contains allowed edges.
    setA = setA.intersection(allEdges)
    setB = setB.intersection(allEdges)
    setC = setC.intersection(allEdges)
    plt.figure(figsize=(7, 7))

    # Plot all edges in A (blue)
    for edge in setA:
        (r, c), typ = edge
        if typ == 'H':
            x1, y1 = c, r
            x2, y2 = c + 1, r
        else:
            x1, y1 = c, r
            x2, y2 = c, r + 1
        plt.plot([x1, x2], [y1, y2], color='blue', lw=2, alpha=0.5)

    # Plot all edges in B (red)
    for edge in setB:
        (r, c), typ = edge
        if typ == 'H':
            x1, y1 = c, r
            x2, y2 = c + 1, r
        else:
            x1, y1 = c, r
            x2, y2 = c, r + 1
        plt.plot([x1, x2], [y1, y2], color='red', lw=2, alpha=0.5)

    # Plot all edges in C (green)
    for edge in setC:
        (r, c), typ = edge
        if typ == 'H':
            x1, y1 = c, r
            x2, y2 = c + 1, r
        else:
            x1, y1 = c, r
            x2, y2 = c, r + 1
        plt.plot([x1, x2], [y1, y2], color='green', lw=2, alpha=0.5)

    # Add a legend manually.
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='blue', lw=2, label='Region A'),
                       Line2D([0], [0], color='red', lw=2, label='Region B'),
                       Line2D([0], [0], color='green', lw=2, label='Region C')]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.title("Regions A (blue), B (red), C (green)")
    plt.xlim(-0.5, n + 0.5)
    plt.ylim(-0.5, n + 0.5)
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.axis('off')
    plt.show()
    print("setA:", setA)
    print("setB:", setB)
    print("setC:", setC)
    return setA, setB, setC

# 1. Local tensor initialization

def full_tensor(p):
    """Construct a full tensor T of shape (2,2,2,2)."""
    T = jnp.zeros((2, 2, 2, 2))
    # Loop over all indices and update via the immutable update API.
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                for l in (0, 1):
                    value = jnp.where((i + j + k + l) % 2 == 0, 1.0 / 2, (1 - 2 * p) ** 2 / 2)
                    T = T.at[i, j, k, l].set(value)
    return T


def incomplete_tensor(p):
    """Construct an incomplete tensor T of shape (2,2,2)."""
    T = jnp.zeros((2, 2, 2))
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                value = jnp.where((i + j + k) % 2 == 0, 1.0 / 2, (1 - 2 * p) ** 2 / 2)
                T = T.at[i, j, k].set(value)
    return T


def corner_tensor(p):
    """Construct a corner tensor T of shape (2,2)."""
    T = jnp.zeros((2, 2))
    for i in (0, 1):
        for j in (0, 1):
            value = jnp.where((i + j) % 2 == 0, 1.0 / 2, (1 - 2 * p) ** 2 / 2)
            T = T.at[i, j].set(value)
    return T


def boundary_full_tensor(p):
    """
    For a full plaquette at the top boundary: sum over the top index of the
    full tensor (shape (2,2,2,2)) to yield a tensor of shape (2,2,2).
    """
    T_full = full_tensor(p)  # shape (2,2,2,2)
    T_top = jnp.sum(T_full, axis=0)
    return T_top


def boundary_incomplete_tensor(p):
    """
    For an incomplete (boundary) plaquette with 3 edges:
      sum over the top index to obtain a tensor of shape (2,2).
    """
    T_inc = incomplete_tensor(p)  # shape (2,2,2)
    T_top_inc = jnp.sum(T_inc, axis=0)
    return T_top_inc


# 2. Initialization of the top row of plaquette tensors

def initial_contraction_top(n, p):
    """
    Build a list of n tensors corresponding to the top row.
    The leftmost and rightmost are boundary incomplete (shape (2,2)),
    while the ones in between are boundary full (shape (2,2,2)).
    """
    tensor_list = []
    tensor_list.append(boundary_incomplete_tensor(p))  # left boundary, shape (2,2)
    for _ in range(n - 2):
        tensor_list.append(boundary_full_tensor(p))  # middle plaquettes, shape (2,2,2)
    tensor_list.append(boundary_incomplete_tensor(p))  # right boundary, shape (2,2)
    return tensor_list


def initial_contraction_bulk(n, p):
    """
    Build a list of n tensors for the bulk contraction.
      Left and right boundaries: incomplete_tensor (shape (2,2)).
      Middle plaquettes: full_tensor (shape (2,2,2)).
    """
    tensor_list = []
    tensor_list.append(incomplete_tensor(p))  # left boundary, shape (2,2)
    for _ in range(n - 2):
        tensor_list.append(full_tensor(p))  # middle plaquettes
    tensor_list.append(incomplete_tensor(p))  # right boundary, shape (2,2)
    return tensor_list


# 3. Contracting the top-boundary tensors via a custom einsum

@partial(jax.jit, static_argnums=(0,))
def contract_top_boundary(n, p):
    """
    Contract a list of n top-boundary tensors from left to right.
    The free vertical indices remain, and internal (horizontal) indices are contracted.
    """
    # Define labels for the indices.
    top_label = string.ascii_letters[:n]
    virtual_label = string.ascii_letters[n:2 * n]
    bottom_label = string.ascii_letters[2 * n:3 * n]

    # Initialize the tensor list.
    tensor_list = initial_contraction_top(n, p)

    einsum_terms = []
    # Left boundary tensor: shape (2,2) → assign indices: [bottom_label[0], virtual_label[0]]
    einsum_terms.append(bottom_label[0] + virtual_label[0])

    # Middle tensors (i = 2,..., n-1): shape (2,2,2)
    for i in range(1, n - 1):
        term = virtual_label[i - 1] + bottom_label[i] + virtual_label[i]
        einsum_terms.append(term)

    # Rightmost tensor: shape (2,2)
    einsum_terms.append(virtual_label[n - 2] + bottom_label[n - 1])

    # The output indices are given by the vertical bonds.
    output_subscript = "".join(bottom_label[i] for i in range(n))
    einsum_str = ",".join(einsum_terms) + "->" + output_subscript
    # Perform the contraction (without optimize=True, since JAX does not support it)
    result = jnp.einsum(einsum_str, *tensor_list)
    return result


@partial(jax.jit, static_argnums=(0,))
def contract_boundary_bulk(n, p, top_tensor):
    """
    Contract the list of n bulk tensors with the top tensor from left to right.
    """
    top_label = string.ascii_letters[:n]
    virtual_label = string.ascii_letters[n:2 * n]
    bottom_label = string.ascii_letters[2 * n:3 * n]

    tensor_list = []
    # Start with the top_tensor.
    tensor_list.append(top_tensor)
    # Append the bulk tensors.
    bulk_tensors = initial_contraction_bulk(n, p)
    for tensor in bulk_tensors:
        tensor_list.append(tensor)

    einsum_terms = []
    # Top part: using the top indices from top_tensor.
    top_terms = "".join(top_label[i] for i in range(n))
    einsum_terms.append(top_terms)

    # Left boundary bulk tensor (shape (2,2,2))
    einsum_terms.append(top_label[0] + virtual_label[0] + bottom_label[0])

    # Middle bulk tensors.
    for i in range(1, n - 1):
        term = top_label[i] + virtual_label[i - 1] + virtual_label[i] + bottom_label[i]
        einsum_terms.append(term)

    # Right boundary bulk tensor.
    einsum_terms.append(top_label[n - 1] + virtual_label[n - 2] + bottom_label[n - 1])

    # The output indices are chosen to be the bottom indices.
    output_subscript = "".join(bottom_label[i] for i in range(n))
    einsum_str_bulk = ",".join(einsum_terms) + "->" + output_subscript

    final_tensor = jnp.einsum(einsum_str_bulk, *tensor_list)
    return final_tensor


@partial(jax.jit, static_argnums=(0,))
def contract_ladder(n, p, initial_tensor):
    """
    Perform a ladder contraction over n tensors.
    """
    top_label = string.ascii_letters[:n]
    virtual_label = string.ascii_letters[n:2 * n + 1]
    bottom_label = string.ascii_letters[2 * n + 1:3 * n]

    tensor = initial_tensor
    for i in range(n - 1):
        tensor_list = []
        einsum_terms = []
        # Append current top tensor.
        tensor_list.append(tensor)
        term = "".join(top_label[k] for k in range(n - i)) + "".join(virtual_label[k] for k in range(i))
        einsum_terms.append(term)

        # Append the corner tensor.
        corner = corner_tensor(p)
        tensor_list.append(corner)
        einsum_terms.append(top_label[0] + virtual_label[i])
        for j in range(n - i - 1):
            tensor_list.append(full_tensor(p))
            einsum_terms.append(virtual_label[i + j] + top_label[j + 1] + virtual_label[i + j + 1] + bottom_label[j])
        output_subscript = "".join(virtual_label[k] for k in range(i)) + "".join(
            bottom_label[k] for k in range(n - i - 1)) + virtual_label[i]
        einsum_str = ",".join(einsum_terms) + "->" + output_subscript
        tensor = jnp.einsum(einsum_str, *tensor_list)
    # Final contraction step.
    tensor_list = []
    einsum_terms = []

    tensor_list.append(tensor)
    einsum_terms.append("".join(top_label[k] for k in range(n - 1)) + virtual_label[n - 1])

    corner = corner_tensor(p)
    tensor_list.append(corner)
    einsum_terms.append(virtual_label[n - 1] + top_label[n - 1])
    output_subscript = "".join(top_label[k] for k in range(n))
    tensor = jnp.einsum(",".join(einsum_terms) + "->" + output_subscript, *tensor_list)

    return tensor


@partial(jax.jit, static_argnums=(0,))
def S_B(n, p):
    """
    Compute the Renyi entropy S_B from the contraction over bulk tensors.
    """
    top_label = string.ascii_letters[:n]
    virtual_label = string.ascii_letters[n:2 * n]
    bottom_label = string.ascii_letters[2 * n:3 * n]

    # Start with an incomplete tensor and contract with a1.
    tensor = incomplete_tensor(p)
    a0 = jnp.array([1.0/2, ((1 - 2 * p) ** 2)/2])
    tensor = jnp.einsum("abc,a->bc", tensor, a0)
    init_corner = corner_tensor(p)
    tensor = jnp.einsum("ab,bc->ac", tensor, init_corner)
    for i in range(2, n):
        tensor_list = []
        einsum_terms = []
        tensor_list.append(tensor)
        einsum_terms.append("".join(top_label[k] for k in range(i)))
        tensor_list.append(incomplete_tensor(p))
        einsum_terms.append(top_label[0] + virtual_label[0] + bottom_label[0])
        for j in range(1, i):
            tensor_list.append(full_tensor(p))
            einsum_terms.append(top_label[j] + virtual_label[j - 1] + virtual_label[j] + bottom_label[j])
        tensor_list.append(corner_tensor(p))
        einsum_terms.append(virtual_label[i - 1] + bottom_label[i])
        output_subscript = "".join(bottom_label[k] for k in range(i + 1))
        einsum_str = ",".join(einsum_terms) + "->" + output_subscript
        tensor = jnp.einsum(einsum_str, *tensor_list)

    flat_tensor = jnp.reshape(tensor, -1)
    renyi = -jnp.log(jnp.sum(flat_tensor ** 2))
    return renyi


@partial(jax.jit, static_argnums=(0,))
def S_turn(n, p, initial_tensor):
    """
    Compute the Renyi entropy S_turn from the contraction of the top tensor.
    """
    top_label = string.ascii_letters[:n]
    virtual_label = string.ascii_letters[n:2 * n]
    bottom_label = string.ascii_letters[2 * n:3 * n]

    tensor = initial_tensor
    for i in range(n - 1):
        tensor_list = []
        einsum_terms = []
        tensor_list.append(tensor)
        einsum_terms.append("".join(top_label[k] for k in range(n - i)))
        corner = corner_tensor(p)
        tensor_list.append(corner)
        einsum_terms.append(top_label[0] + virtual_label[0])
        for j in range(1, n - i - 1):
            tensor_list.append(full_tensor(p))
            einsum_terms.append(top_label[j] + virtual_label[j - 1] + virtual_label[j] + bottom_label[j])
        tensor_list.append(incomplete_tensor(p))
        einsum_terms.append(top_label[n - i - 1] + virtual_label[n - i - 2] + bottom_label[n - i - 1])
        output_subscript = "".join(bottom_label[k] for k in range(1, n - i))
        einsum_str = ",".join(einsum_terms) + "->" + output_subscript
        tensor = jnp.einsum(einsum_str, *tensor_list)

    return -jnp.log(tensor[0]/2 + (1 - 2 * p) ** 2 * tensor[1]/2)

# Sequence contraction version

@partial(jax.jit, static_argnums=(0,))
def contract_top_boundary_seq(n, p):
    """
    Contract a list of n top-boundary tensors from left to right.
    The free vertical indices remain, and internal (horizontal) indices are contracted.
    """
    # Define labels for the indices.
    label = string.ascii_letters
    # Initialize the tensor list.
    tensor_list = initial_contraction_top(n, p)

    # Left boundary tensor: shape (2,2) → assign indices: [bottom_label[0], virtual_label[0]]
    top_term = (label[0] + label[1])
    top_tensor = tensor_list[0]
    # Middle tensors (i = 2,..., n-1): shape (2,2,2)
    for i in range(1, n - 1):
        term = label[i] + label[i + 1] + label[i + 2]
        output_subscript = top_term[:-1] + label[i + 1] + label[i + 2]
        einsum_subscript = top_term + "," + term + "->" + output_subscript
        top_tensor = jnp.einsum(einsum_subscript, top_tensor, tensor_list[i])
        top_term = label[:len(output_subscript)]

    # Rightmost tensor: shape (2,2)
    term = label[n - 1] + label[n]
    output_subscript = label[:n]
    einsum_subscript = top_term + "," + term + "->" + output_subscript
    top_tensor = jnp.einsum(einsum_subscript, top_tensor, tensor_list[n - 1])

    return top_tensor


@partial(jax.jit, static_argnums=(0,))
def contract_boundary_bulk_seq(n, p, top_tensor):
    """
    Contract the list of n bulk tensors with the top tensor from left to right.
    """
    label = string.ascii_letters
    bulk_tensors = initial_contraction_bulk(n, p)
    top_terms = "".join(label[i] for i in range(n))

    term = label[0] + label[n] + label[n + 1]
    output_subscript = top_terms[1:] + label[n] + label[n + 1]

    einsum_subscript = top_terms + "," + term + "->" + output_subscript
    top_tensor = jnp.einsum(einsum_subscript, top_tensor, bulk_tensors[0])
    top_terms = label[:len(output_subscript)]

    for i in range(1, n - 1):
        term = label[i] + label[i + 1] + label[n + 1] + label[n + 2]
        output_subscript = top_terms[1:i + 1] + label[n + 1] + label[n + 2] + top_terms[i + 2:]
        einsum_subscript = top_terms + "," + term + "->" + output_subscript
        top_tensor = jnp.einsum(einsum_subscript, top_tensor, bulk_tensors[i])

    term = label[n - 1] + label[n] + label[n + 1]
    output_subscript = label[:n]
    einsum_subscript = top_terms + "," + term + "->" + output_subscript
    final_tensor = jnp.einsum(einsum_subscript, top_tensor, bulk_tensors[n - 1])

    return final_tensor


@partial(jax.jit, static_argnums=(0,))
def contract_ladder_seq(n, p, initial_tensor):
    """
    Perform a ladder contraction over n tensors.
    """
    label = string.ascii_letters

    tensor = initial_tensor
    for i in range(n - 1):
        term1 = "".join(label[k] for k in range(n))
        term2 = label[0] + label[n]
        output_subscript = label[n] + term1[1:]
        einsum_subscript = term1 + "," + term2 + "->" + output_subscript
        tensor = jnp.einsum(einsum_subscript, tensor, corner_tensor(p))

        for j in range(n - i - 1):
            term2 = label[j] + label[j + 1] + label[n] + label[n + 1]
            output_subscript = label[:j] + label[n] + label[n + 1] + label[j + 2:n]
            einsum_str = term1 + "," + term2 + "->" + output_subscript
            tensor = jnp.einsum(einsum_str, tensor, full_tensor(p))
    # Final contraction step.

    corner = corner_tensor(p)
    term2 = label[0] + label[n]
    output_subscript = label[1:n + 1]
    einsum_str = term1 + "," + term2 + "->" + output_subscript
    tensor = jnp.einsum(einsum_str, tensor, corner)

    return tensor


def S_B_seq(n, p):
    """
    Compute the Renyi entropy S_B from the contraction over bulk tensors.
    """
    label = string.ascii_letters

    # Start with an incomplete tensor and contract with a1.
    tensor = incomplete_tensor(p)
    a0 = jnp.array([1.0 / 2, ((1 - 2 * p) ** 2) / 2])
    tensor = jnp.einsum("abc,a->bc", tensor, a0)
    init_corner = corner_tensor(p)
    tensor = jnp.einsum("ab,bc->ac", tensor, init_corner)
    for i in range(2, n):

        term = "".join(label[k] for k in range(i))
        edge = incomplete_tensor(p)
        edge_term = label[0] + label[i] + label[i + 1]
        output_subscript = term[1:] + label[i] + label[i + 1]
        einsum_subscript = term + "," + edge_term + "->" + output_subscript
        tensor = jnp.einsum(einsum_subscript, tensor, edge)

        for j in range(1, i):
            term = "".join(label[k] for k in range(i + 1))
            bulk = full_tensor(p)
            bulk_term = label[j] + label[j + 1] + label[i + 1] + label[i + 2]
            output_subscript = label[:j] + label[j + 2:i + 3]
            einsum_subscript = term + "," + bulk_term + "->" + output_subscript
            tensor = jnp.einsum(einsum_subscript, tensor, bulk)

        corner = corner_tensor(p)
        term = label[:i + 1]
        corner_term = label[i] + label[i + 1]
        output_subscript = label[:i] + label[i + 1]
        einsum_str = term + "," + corner_term + "->" + output_subscript
        tensor = jnp.einsum(einsum_str, tensor, corner)

    flat_tensor = jnp.reshape(tensor, -1)
    renyi = -jnp.log(jnp.sum(flat_tensor ** 2))
    return renyi


@partial(jax.jit, static_argnums=(0,))
def S_turn_seq(n, p, initial_tensor):
    """
    Compute the Renyi entropy S_turn from the contraction of the top tensor.
    """
    label = string.ascii_letters

    tensor = initial_tensor
    for i in range(n - 1):
        corner = corner_tensor(p)
        term = label[:n - i]
        corner_term = label[0] + label[n - i]
        output_subscript = term[1:] + label[n - i]
        einsum_str = term + "," + corner_term + "->" + output_subscript
        tensor = jnp.einsum(einsum_str, tensor, corner)

        for j in range(1, n - i - 1):
            bulk = full_tensor(p)
            bulk_term = label[j - 1] + label[j] + label[n - i] + label[n - i + 1]
            output_subscript = label[:j - 1] + label[j + 1:n - i + 2]
            einsum_str = term + "," + bulk_term + "->" + output_subscript
            tensor = jnp.einsum(einsum_str, tensor, bulk)

        edge = incomplete_tensor(p)
        edge_term = label[n - i - 2] + label[n - i - 1] + label[n - i]
        output_subscript = label[:n - i - 2] + label[n - i]
        einsum_str = term + "," + edge_term + "->" + output_subscript
        tensor = jnp.einsum(einsum_str, tensor, edge)

    return -jnp.log(tensor[0] / 2 + (1 - 2 * p) ** 2 * tensor[1] / 2)


# 4. Main demonstration
CMI = []
# Create a probability array. Using jnp.linspace and jnp.concatenate.
prob = jnp.concatenate((jnp.linspace(0, 0.25, 10), jnp.linspace(0.25, 0.5, 100)))

# Loop over n and p.
# Convert the DeviceArray 'prob' to a regular Python list using .tolist()

for n in range(2, 31):
    for p in prob.tolist():
        # Contract the top-boundary row.
        top_tensor = contract_top_boundary_seq(n, p)
        for _ in range(n - 1):
            final_tensor = contract_boundary_bulk_seq(n, p, top_tensor)
            top_tensor = final_tensor
        top_tensor = contract_ladder_seq(n, p, top_tensor)
        renyi_BC = S_turn_seq(n, p, top_tensor)
        for _ in range(n):
            final_tensor = contract_boundary_bulk_seq(n, p, top_tensor)
            top_tensor = final_tensor
        renyi_AB = S_turn_seq(n, p, top_tensor)

        top_tensor = contract_ladder_seq(n, p, top_tensor)
        for _ in range(n):
            final_tensor = contract_boundary_bulk_seq(n, p, top_tensor)
            top_tensor = final_tensor
        renyi_ABC = -jnp.log(jnp.sum(top_tensor))

        # S_B from the bulk.
        renyi_B = S_B_seq(n, p)
        # Compute the conditional mutual information.
        CMI.append(renyi_AB + renyi_BC - renyi_ABC - renyi_B)
CMI = jnp.array(CMI).reshape(-1, len(prob))
jnp.save("result/CMI.npy", CMI)
for i in range(CMI.shape[0]):
    plt.plot(prob, CMI[i], label=f"n={i + 2}")
plt.legend()
plt.savefig("result/CMI.png")
plt.show()
