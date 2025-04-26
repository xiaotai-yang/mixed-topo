import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import jax.numpy as jnp
import random
import math
from itertools import chain
import string
from jax import lax

def get_edges(n):
    '''
    left-top corner is (0, 0)
    The plaquette is made of two side rectangles (2n * n) and one middle square (n*n) with boundary removed except for the top boundary.
    '''
    edges = []
    # left-top corner
    for r in range(3*n):
        for c in range(5*n):
            edge = ((r, c), 'H')
            edges.append(edge)
            edge = ((r, c), 'V')
            edges.append(edge)
    for r in range(3*n):
        edge = ((r, 5*n), 'V')
        edges.append(edge)
    for c in range(5*n):
        edge = ((3*n, c), 'H')
        edges.append(edge)
    for c in range(5*n):
        edges.remove(((0, c), 'H'))

    return edges

def define_B_and_C(n):
    all_edges = set(get_edges(n))
    setC = set()
    for r in range(2*n):
        for c in range(n, 4*n):
            setC.add(((r, c), 'H'))
            setC.add(((r, c), 'V'))
    for r in range(2*n):
        setC.add(((r, 4*n), 'V'))
    for c in range(3*n):
        setC.add(((2*n, n+c), 'H'))

    setB = set()
    # Extending the grid distance by n steps.
    for r in range(n):
        for c in range(2*n, 3*n):
            setB.add(((r, c), 'H'))
            setB.add(((r, c), 'V'))
    for r in range(n):
        setB.add(((r, 3*n), 'V'))
    for c in range(n):
        setB.add(((n, 2*n+c), 'H'))
    setC = setC - setB
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
    setA = setA.intersection(allEdges)
    setB = setB.intersection(allEdges)
    setC = setC.intersection(allEdges)
    bdy = []
    for i in range(n+2):
        bdy.append(((0, n - 1 + i), 'H'))
    bdy.append(((0, n - 1), 'V'))
    bdy.append(((0, 2*n +  1), 'V'))
    for i in range(n - 1):
        bdy.append(((n - 1 - i, i), 'H'))
        bdy.append(((n - 1 - i, i), 'V'))
        bdy.append(((n - 1 - i, 3 * n - 1 - i), 'H'))
        bdy.append(((n - 1 - i, 3 * n - i), 'V'))
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
                       Line2D([0], [0], color='red', lw=2, label='Region C'),
                       Line2D([0], [0], color='green', lw=2, label='Region B')]
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

def boundary_T_tensor(p):
    return jnp.sqrt(jnp.array([1 - p, p]))*(1-(1-2**(0.25))*2*p)



def T_tensor(p):
    """
    Constructs the two-leg tensor that represents the weight on each edge.

    T_{s1 s2} = delta(s1,s2) * (p**s1)*((1-p)**(1-s1))
    It is a 2x2 diagonal matrix.
    """
    return jnp.sqrt(jnp.array([[1 - p, 0], [0, p]]))*(1-(1-2**(0.25))*2*p)

def Q_tensor(m):
    """
    Constructs the four-leg tensor for a plaquette that enforces the parity.

    Q^m_{s1 s2 s3 s4} = 1 if (s1 + s2 + s3 + s4) mod 2 equals m, else 0.
    The tensor has shape (2,2,2,2).

    m should be 0 or 1.
    """
    Q = jnp.zeros((2, 2, 2, 2))
    # Loop over all 16 combinations of {s1, s2, s3, s4}.
    for s1 in (0, 1):
        for s2 in (0, 1):
            for s3 in (0, 1):
                for s4 in (0, 1):
                    #print(s1, s2, s3, s4)
                    parity = (s1 + s2 + s3 + s4) % 2
                    # If parity matches the anyon measurement, set entry to 1.
                    value = lax.select(parity == m, 1., 0.)
                    Q = Q.at[s1, s2, s3, s4].set(value)
    return Q


def incomplete_Q_tensor(m):
    Q = jnp.zeros((2, 2, 2))
    # Loop over all 16 combinations of {s1, s2, s3, s4}.
    for s1 in (0, 1):
        for s2 in (0, 1):
            for s3 in (0, 1):
                parity = (s1 + s2 + s3) % 2
                # If parity matches the anyon measurement, set entry to 1.
                value = lax.select(parity == m, 1., 0.)
                Q = Q.at[s1, s2, s3].set(value)
    return Q

def corner_Q_tensor(m):
    Q = jnp.zeros((2, 2))
    # Loop over all 16 combinations of {s1, s2, s3, s4}.
    for s1 in (0, 1):
        for s2 in (0, 1):
            parity = (s1 + s2) % 2
            # If parity matches the anyon measurement, set entry to 1.
            value = lax.select(parity == m, 1., 0.)
            Q = Q.at[s1, s2].set(value)
    return Q

def full_tensor(m, p):
    tensor_list = []
    Q = Q_tensor(m)
    T = T_tensor(p)
    tensor_list.append(Q)
    for i in range(4):
        tensor_list.append(T)
    einsum_str = 'ijkl, ia, jb, kc, ld->abcd'

    return jnp.einsum(einsum_str, *tensor_list)


def incomplete_tensor(m, p):
    tensor_list = []
    Q = Q_tensor(m)
    T = T_tensor(p)
    bT = boundary_T_tensor(p)
    tensor_list.append(Q)
    for i in range(3):
        tensor_list.append(T)
    tensor_list.append(bT)
    einsum_str = 'ijkl, ia, jb, kc, l->abc'

    return jnp.einsum(einsum_str, *tensor_list)


def corner_tensor(m, p):
    tensor_list = []
    Q = Q_tensor(m)
    T = T_tensor(p)
    bT = boundary_T_tensor(p)
    tensor_list.append(Q)
    for i in range(2):
        tensor_list.append(T)
    tensor_list.append(bT)
    tensor_list.append(bT)
    einsum_str = 'ijkl, ia, jb, k, l->ab'

    return jnp.einsum(einsum_str, *tensor_list)


def inner_Q_tensor(m, p):
    tensor_list = []
    Q = Q_tensor(m)
    T = T_tensor(p)
    tensor_list.append(Q)
    for i in range(2):
        tensor_list.append(T)
    einsum_str = 'ijkl, ia, jb->abkl'

    return jnp.einsum(einsum_str, *tensor_list)


def inner_edge_tensor(m, p):
    Q = incomplete_Q_tensor(m)
    T = T_tensor(p)
    einsum_str = 'ijk, ja->iak'

    return jnp.einsum(einsum_str, Q, T)

def inner_edge_corner_tensor(m, p):
    Q = Q_tensor(m)
    T = T_tensor(p)
    einsum_str = 'hijk, ia, jb->habk'
    return jnp.einsum(einsum_str, Q, T, T)

def inner_edge_corner_end_tensor(m, p):
    Q = Q_tensor(m)
    T = T_tensor(p)
    einsum_str = 'ijkl, ja, kb->iabl'
    return jnp.einsum(einsum_str, Q, T, T)

def inner_corner_start_tensor(m, p):
    Q = corner_Q_tensor(m)
    T = T_tensor(p)
    einsum_str = 'ij, ja -> ia'
    return jnp.einsum(einsum_str, T, Q)

def inner_corner_end_tensor(m, p):
    Q = corner_Q_tensor(m)
    T = T_tensor(p)
    einsum_str = 'ij, ja->ia'
    return jnp.einsum(einsum_str, Q, T)