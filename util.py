import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import jax.numpy as jnp
import random
import math
from itertools import chain
import string


def get_edges(n):
    '''
    left-top corner is (0, 0)
    The plaquette is made of two side rectangles (2n * n) and one middle square (n*n) with boundary removed except for the top boundary.
    '''
    edges = []
    # left-top corner
    for r in range(n):
        for c in range(n - 1, n - 1 - r, -1):
            edge = ((r, c), 'H')
            edges.append(edge)

    for r in range(n):
        for c in range(n - 1, n - r - 1, -1):
            edge = ((r, c), 'V')
            edges.append(edge)

    for r in range(n):
        for c in range(r + 1):
            if c != r:
                edge = ((r, 2 * n + c), 'H')
                edges.append(edge)
            edge = ((r, 2 * n + c), 'V')
            edges.append(edge)

    # Horizontal edges for the top-left and top-right block
    for r in range(n):
        for c in range(n):
            edge1 = ((n + r, c), 'H')
            edge2 = ((n + r, c + 2 * n), 'H')
            edges.append(edge1)
            edges.append(edge2)

    # Vertical edges for the top-left and top-right block
    for r in range(n):
        for c in range(n):
            edge1 = ((n + r, c), 'V')
            if c == 0:
                continue
            edges.append(edge1)
    for r in range(n):
        for c in range(n):
            edge2 = ((n + r, c + 2 * n), 'V')
            if c == 0 and r < n:
                continue
            edges.append(edge2)
    # left-bottom corner
    for r in range(n):
        for c in range(n - r - 1, -1, -1):
            edge = ((2 * n + r, r + c), 'H')
            edges.append(edge)

    for r in range(n):
        for c in range(n - r):
            edge = ((2 * n + r, n - c), 'V')
            edges.append(edge)
    # right-bottom corner
    for r in range(n):
        for c in range(n - r, 0, -1):
            edge = ((2 * n + r, 3 * n - (r + c)), 'H')
            edges.append(edge)
    for r in range(n):
        for c in range(n - r):
            edge = ((2 * n + r, 2 * n + c), 'V')
            edges.append(edge)

    # Vertical edges for the left for the middle square
    for i in range(n):
        edge = ((2 * n + i, n), 'V')
        edges.append(edge)
        edge = ((i, n), 'V')
        edges.append(edge)
    # Horizontal edges for the middle square
    for r in range(n - 1):
        for c in range(n):
            edge = ((r + 2 * n + 1, c + n), 'H')
            edges.append(edge)
            edge = ((r + 1, c + n), 'H')
            edges.append(edge)
    # Vertical edges for the middle square
    for r in range(n):
        for c in range(n - 1):
            edge = ((r + 2 * n, c + n + 1), 'V')
            edges.append(edge)
            edge = ((r, c + n + 1), 'V')
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
            for e in edges_of_plaquette(r, c):
                if e not in setC:
                    setC.add(e)
            for e in edges_of_plaquette(r + n, c):
                if e not in setC:
                    setC.add(e)
            for e in edges_of_plaquette(r + n, c + 2 * n):
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
    for r in range(n):
        for c in range(n - 1):
            setB.add(((r, n + 1 + c), 'V'))
    for r in range(n - 1):
        for c in range(n):
            setB.add(((r + 1, n + c), 'H'))
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


def plaq_ABC_func(n):
    plaq_ABC = []
    #top-middle
    for i in range(n):
        plaq_ABC.append([])
        for j in range(n-1):
            plaq_ABC[i].append((j+1, i+n))
    #top right
    for i in range(n-1):
        plaq_ABC.append([])
        for j in range(n-i-1):
            plaq_ABC[i+n].append((i+j+1, i+2*n))
    # middle right
    for i in range(n):
        plaq_ABC.append([])
        for j in range(n-1):
            plaq_ABC[i+2*n-1].append((n+i, 3*n-j-2))
    # bottom right
    for i in range(n-1):
        plaq_ABC.append([])
        for j in range(n-i-1):
            plaq_ABC[-1].append((2*n+i, 3*n-i-j-2))
    # bottom middle
    for i in range(n):
        plaq_ABC.append([])
        for j in range(n-1):
            plaq_ABC[-1].append((3*n-2-j, 2*n-i-1))
    # bottom left
    for i in range(n-1):
        plaq_ABC.append([])
        for j in range(n-i-1):
            plaq_ABC[-1].append((3*n-2-i-j, n-i-1))
    # middle left
    for i in range(n):
        plaq_ABC.append([])
        for j in range(n-1):
            plaq_ABC[-1].append((2*n-i-1, j+1))
    # top left
    for i in range(n-1):
        plaq_ABC.append([])
        for j in range(n-i-1):
            plaq_ABC[-1].append((n-i-1, i+1+j))
    return plaq_ABC

def plaq_B_func(n):
    plaq_B = []
    for i in range(n-2):
        plaq_B.append([])
        for j in range(n-2):
            plaq_B[i].append((j+1, i+n+1))
    return plaq_B

def plaq_BC_func(n):
    plaq_BC = []
    # middle left
    for i in range(n):
        plaq_BC.append([])
        for j in range(n-2):
            plaq_BC[-1].append((2*n-1-i, j+1))
    # middle top
    for i in range(n-1):
        plaq_BC.append([])
        for j in range(n-i-1):
            plaq_BC[-1].append((n-i-1, i+1+j))
    # top middle
    for i in range(n):
        plaq_BC.append([])
        for j in range(n-2):
            plaq_BC[-1].append((j+1, i+n))
    # top right
    for i in range(n-1):
        plaq_BC.append([])
        for j in range(n-i-1):
            plaq_BC[-1].append((i+j+1, i+2*n))
    # middle right
    for i in range(n):
        plaq_BC.append([])
        for j in range(n-2):
            plaq_BC[-1].append((n+i, 3*n-j-2))
    return plaq_BC


def plaq_AC_func(n):
    plaq_AC = []
    plaq_AC.append([])
    # top right
    for i in range(n - 1):
        plaq_AC[-1].append((i + 1, 2 * n - 1))
    # middle right
    for i in range(n):
        plaq_AC.append([])
        for j in range(n - 1):
            plaq_AC[-1].append((n + i, 3 * n - j - 2))
    # bottom right
    for i in range(n - 1):
        plaq_AC.append([])
        for j in range(n - i - 1):
            plaq_AC[-1].append((2 * n + i, 3 * n - i - j - 2))
    # bottom middle
    for i in range(n):
        plaq_AC.append([])
        for j in range(n - 1):
            plaq_AC[-1].append((3 * n - 2 - j, 2 * n - i - 1))
    # bottom left
    for i in range(n - 1):
        plaq_AC.append([])
        for j in range(n - i - 1):
            plaq_AC[-1].append((3 * n - 2 - i - j, n - i - 1))
    # middle left
    for i in range(n):
        plaq_AC.append([])
        for j in range(n - 1):
            plaq_AC[-1].append((2 * n - i - 1, j + 1))
    # top left
    for i in range(n - 1):
        plaq_AC.append([])
        for j in range(n - i - 1):
            plaq_AC[-1].append((n - i - 1, i + 1 + j))
    plaq_AC.append([])
    for i in range(n - 1):
        plaq_AC[-1].append((i + 1, n))

    return plaq_AC


def T_tensor(p):
    """
    Constructs the two-leg tensor that represents the weight on each edge.

    T_{s1 s2} = delta(s1,s2) * (p**s1)*((1-p)**(1-s1))
    It is a 2x2 diagonal matrix.
    """
    return jnp.array([[1 - p, 0], [0, p]])


def boundary_T_tensor(p):
    return jnp.array([1 - p, p])


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
                    parity = (s1 + s2 + s3 + s4) % 2
                    # If parity matches the anyon measurement, set entry to 1.
                    value = 1 if parity == m else 0
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
                value = 1 if parity == m else 0
                Q = Q.at[s1, s2, s3].set(value)
    return Q


def full_tensor(m, p):
    tensor_list = []
    Q = Q_tensor(m)
    T = T_tensor(p)
    tensor_list.append(Q)
    for i in range(4):
        tensor_list.append(T)
    einsum_str = 'ijkl,ia, jb, kc, ld->abcd'

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


def inner_corner_tensor(m, p):
    tensor_list = []
    Q = Q_tensor(m)
    T = T_tensor(p)
    tensor_list.append(Q)
    for i in range(2):
        tensor_list.append(T)
    einsum_str = 'ijkl, ia, jb->abkl'

    return jnp.einsum(einsum_str, *tensor_list)


def inner_edge_tensor(m, p):
    tensor_list = []
    Q = incomplete_Q_tensor(m)
    T = T_tensor(p)
    tensor_list.append(Q)
    einsum_str = 'ijk, ia->ajk'

    return jnp.einsum(einsum_str, Q, T)