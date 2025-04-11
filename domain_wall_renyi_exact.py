import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import random
import math
from itertools import chain
import string
from itertools import product
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


# ----- Plotting Regions A, B, and C -----

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


def generate_initial_config(edges):
    """
    Generate the initial configuration: a dictionary for all edges (not in D) set to 0 (inactive).
    """
    config = {}
    for e in edges:
        config[e] = 0
    return config


def sample_step(config, all_edges):
    for edge in all_edges:
        p = random.random()
        if p < 0.5:
            config[edge] = 1 - config[edge]
    return config


def sample_configurations(all_edges, num_samples):
    """
    Uniform sampling to get the configurations.

    Args:
      num_samples: Number of configurations to sample.

    Returns:
      A list of configuration dictionaries.
    """
    samples = []
    config = generate_initial_config(all_edges)

    for _ in range(num_samples):
        sample_step(config, all_edges)
        samples.append(config.copy())
    return samples


def region_plaquettes(regionEdges, all_plaq):
    """
    For a given set of allowed edges (regionEdges), return the list of plaquettes
    (by their top-left coordinate) that are partially contained in the region.
    """
    plaq = []
    for (r, c) in all_plaq:
        edges = [((r, c), 'H'),
                 ((r + 1, c), 'H'),
                 ((r, c), 'V'),
                 ((r, c + 1), 'V')]
        if any(e in regionEdges for e in edges):
            plaq.append((r, c))
    return plaq


def region_weight(config, regionPlaquettes, p):
    """
    For a given configuration and region:
      - Count the number of excitations in the plaquettes inside the region.
      - Return the weight (1-2p)^(2 * (# excitations)).
    """
    # Check for active edge outside the region.

    num_excited = 0
    for (r, c) in regionPlaquettes:
        edges = [((r, c), 'H'), ((r + 1, c), 'H'), ((r, c), 'V'), ((r, c + 1), 'V')]
        count = sum(config.get(e, 0) for e in edges)
        if count % 2 == 1:
            num_excited += 1
    if num_excited == 0:
        return 1
    else:
        return (1 - 2 * p) ** (2 * num_excited)

def all_renyi_configs(edges):
    """
    Generate all configurations: each edge can be 0 or 1.
    Yields dictionaries mapping each edge to 0 or 1.
    """
    for states in product([0, 1], repeat=len(edges)):
        yield dict(zip(edges, states))

# ===== Main: Sample configurations and compute Renyi-2 weights for regions =====

n = 1  # Lattice size
CMI = []  # Store CMI values for each p.
std_CMI = []
all_plaq = list(chain(
    ((r, c) for r in range(2 * n) for c in range(n)),
    ((r, c + 2 * n) for r in range(2 * n) for c in range(n)),
    ((r, c + n) for r in range(n, 2 * n) for c in range(n))
))
allEdges = set(get_edges(n))

# Define A, B, C regions
setA, setB, setC = define_B_and_C(n)
# Intersect with allEdges to be safe:
setA = setA.intersection(allEdges)
setB = setB.intersection(allEdges)
setC = setC.intersection(allEdges)

# Define regions:
region_ABC = allEdges  # Entire allowed lattice
region_AB = setA.union(setB)
region_BC = setB.union(setC)
region_B = setB
region_C = setC
# For each region, get the list of plaquettes that are fully contained in it.
plaq_ABC = region_plaquettes(region_ABC, all_plaq)
plaq_AB = region_plaquettes(region_AB, all_plaq)
plaq_BC = region_plaquettes(region_BC, all_plaq)
plaq_B = region_plaquettes(region_B, all_plaq)
plaq_C = region_plaquettes(region_C, all_plaq)

exp_S2_B = 0
exp_S2_AB = 0
exp_S2_BC = 0
exp_S2_ABC = 0
CMI_exact = []

p = 0.35  # Probability of excitation
for config in all_renyi_configs(setB):
    w_B = region_weight(config, plaq_B, p)
    exp_S2_B += w_B
for config in all_renyi_configs(setA.union(setB)):
    w_AB = region_weight(config, plaq_AB, p)
    exp_S2_AB += w_AB
for config in all_renyi_configs(setB.union(setC)):
    w_BC = region_weight(config, plaq_BC, p)
    exp_S2_BC += w_BC
for config in all_renyi_configs(allEdges):
    w_ABC = region_weight(config, plaq_ABC, p)
    exp_S2_ABC += w_ABC

S2_B = -np.log(exp_S2_B)
S2_AB = -np.log(exp_S2_AB)
S2_BC = -np.log(exp_S2_BC)
S2_ABC = -np.log(exp_S2_ABC)
CMI_exact.append(-(S2_ABC - S2_AB - S2_BC + S2_B))
np.save("CMI_exact.npy", np.array(CMI_exact))
print("CMI_exact:", CMI_exact)
