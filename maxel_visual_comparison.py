import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import Counter
import random
import timeit
from typing import Dict, Set

# --- Maxel Class (Copy-pasted for standalone script) ---
# In a real project, this would be: `from maxel import Maxel`
class Maxel:
    def __init__(self, data: Dict[int, Counter] = None, support: Set[int] = None):
        self.data: Dict[int, Counter] = data if data is not None else {}
        self.support: Set[int] = support if support is not None else set()
        if not self.support:
            self._update_support()

    def _update_support(self):
        nodes = set(self.data.keys())
        for vexel in self.data.values():
            nodes.update(vexel.keys())
        self.support = nodes

    def __add__(self, other: 'Maxel') -> 'Maxel':
        new_data = {}
        all_nodes = self.support.union(other.support)
        for i in all_nodes:
            v1 = self.data.get(i, Counter())
            v2 = other.data.get(i, Counter())
            if v1 or v2:
                new_data[i] = v1 + v2
        return Maxel(new_data)

    def restrict(self, j: Set[int]) -> 'Maxel':
        new_data = {}
        for i in j:
            if i in self.data:
                original_vexel = self.data[i]
                new_vexel = Counter()
                for target, count in original_vexel.items():
                    if target in j:
                        new_vexel[target] = count
                if new_vexel:
                    new_data[i] = new_vexel
        return Maxel(new_data, j)

    def __matmul__(self, other: 'Maxel') -> 'Maxel':
        new_data = {}
        for i in self.support:
            r_i = self.data.get(i, Counter())
            if not r_i: continue
            result_vexel = Counter()
            for k, count_ik in r_i.items():
                c_k = other.data.get(k, Counter())
                if not c_k: continue
                for j, count_kj in c_k.items():
                    walk_count = count_ik * count_kj
                    result_vexel[j] += walk_count
            if result_vexel:
                new_data[i] = result_vexel
        return Maxel(new_data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Maxel):
            return NotImplemented
        return self.data == other.data

    def __repr__(self):
        return f"Maxel(data={self.data}, support={self.support})"

    def __str__(self):
        output = [f"--- Maxel Graph (Support J={sorted(list(self.support))}) ---"]
        for i, vexel in sorted(self.data.items()):
            edges = []
            for j, count in sorted(vexel.items()):
                edges.append(f"{j} (x{count})")
            output.append(f"Node {i} -> " + ", ".join(edges))
        return "\n".join(output)

# --- Graph Visualization Function ---
def draw_maxel_graph(ax, maxel_obj: Maxel, title: str):
    G = nx.MultiDiGraph() # Use MultiDiGraph to handle multiple edges (counts)

    all_nodes = list(maxel_obj.support)
    if not all_nodes:
        ax.set_title(f"{title}\n(Empty Graph)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # Add nodes to the graph
    G.add_nodes_from(all_nodes)

    # Add edges with weights (counts)
    for source, vexel in maxel_obj.data.items():
        for target, count in vexel.items():
            for _ in range(count): # Add 'count' number of edges for visualization
                G.add_edge(source, target, weight=count)

    # Position nodes using a layout algorithm
    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42) # 'k' adjusts distance, 'seed' for reproducibility

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightgreen', node_size=700)

    # Draw edges
    edge_labels = {}
    unique_edges = set() # To store (u, v) for labeling, avoiding duplicate labels for multiedges

    for u, v, data in G.edges(data=True):
        edge_weight = data['weight']
        if (u,v) not in unique_edges: # Only label the first instance of a multiedge
            edge_labels[(u, v)] = f"x{edge_weight}"
            unique_edges.add((u,v))

    # Draw standard arrows for directed edges
    nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle='->', arrowsize=20, edge_color='gray', width=1)

    # Draw edge labels (weights)
    # Using 'bbox' to give a background to the label, making it more readable
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_color='red',
                                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Draw node labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_color='black')

    ax.set_title(title, fontsize=12)
    ax.axis('off') # Hide axes

# --- Graph Generation Helper (from previous response) ---
def generate_maxel(N: int, edge_density: float, max_weight: int) -> Maxel:
    data = {}
    nodes = list(range(N))
    for i in nodes:
        vexel = Counter()
        for j in nodes:
            if random.random() < edge_density:
                vexel[j] = random.randint(1, max_weight)
        if vexel:
            data[i] = vexel
    return Maxel(data, set(nodes))

# --- Convert Maxel to Dense Matrix (from previous response) ---
def maxel_to_dense_matrix(maxel_obj: Maxel) -> np.ndarray:
    nodes = sorted(list(maxel_obj.support))
    N = len(nodes)
    node_to_index = {node: i for i, node in enumerate(nodes)}
    if N == 0:
        return np.array([]), []
    matrix = np.zeros((N, N), dtype=int)
    for i, vexel in maxel_obj.data.items():
        if i in node_to_index:
            row_idx = node_to_index[i]
            for j, count in vexel.items():
                if j in node_to_index:
                    col_idx = node_to_index[j]
                    matrix[row_idx, col_idx] = count
    return matrix, nodes

# --- BENCHMARK EXECUTION (from previous response) ---
def run_benchmark(m1, m2, N, runs=10):
    # 1. Maxel Multiplication (Sparse Method)
    maxel_time = timeit.timeit(lambda: m1 @ m2, number=runs)

    # 2. Dense Matrix Multiplication (NumPy Method)
    m1_dense_data, _ = maxel_to_dense_matrix(m1)
    m2_dense_data, _ = maxel_to_dense_matrix(m2)
    dense_time = timeit.timeit(lambda: np.matmul(m1_dense_data, m2_dense_data), number=runs)

    return maxel_time / runs, dense_time / runs

# --- Main Visualization Script ---
if __name__ == "__main__":
    # --- Part 1: Graph Representation of Maxel MVP Operations ---
    fig1, axs1 = plt.subplots(1, 3, figsize=(18, 6))
    fig1.suptitle('Maxel MVP Operations: Graph Visualization', fontsize=16)

    # Maxel 1
    m1_viz = Maxel({
        1: Counter({2: 2, 3: 1}),
        2: Counter({3: 1, 4:1})
    }, {1, 2, 3, 4})
    draw_maxel_graph(axs1[0], m1_viz, "Initial Maxel (m1)")

    # Restriction Example
    j_restrict = {1, 2, 3}
    m_restricted = m1_viz.restrict(j_restrict)
    draw_maxel_graph(axs1[1], m_restricted, f"Restricted to J={j_restrict}")

    # Multiplication Example (simple walk)
    m2_viz = Maxel({
        2: Counter({5: 1}),
        3: Counter({5: 2})
    }, {1, 2, 3, 4, 5})
    m_product = m1_viz @ m2_viz # m1 -> k -> m2
    draw_maxel_graph(axs1[2], m_product, "Product (m1 @ m2) - Walks of length 2")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
   # plt.show() # Display MVP operations graph visualization
    plt.savefig("maxel_operations_viz.png", bbox_inches='tight') # <-- ADDED
    plt.close(fig1) # <-- ADDED (Close fig1 to free memory)

    # --- Part 2: Efficiency Comparison Visualization ---
    N = 200 # Using a smaller N for quicker demo, can increase to 500 for more dramatic results
    MAX_WEIGHT = 1
    NUM_RUNS = 5 # Fewer runs for quicker demo

    print(f"\n--- Running Efficiency Benchmark (N={N} Nodes, Avg over {NUM_RUNS} runs) ---")

    # Scenario 1: SPARSE Graph (Density 1%)
    SPARSE_DENSITY = 0.01
    m1_sparse = generate_maxel(N, SPARSE_DENSITY, MAX_WEIGHT)
    m2_sparse = generate_maxel(N, SPARSE_DENSITY, MAX_WEIGHT)
    maxel_sparse_time, dense_sparse_time = run_benchmark(m1_sparse, m2_sparse, N, runs=NUM_RUNS)
    print(f"SPARSE (Density {SPARSE_DENSITY*100}%): Maxel={maxel_sparse_time:.6f}s, NumPy={dense_sparse_time:.6f}s")

    # Scenario 2: DENSE Graph (Density 50%)
    DENSE_DENSITY = 0.5
    m1_dense = generate_maxel(N, DENSE_DENSITY, MAX_WEIGHT)
    m2_dense = generate_maxel(N, DENSE_DENSITY, MAX_WEIGHT)
    maxel_dense_time, dense_dense_time = run_benchmark(m1_dense, m2_dense, N, runs=NUM_RUNS)
    print(f"DENSE (Density {DENSE_DENSITY*100}%): Maxel={maxel_dense_time:.6f}s, NumPy={dense_dense_time:.6f}s")


    # --- Plotting the Benchmark Results ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    data_maxel = [maxel_sparse_time, maxel_dense_time]
    data_numpy = [dense_sparse_time, dense_dense_time]

    scenarios = ['Sparse Graph', 'Dense Graph']
    x = np.arange(len(scenarios))
    width = 0.35

    rects1 = ax2.bar(x - width/2, data_maxel, width, label='Maxel (Sparse Method)', color='skyblue')
    rects2 = ax2.bar(x + width/2, data_numpy, width, label='NumPy (Dense Method)', color='tomato')

    ax2.set_ylabel('Execution Time (seconds, log scale)', fontsize=12)
    ax2.set_title(f'Efficiency Comparison: Maxel vs. Dense Matrix Multiplication (N={N})', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, fontsize=10)
    ax2.legend()
    ax2.set_yscale('log') # Log scale is crucial for large differences
    ax2.set_ylim(min(data_maxel + data_numpy) * 0.5, max(data_maxel + data_numpy) * 2)

    def autolabel_time(rects):
        for rect in rects:
            height = rect.get_height()
            ax2.annotate(f'{height:.3g}s', # Using .3g for potentially very small or large numbers
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel_time(rects1)
    autolabel_time(rects2)

    fig2.tight_layout()
 #   plt.show() # Display effic
    plt.savefig("maxel_efficiency_benchmark.png", bbox_inches='tight') # <-- ADDED
    plt.close(fig2) # <-- ADDED (Close fig2 to free memory)
