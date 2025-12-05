import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from typing import Set, Dict, Any, Tuple

# --- Maxel/Vexel Foundation Class (Obfuscated Core) ---

class Maxel:
    """
    A Maxel object representing a sparse, multiset-based graph.
    The graph is stored as a dictionary mapping source nodes (i) to their
    out-neighborhood Vexels (r_i), which are Counters.
    """
    def __init__(self, data: Dict[int, Counter] = None, support: Set[int] = None):
        # The core storage: {source_id: Counter(target_id: count)}
        self.data: Dict[int, Counter] = data if data is not None else {}

        # The active support set J (important for Restriction/e_J)
        self.support: Set[int] = support if support is not None else set()
        if not self.support:
            self._update_support()

    def _update_support(self):
        """Recalculates the active support set J based on current edges."""
        nodes = set(self.data.keys())
        for vexel in self.data.values():
            nodes.update(vexel.keys())
        self.support = nodes

    def get_value(self, i: int, j: int) -> int:
        return self.data.get(i, Counter()).get(j, 0)

    # --- MVP Operation 1: Vexel Union (Maxel Addition) ---
    def __add__(self, other: 'Maxel') -> 'Maxel':
        """Performs Maxel Addition (Multiset Union of edges)."""
        new_data = {}
        # Get all unique node IDs across both Maxels
        all_nodes = self.support.union(other.support)

        for i in all_nodes:
            # Vexel addition is equivalent to Counter addition
            v1 = self.data.get(i, Counter())
            v2 = other.data.get(i, Counter())
            if v1 or v2:
                # Counter addition performs multiset union (summing counts)
                new_data[i] = v1 + v2

        return Maxel(new_data)

    # --- MVP Operation 2: Maxel Restriction (Induced Subgraph e_J m e_J) ---
    def restrict(self, J: Set[int]) -> 'Maxel':
        """
        Calculates the induced subgraph m' = e_J m e_J.
        This is a metadata/filtering operation, not a multiplication.
        """
        new_data = {}
        # 1. Restrict Source Nodes (i): Keep only nodes i in J
        for i in J:
            if i in self.data:
                original_vexel = self.data[i]
                new_vexel = Counter()

                # 2. Restrict Target Nodes (j): Keep only targets j in J
                for j, count in original_vexel.items():
                    if j in J:
                        new_vexel[j] = count

                if new_vexel:
                    new_data[i] = new_vexel

        return Maxel(new_data, J)

    # --- MVP Operation 3: Maxel Multiplication (Walk Counting) ---
    def __matmul__(self, other: 'Maxel') -> 'Maxel':
        """
        Performs Maxel Multiplication (m1 @ m2), counting walks of length 2.
        Walks (i -> k -> j) are counted by the multiset definition of the product.
        """
        new_data = {}
        # The result Maxel is defined on the union of support sets
        result_support = self.support.union(other.support)

        # Iterate over all possible starting nodes (i) in the result
        for i in self.support:
            r_i = self.data.get(i, Counter()) # r_i is the Vexel of m1
            result_vexel = Counter()

            # Iterate over all intermediate nodes (k) reachable from i
            for k, count_ik in r_i.items():
                # c_k is the Vexel of m2's out-neighborhood (k -> j)
                c_k = other.data.get(k, Counter())

                # For every walk i -> k -> j:
                # Add (count_ik * count_kj) to the total walk count i -> j
                for j, count_kj in c_k.items():
                    # count_ik and count_kj are scalars/integers
                    walk_count = count_ik * count_kj
                    result_vexel[j] += walk_count

            if result_vexel:
                new_data[i] = result_vexel

        return Maxel(new_data, result_support)

    def __eq__(self, other: 'Maxel') -> bool:
        """Simple check for demonstration purposes."""
#        if not isinstance(other, Maxel): return False

        # Check that non-zero entries are identical
#        self_entries = {(i, j, v) for i, c in self.data.items() f>
#        other_entries = {(i, j, v) for i, c in other.data.items()>

#        return self_entries == other_entries

        if not isinstance(other, Maxel): return False

        # Extract all (i, j, value) tuples where value is non-zero
        self_entries = {(i, j, v)
                        for i, c in self.data.items()
                        for j, v in c.items() if v != 0}

        other_entries = {(i, j, v)
                         for i, c in other.data.items()
                         for j, v in other.data.get(i, Counter()).items() if v != 0}

        # The comparison should succeed if the set of non-zero (co>
        return self_entries == other_entries

#class Maxel:
#    """
#    Generalized Maxel representation for graph adjacency matrices/tensors.
#    Handles sparse, weighted data over any semiring/field.
#    """
#    def __init__(self, data: Dict[int, Counter], support: Set[int] = None):
#        self.data = data
#        if support is None:
#            self.support = set(data.keys()) | {n for counters in data.values() for n in counters.keys()}
#        else:
#            self.support = support
#        self.nodes = sorted(list(self.support))
#        self.node_map = {node: i for i, node in enumerate(self.nodes)}
#
#    def __matmul__(self, other: 'Maxel') -> 'Maxel':
#        """
#        Maxel Multiplication (Generalized Matrix Product over a semiring/field).
#        This simulates the core path-finding or transition operation.
#        """
#        if not isinstance(other, Maxel):
#            raise TypeError("Can only multiply Maxel by another Maxel.")
#        
#        # In this simple case (counting semiring), it's standard matrix multiplication
#        result_data = {}
#        for i_node in self.nodes:
#            row_counter = Counter()
#            for k_node, val_ik in self.data.get(i_node, Counter()).items():
#                if k_node in other.data:
#                    for j_node, val_kj in other.data[k_node].items():
#                        # The operation is: sum (A[i,k] * B[k,j])
#                        # This works for counting (N) and real (R) fields.
#                        row_counter[j_node] += val_ik * val_kj
#            if row_counter:
#                result_data[i_node] = row_counter
#        
#        return Maxel(result_data, self.support | other.support)
#
#    def restrict(self, J: Set[int]) -> 'Maxel':
#        """
#        Maxel Restriction (Filtering / Applying the e_J Vexel mask).
#        Filters the Maxel to keep only edges (i, j) where i, j are in J.
#        """
#        new_data = {}
#        for i, counter in self.data.items():
#            if i in J:
#                new_counter = Counter({j: val for j, val in counter.items() if j in J})
#                if new_counter:
#                    new_data[i] = new_counter
#        # If we were doing the OPTIMIZED path, we'd just filter the result, not copy.
#        # But for the UNOPTIMIZED path, we create a new Maxel over the restricted support.
#        return Maxel(new_data, J)
#
#    def get_value(self, i: int, j: int) -> int:
#        return self.data.get(i, Counter()).get(j, 0)
#        
#    def __eq__(self, other: 'Maxel') -> bool:
#        """Simple check for demonstration purposes."""
##        if not isinstance(other, Maxel): return False
#        
#        # Check that non-zero entries are identical
##        self_entries = {(i, j, v) for i, c in self.data.items() for j, v in c.items()}
##        other_entries = {(i, j, v) for i, c in other.data.items() for j, v in c.items()}
#        
##        return self_entries == other_entries
#
#        if not isinstance(other, Maxel): return False
#    
#        # Extract all (i, j, value) tuples where value is non-zero
#        self_entries = {(i, j, v) 
#                        for i, c in self.data.items() 
#                        for j, v in c.items() if v != 0}
#                    
#        other_entries = {(i, j, v) 
#                         for i, c in other.data.items() 
#                         for j, v in other.data.get(i, Counter()).items() if v != 0}
#    
#        # The comparison should succeed if the set of non-zero (coordinate, value) tuples matches.
#        return self_entries == other_entries
#        

def draw_maxel_graph(m: Maxel, J: Set[int], title: str = "Graph Topology"):
    """
    Converts a Maxel object to a NetworkX graph and draws it, 
    highlighting the restricted subgraph J.
    """
    G = nx.DiGraph()
    
    # 1. Build the graph structure from Maxel data
    for u, counter in m.data.items():
        for v, weight in counter.items():
            if weight != 0:
                G.add_edge(u, v, weight=weight)
                
    # 2. Define node colors and positions
    pos = nx.spring_layout(G, seed=42) 
    
    node_color_map = []
    # Nodes in J (the restricted subgraph) are highlighted.
    for node in G.nodes():
        if node in J:
            node_color_map.append('skyblue') # Highlight color
        else:
            node_color_map.append('lightgray') # Background color
    
    # 3. Drawing using Matplotlib
    fig, ax = plt.subplots(figsize=(7, 5))
    
    nx.draw_networkx_nodes(G, pos, 
                           node_color=node_color_map, 
                           node_size=1000, 
                           ax=ax)
                           
    nx.draw_networkx_edges(G, pos, 
                           edge_color='gray', 
                           arrows=True, 
                           arrowsize=20, 
                           ax=ax)
                           
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    
    # Add edge weights (if applicable)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', ax=ax)

    ax.set_title(title, fontsize=14)
    ax.axis('off')
    
    # Return the Matplotlib figure object for Streamlit to render
    return fig

# --- MVP Functions ---

def algebraic_optimizer_mvp(m: Maxel, J: Set[int], k: int) -> Tuple[Maxel, Maxel, int, int]:
    """
    Runs both the unoptimized and optimized plans for MVP 1.
    Returns: (unoptimized_result, optimized_result, unoptimized_cost, optimized_cost)
    """
    
    # 1. The UNOPTIMIZED (Traditional/Literal) Sequence: (m.restrict(J))^k
    m_J_copy = m.restrict(J)
    m_unoptimized_result = m_J_copy
    for _ in range(1, k):
        m_unoptimized_result = m_unoptimized_result @ m_J_copy
    
    # Simulate Unoptimized Cost: O(|J|^2 * k)
    J_size = len(J)
    cost_unoptimized = (J_size ** 2) * k 

    # 2. The OPTIMIZED (MVGC) Sequence: m^k.restrict(J)
    m_full_power = m
    for _ in range(1, k):
        m_full_power = m_full_power @ m
    
    m_optimized_result = m_full_power.restrict(J)
    
    # Simulate Optimized Cost: O(|E| * k + |J|^2)
    # E is the number of edges in the full graph (size of m.data)
    E_size = sum(len(c) for c in m.data.values())
    cost_optimized = (E_size * k) + (J_size ** 2) 

    return m_unoptimized_result, m_optimized_result, cost_unoptimized, cost_optimized

def rank_reduction_mvp() -> Tuple[float, float]:
    """
    Solves the flow network system M*x = b over R.
    Returns: (x_AB, x_AC)
    """
    # Maxel M (Coefficient Matrix)
    M = np.array([
        [1, 1],   # v1: x_AB + x_AC = 5
        [1, -3]  # v2: x_AB - 3*x_AC = 0
    ])
    # RHS Vexel (Vector)
    b = np.array([5, 0])
    
    # Algebraic Inversion (Rank Reduction)
    solution = np.linalg.solve(M, b)
    
    return solution[0], solution[1] # x_AB, x_AC

def recurrence_prediction_mvp(k: int) -> int:
    """
    Predicts the k-th element of the linear recurrence L_t = L_{t-1} + L_{t-2}
    using Matrix Exponentiation (O(log k)).
    Returns: L_k
    """
    if k <= 1: return k
    
    # Initial Vexel (State Vector): [L_1, L_0] = [1, 0]
    initial_state = np.array([1, 0])
    
    # Characteristic Maxel (Transition Operator M_q):
    M_q = np.array([
        [1, 1],
        [1, 0]
    ])
    
    # Compute M_q raised to the power (k-1) using O(log k) exponentiation
    M_q_power_k_minus_1 = np.linalg.matrix_power(M_q, k - 1)
    
    # Final Vexel (State Vector) is M_q^{k-1} * initial_state
    final_state = M_q_power_k_minus_1 @ initial_state
    
    L_k = int(final_state[0])
    return L_k

def simulate_O_N_time(k: int) -> float:
    """Simulates O(k) linear time for comparison."""
    # Base O(k) time, scaled for visualization
    return k * 0.05 

def simulate_O_logN_time(k: int) -> float:
    """Simulates O(log k) logarithmic time for comparison."""
    # Base O(log k) time, scaled and offset for visualization
    return np.log2(k) * 0.5 + 1.0 if k > 1 else 1.0

# New function to add to maxel_core.py

#import networkx as nx
#import matplotlib.pyplot as plt

def draw_maxel_graph(m: Maxel, highlight_nodes: Set[int] = None, title: str = "Graph Structure (Maxel M)"):
    """
    Converts a Maxel object to a NetworkX graph and draws it using Matplotlib.
    
    Args:
        m (Maxel): The Maxel object to draw.
        highlight_nodes (Set[int]): Nodes to highlight (e.g., the restricted subgraph J).
        title (str): Title for the plot.
    """
    G = nx.DiGraph()
    
    # Add nodes
    G.add_nodes_from(m.support)
    
    # Add weighted edges
    for u, counter in m.data.items():
        for v, weight in counter.items():
            if weight != 0:
                G.add_edge(u, v, weight=weight)
                
    # --- Drawing ---
    fig, ax = plt.subplots(figsize=(8, 5))
    pos = nx.spring_layout(G, seed=42) # Layout for consistency
    
    # Determine colors
    node_color_map = []
    if highlight_nodes:
        # Highlight nodes in J differently
        for node in G.nodes():
            if node in highlight_nodes:
                node_color_map.append('skyblue')
            else:
                node_color_map.append('lightgray')
    else:
        node_color_map = ['lightgray'] * len(G.nodes())
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_color_map, node_size=1500, ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='black', ax=ax)
    
    # Add edge weights if they are not all 1
    edge_labels = nx.get_edge_attributes(G, 'weight')
    if any(w != 1 for w in edge_labels.values()):
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', ax=ax)

    ax.set_title(title)
    ax.axis('off')
    
    return fig


# --- Test Case Setup ---
# Simple graph: 1->2, 2->3, 3->4, 4->1.
m_data_test = {
    1: Counter({2: 1}),
    2: Counter({3: 1}),
    3: Counter({4: 1}),
    4: Counter({1: 1}),
}
M_TEST = Maxel(m_data_test)

