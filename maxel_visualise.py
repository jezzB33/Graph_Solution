import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from typing import Dict, Set

# --- Maxel Class (Complete Definition) ---
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
        """Performs Maxel Addition (Multiset Union of edges)."""
        new_data = {}
        all_nodes = self.support.union(other.support)
        for i in all_nodes:
            v1 = self.data.get(i, Counter())
            v2 = other.data.get(i, Counter())
            if v1 or v2:
                new_data[i] = v1 + v2
        return Maxel(new_data)

    def restrict(self, j: Set[int]) -> 'Maxel':
        """Calculates the induced subgraph m' = e_J m e_J."""
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
        """Performs Maxel Multiplication (Walk Counting)."""
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

# --- Graph Visualization Function ---
def draw_maxel_graph(ax, maxel_obj: Maxel, title: str, highlight_edges: Dict[tuple, str] = None):
    G = nx.MultiDiGraph()
    all_nodes = list(maxel_obj.support)
    G.add_nodes_from(all_nodes)

    for source, vexel in maxel_obj.data.items():
        for target, count in vexel.items():
            G.add_edge(source, target, weight=count, label=f"x{count}")
    
    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=1000)

    # Prepare edge labels and colors
    edge_labels = {}
    edge_list = []
    edge_colors = []
    for u, v, data in G.edges(data=True):
        edge_list.append((u, v))
        edge_labels[(u, v)] = data['label']
        # Default color is gray, highlight if specified
        edge_colors.append(highlight_edges.get((u, v), 'gray') if highlight_edges else 'gray')

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, arrowstyle='->', arrowsize=20, width=2)
    
    # Draw edge labels (weights)
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_color='red',
                                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Draw node labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_color='black')

    ax.set_title(title, fontsize=14)
    ax.axis('off')

# --- Main Visualization Script ---
if __name__ == "__main__":
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Maxel Framework: Core Algebraic Operations (Multiset Graph)', fontsize=18)
    axs = axs.flatten()

    # --- Setup Example Maxels ---
    m_base = Maxel({
        1: Counter({2: 2, 4: 1}),
        2: Counter({3: 1}),
        3: Counter({1: 1}),
        4: Counter({5: 3})
    })

    m_add = Maxel({
        1: Counter({2: 1, 3: 1}),
        3: Counter({2: 1}),
        5: Counter({1: 1})
    })

    m_mult = Maxel({
        2: Counter({6: 1}),
        3: Counter({6: 2})
    })
    
    # --- 1. Maxel Addition (m_base + m_add) ---
    m_sum = m_base + m_add
    
    # Highlight edges that changed/were added
    highlights = {(1, 2): 'blue', (1, 3): 'blue', (3, 2): 'blue', (5, 1): 'blue'} 
    draw_maxel_graph(axs[0], m_sum, "1. Addition (m_base + m_add)", highlights)
    axs[0].text(0.5, 0.95, "Edge counts are summed (e.g., 1->2 is x3). New nodes (5) included.", 
                transform=axs[0].transAxes, fontsize=10, ha='center', va='top')

    # --- 2. Maxel Restriction (m_base restricted to J={1, 2, 3}) ---
    J_restrict = {1, 2, 3}
    m_restricted = m_base.restrict(J_restrict)
    
    # Highlight nodes that were deleted (4, 5) and the deleted edge (4->5)
    draw_maxel_graph(axs[1], m_restricted, f"2. Restriction (e_J m e_J) where J={J_restrict}")
    axs[1].text(0.5, 0.95, "Nodes outside J (4, 5) and their edges are removed.", 
                transform=axs[1].transAxes, fontsize=10, ha='center', va='top')

    # --- 3. Maxel Multiplication (m_base @ m_mult) ---
    m_product = m_base @ m_mult
    
    # Highlight the resulting edges (length-2 walks)
    highlights_prod = {(1, 6): 'purple', (2, 6): 'purple', (3, 6): 'purple'}
    draw_maxel_graph(axs[2], m_product, "3. Multiplication (m_base @ m_mult)")
    axs[2].text(0.5, 0.95, "Counts walks of length 2 (i -> k -> j). Resulting edges are new.", 
                transform=axs[2].transAxes, fontsize=10, ha='center', va='top')

    # --- 4. Empty Panel for Summary/Explanation ---
    axs[3].set_title("Maxel: Sparse, Multiset Graph Algebra", fontsize=14)
    axs[3].text(0.5, 0.7, "**The Core Structure:**", fontsize=12, ha='center')
    axs[3].text(0.5, 0.6, "Maxel = Dict {Source Node $\to$ Counter {Target Node: Count}}", fontsize=10, ha='center')
    axs[3].text(0.5, 0.45, "**Efficiency Rationale:**", fontsize=12, ha='center')
    axs[3].text(0.5, 0.35, "Leverages Python's Counter for fast multiset math.", fontsize=10, ha='center')
    axs[3].text(0.5, 0.25, "Avoids $N^2$ storage/processing inherent in dense matrices (sparsity).", fontsize=10, ha='center')
    axs[3].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
#    plt.show()    # Save the figure instead of trying to show it in a GUI window
    plt.savefig("maxel_visualization.png", bbox_inches='tight')
    plt.close(fig) # Close the figure to free up memory
#

# 
