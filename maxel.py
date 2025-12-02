from collections import Counter
from typing import Set, Dict, Any

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
