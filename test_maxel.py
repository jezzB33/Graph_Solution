import unittest
from collections import Counter
from typing import Set, Dict

# Assuming the Maxel class from the previous response is available
# (You would typically put the Maxel class in a separate file like maxel_framework.py)

# Placeholder for the Maxel class structure for completeness:
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

    # --- Implementation for MVP Operation 2: Maxel Restriction ---
    def restrict(self, J: Set[int]) -> 'Maxel':
        """
        Calculates the induced subgraph m' = e_J m e_J.
        This is a metadata/filtering operation.
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
                
                # Only include the Vexel if it's not empty after restriction
                if new_vexel:
                    new_data[i] = new_vexel
                    
        return Maxel(new_data, J)

    def __eq__(self, other: object) -> bool:
        """Helper for comparison in tests."""
        if not isinstance(other, Maxel):
            return NotImplemented
        return self.data == other.data

    def __repr__(self):
        return f"Maxel(data={self.data}, support={self.support})"

# --- UNIT TESTS ---

class TestMaxelRestriction(unittest.TestCase):
    
    def setUp(self):
        """Setup a common Maxel for all tests based on the example."""
        # Maxel m:
        # 1 -> 2 (count 2)
        # 1 -> 3 (count 1)
        # 2 -> 3 (count 1)
        self.m_original = Maxel({
            1: Counter({2: 2, 3: 1}),
            2: Counter({3: 1})
        })
    
    # --- T7: Full Graph Restriction ---
    def test_full_graph_restriction(self):
        """Restricting m to its full support set J should return an identical Maxel."""
        J_full = self.m_original.support
        m_sub = self.m_original.restrict(J_full)
        
        # We expect the result to be functionally equal to the original
        self.assertEqual(m_sub, self.m_original, "Restriction to full support should be identity.")

    # --- T8 & T9: Edge and Vertex Deletion (The User's Core Test Case) ---
    def test_induced_subgraph_deletion(self):
        """Restricting to J={1, 2} should delete all edges connected to node 3."""
        J = {1, 2}
        m_sub = self.m_original.restrict(J)
        
        # Expected Maxel: only 1 -> 2 (count 2) remains.
        expected_data = {
            1: Counter({2: 2})
        }
        
        self.assertEqual(m_sub.data, expected_data, 
                         "Restriction failed to delete edges 1->3 and 2->3.")
        
        # Also check the new support set
        self.assertEqual(m_sub.support, J, 
                         "Restriction failed to correctly set the resulting support J.")

    # --- T10: Empty Restriction ---
    def test_empty_restriction(self):
        """Restricting m to an empty set J should result in an empty Maxel."""
        J_empty: Set[int] = set()
        m_sub = self.m_original.restrict(J_empty)
        
        self.assertEqual(m_sub.data, {}, "Restriction to empty set should yield empty data.")
        self.assertEqual(m_sub.support, J_empty, "Restriction to empty set should yield empty support.")

    # --- T9 (Specific): Deletion of a source node that is not the target of any edge ---
    def test_source_node_deletion(self):
        """Restricting J={2, 3} should delete source node 1 and all its edges."""
        J = {2, 3}
        m_sub = self.m_original.restrict(J)

        # Expected Maxel: only 2 -> 3 (count 1) remains.
        # Edge 1->3 is deleted because source 1 is outside J.
        expected_data = {
            2: Counter({3: 1})
        }
        
        self.assertEqual(m_sub.data, expected_data, 
                         "Restriction failed to delete source node 1 and its edges.")
        self.assertEqual(m_sub.support, J, "Support set should be J={2, 3}.")

# --- Execution ---

if __name__ == '__main__':
    # Run the tests
    print("Running Maxel Restriction Tests...")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
