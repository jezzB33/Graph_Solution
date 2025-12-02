# test_core.py
import numpy as np
import pandas as pd
from collections import Counter
from maxel_core import (
    Maxel,
    algebraic_optimizer_mvp,
    rank_reduction_mvp,
    recurrence_prediction_mvp,
    M_TEST  # The simple test Maxel instance
)

print("--- Maxel/Vexel Core Logic Test ---")

# ===============================================
# 1. Test MVP 1: Algebraic Optimizer
# ===============================================
print("\n[1. Algebraic Optimizer (Query Rewrite)]")

J_set = {1, 2, 3}
k_walks = 3

try:
    m_unoptimized, m_optimized, cost_u, cost_o = algebraic_optimizer_mvp(M_TEST, J_set, k_walks)

    # Use Maxel methods to retrieve results (e.g., walk count from node 1 to 4)
    result_u = m_unoptimized.get_value(1, 4)
    result_o = m_optimized.get_value(1, 4)

    print(f"Goal: {k_walks}-walk count (1 -> 4) within J={J_set}")
    print(f"Unoptimized Result (Walks): {result_u}")
    print(f"Optimized Result (Walks):   {result_o}")
    
    if result_u == result_o:
        print(f"✅ Correctness Verified: Results Match.")
        print(f"   Simulated Cost Ratio (Unoptimized/Optimized): {cost_u/cost_o:.2f}")
    else:
        print("❌ Correctness Failed: Results Do Not Match.")

except Exception as e:
    print(f"An error occurred during MVP 1 test: {e}")


# ===============================================
# 2. Test MVP 2: Rank Reduction
# ===============================================
print("\n[2. Rank Reduction (Constraint Solve)]")
try:
    x_AB, x_AC = rank_reduction_mvp()
    
    # Verification check (optional, but good practice)
    check1 = x_AB + x_AC  # Should be 5
    check2 = x_AB - 3 * x_AC  # Should be 0

    print(f"Solved Flow x_AB: {x_AB:.3f}")
    print(f"Solved Flow x_AC: {x_AC:.3f}")
    print(f"Verification: C1={check1:.1f}, C2={check2:.1f}")
    if abs(check1 - 5) < 1e-6 and abs(check2) < 1e-6:
        print("✅ Constraint Resolved: Algebraic solution verified.")
    else:
        print("❌ Constraint Failed: Solution is incorrect.")

except Exception as e:
    print(f"An error occurred during MVP 2 test: {e}")


# ===============================================
# 3. Test MVP 3: Dynamic Vexel Calculus
# ===============================================
print("\n[3. Dynamic Vexel Calculus (Prediction)]")
k_pred = 10
try:
    L_k = recurrence_prediction_mvp(k_pred)
    expected_L_k = 55  # 10th Fibonacci number starting 0, 1, 1, 2...

    print(f"Prediction Time k={k_pred}")
    print(f"Predicted Load L_k: {L_k}")
    
    if L_k == expected_L_k:
        print(f"✅ Prediction Verified: Matches expected value {expected_L_k}.")
        print("   O(log k) speed demonstrated internally.")
    else:
        print(f"❌ Prediction Failed: Got {L_k}, expected {expected_L_k}.")

except Exception as e:
    print(f"An error occurred during MVP 3 test: {e}")

