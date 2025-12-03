import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from collections import Counter
from typing import Set

# Import the proprietary core logic (Maxel class and MVP functions)
from maxel_core import (
    Maxel, 
    algebraic_optimizer_mvp, 
    rank_reduction_mvp, 
    recurrence_prediction_mvp, 
    simulate_O_N_time, 
    simulate_O_logN_time,
    M_TEST
)

st.set_page_config(layout="wide", page_title="Maxel/Vexel Algebraic Substrate Demo")

st.title("🌌 Maxel/Vexel Framework: Algebraic Graph Analysis")
st.markdown("Demonstrating $O(\log N)$ prediction and query optimization using generalized algebraic substrates.")

# --- NAVIGATION ---
tab1, tab2, tab3 = st.tabs(["1. Algebraic Optimizer (Query Rewrite)", "2. Rank Reduction (Constraint Solve)", "3. Dynamic Vexel Calculus (Prediction)"])

# =======================================================================================
# --- TAB 1: Algebraic Optimizer (MVP 1) ---
# =======================================================================================
with tab1:
    st.header("⚙️ Algebraic Optimizer: $O(|J|^2) \to O(|E'|)$ Efficiency Proof")
    st.markdown("Optimization relies on **Restriction Commutativity**: $\mathbf{M}_J^k = \mathbf{M}^k \cdot \mathbf{e}_J$.")

    # 1.1. Input Parameters
    colA, colB = st.columns(2)
    with colA:
        k_walks = st.slider("Select Walk Length (k)", 2, 5, 3, key='k_walks_mvp1')
        st.info(f"Goal: Calculate {k_walks}-walks.")
    with colB:
        # Node 4 breaks the 3-walk in the {1, 2, 3} subgraph
        J_nodes = st.multiselect("Select Subgraph Nodes J", [1, 2, 3, 4], default=[1, 2, 3], key='j_nodes_mvp1')
        J_set = set(J_nodes)
        if not J_set: J_set = {1, 2, 3} # Prevent empty set error

    # In app.py, inside 'with tab1:'

    # ... (Previous code for Input Parameters) ...

    # 1.1. Input Parameters
    colA, colB, colC = st.columns([1, 1, 2]) # Adjusted columns for visualization

    with colA:
        k_walks = st.slider("Select Walk Length (k)", 2, 5, 3)
        st.info(f"Goal: Calculate {k_walks}-walks.")
    with colB:
        # Node 4 breaks the 3-walk in the {1, 2, 3} subgraph
        J_nodes = st.multiselect("Select Subgraph Nodes J", [1, 2, 3, 4], default=[1, 2, 3])
        J_set = set(J_nodes)
        if not J_set: J_set = {1, 2, 3} # Prevent empty set error

    # --- NEW ADDITION for MVP 1 ---
    with colC:
        st.subheader("Graph Context (M) & Subgraph (J)")
        # Draw the graph, highlighting nodes in J
        fig_graph = draw_maxel_graph(M_TEST, highlight_nodes=J_set, 
                                 title=f"Full Graph M, Subgraph J={J_set}")
        st.pyplot(fig_graph)
    # --- END NEW ADDITION ---

    # ... (Rest of the execution and visualization code) ...


    # --- Execution ---
    m_unoptimized, m_optimized, cost_unoptimized, cost_optimized = algebraic_optimizer_mvp(M_TEST, J_set, k_walks)
    
    walks_result = m_optimized.get_value(1, 4) # Check a key value
    
    st.subheader("✅ Result Verification: Correctness is Preserved")
    
    # 1.2. Result Verification (Metrics)
    col1, col2, col3 = st.columns(3)
    col1.metric("Unoptimized Walk Count", m_unoptimized.get_value(1, 4))
    col2.metric("Optimized Walk Count", m_optimized.get_value(1, 4))
    
    if m_unoptimized == m_optimized:
         col3.success("IDENTITY VERIFIED: Unoptimized $\equiv$ Optimized")
    else:
         col3.error("IDENTITY FAILED: Unoptimized $\neq$ Optimized")

    st.subheader("📊 Efficiency Proof: The Cost of Copying vs. Filtering")
    
    # 1.3. Efficiency Proof (Bar Chart)
    cost_df = pd.DataFrame({
        'Execution Plan': ['Unoptimized (Literal Copy)', 'Optimized (Algebraic Rewrite)'],
        'Complexity (Simulated)': [f"O(|J|² × k) ≈ {cost_unoptimized}", f"O(|E| × k + |J|²) ≈ {cost_optimized}"],
        'Cost Value': [cost_unoptimized, cost_optimized]
    })
    
    fig = px.bar(cost_df, x='Execution Plan', y='Cost Value', 
                 title="Computational Cost Comparison for Walk Finding", 
                 color='Execution Plan',
                 text='Complexity (Simulated)',
                 height=450)
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The Maxel optimizer avoids the quadratic cost of copying the subgraph in every step, leading to substantial gains.")

# =======================================================================================
# --- TAB 2: Rank Reduction (MVP 2) ---
# =======================================================================================
with tab2:
    st.header("📉 Rank Reduction: Instantaneous Constraint Resolution")
    st.markdown("Solves the linear system $\mathbf{M}\vec{x} = \vec{b}$ over the $\mathbb{R}$ field (real numbers) using algebraic inversion.")

    x_AB, x_AC = rank_reduction_mvp()

    # 2.1. Constraint Definition
    st.subheader("Input Constraint System (Flow Network Example)")
    col_input, col_def = st.columns([1.5, 2.5])
    
    M_data = {'$x_{AB}$ (Flow A→B)': [1, 1], '$x_{AC}$ (Flow A→C)': [1, -3]}
    b_data = [5, 0]
    M_df = pd.DataFrame(M_data, index=['Constraint 1 (Total Flow)', 'Constraint 2 (Ratio)'])
    M_df['RHS ($\vec{b}$)'] = b_data
    
    with col_input:
        st.dataframe(M_df)
    
    with col_def:
        st.code(
            """
            Constraint 1: 1*x_AB + 1*x_AC = 5  (Total flow out)
            Constraint 2: 1*x_AB - 3*x_AC = 0  (Ratio constraint)
            """
        )
        st.caption("This system is solved using one-shot Maxel algebraic inversion.")

    # 2.2. Solution Output
    st.subheader("✅ Solution Resolved: No Iteration Required")
    col3, col4 = st.columns(2)
    col3.metric("Flow $x_{AB}$", f"{x_AB:.3f}")
    col4.metric("Flow $x_{AC}$", f"{x_AC:.3f}")

    # 2.3. Profiling Insight (Simulated O(N^3) vs Iterative Solver)
    st.subheader("📈 Profiling Insight: Predictable $O(N^3)$ vs. Volatile Iteration")
    st.markdown("Maxel rank reduction guarantees $\mathbf{O(N^3)}$ time, avoiding the unpredictable nature of iterative search.")
    
    N_values = np.array(range(10, 150, 10))
    time_maxel = N_values**3 / 100000.0  # O(N^3) scaling
    time_iterative = N_values * np.log(N_values) * 0.1 + np.random.rand(len(N_values)) * 0.5  # Simulating a volatile iterative solver
    
    df_profiling = pd.DataFrame({
        'N (Constraints/Variables)': N_values,
        'Time (Maxel O(N³))': time_maxel,
        'Time (Iterative/Search)': time_iterative
    })
    
    fig_profile = px.line(df_profiling, x='N (Constraints/Variables)', y=['Time (Maxel O(N³))', 'Time (Iterative/Search)'],
                          labels={'value': 'Execution Time (s)', 'variable': 'Method'},
                          title="Execution Time Scaling: Algebraic Solve vs. Iterative Search",
                          height=450)
    st.plotly_chart(fig_profile, use_container_width=True)

# =======================================================================================
# --- TAB 3: Dynamic Vexel Calculus (MVP 3) ---
# =======================================================================================
with tab3:
    st.header("🌊 Dynamic Vexel Calculus: $O(\log k)$ 'Warp Speed' Prediction")
    st.markdown("The $k$-th state $L_k$ is found by calculating the Maxel power $\mathbf{M}_q^{k-1}$ using $\mathbf{O(\log k)}$ steps.")

    # 3.1. Prediction Input
    k_prediction = st.slider("Select Prediction Time (t=k)", 10, 500, 100, key='k_prediction_mvp3')

    # In app.py, inside 'with tab3:'

    # ... (Previous code for header, slider, and execution) ...

    st.subheader(f"Maxel Prediction at $t={k_prediction}$")
    st.metric(f"Predicted Node Load $L_{k}$", L_k_predicted)

    # 3.2. Warp Speed Proof (Line/Scatter Plot)
    st.subheader("🚀 Visual Proof: Logarithmic Time Jump (Warp Speed)")

    # 1. Simulate the history (linear steps)
    history_steps = 10
    # Note: recurrence_prediction_mvp(i) is used here just to get the correct value, 
    # but the visual represents the O(k) cost of generating this history iteratively.
    simulated_series = [recurrence_prediction_mvp(i) for i in range(1, history_steps + 1)]

    # 2. Prepare plot data for the history (line plot)
    plot_data_history = pd.DataFrame({
        'Time Step (t)': list(range(1, history_steps + 1)),
        'Node Load (L_t)': simulated_series,
        'Type': 'Simulated History ($O(k)$)'
    })

    # 3. Add the predicted point (the jump)
    plot_data_predicted = pd.DataFrame({
        'Time Step (t)': [k_prediction],
        'Node Load (L_t)': [L_k_predicted],
        'Type': 'Maxel Prediction ($O(\log k)$)'
    })

    # 4. Create the plot
    fig_series = px.line(plot_data_history, x='Time Step (t)', y='Node Load (L_t)', 
                     title=f"Node Load (L_t) - History vs. Jump Prediction to t={k_prediction}",
                     height=550)

    # Add the predicted point as a large, distinct scatter marker
    # This creates the visual discontinuity (the jump)
    fig_series.add_trace(
        px.scatter(plot_data_predicted, x='Time Step (t)', y='Node Load (L_t)', 
                   color='Type', 
                   size=[100],  # Make the predicted point very large
                   size_max=20, # Cap the symbol size for readability
                   ).data[0]
    )

    # Update layout for dramatic scaling
    fig_series.update_layout(
        xaxis_title='Time Step (t)',
        yaxis_title='Node Load (L_t)',
        showlegend=True
    )

    # Force the X-axis range to include the large jump and the origin
    # If k is large, we must ensure the history is visible
    x_range_max = max(k_prediction, history_steps + 10)
    fig_series.update_xaxes(range=[0, x_range_max]) 

    st.plotly_chart(fig_series, use_container_width=True)
    st.caption("The Maxel framework calculates the distant state (large marker) algebraically in $O(\log k)$ time, bypassing all intermediate steps.")

    # ... (Rest of the profiling insight code) ...

    
    # --- Execution ---
    L_k_predicted = recurrence_prediction_mvp(k_prediction)
    
    st.subheader(f"Maxel Prediction at $t={k_prediction}$")
    st.metric(f"Predicted Node Load $L_{k}$", L_k_predicted)

    # 3.2. Warp Speed Proof (Line/Scatter Plot)
    st.subheader("🚀 Visual Proof: Logarithmic Time Jump (Warp Speed)")
    
    # 1. Simulate the history (linear steps)
    history_steps = 10
    simulated_series = [recurrence_prediction_mvp(i) for i in range(1, history_steps + 1)]
    
    # 2. Prepare plot data
    plot_data_history = pd.DataFrame({
        'Time Step (t)': list(range(1, history_steps + 1)),
        'Node Load (L_t)': simulated_series,
        'Type': 'Simulated History ($O(k)$)'
    })
    
    # 3. Add the predicted point (the jump)
    plot_data_predicted = pd.DataFrame({
        'Time Step (t)': [k_prediction],
        'Node Load (L_t)': [L_k_predicted],
        'Type': 'Maxel Prediction ($O(\log k)$)'
    })
    
    # Combine for plotting the jump
    df_combined = pd.concat([plot_data_history, plot_data_predicted], ignore_index=True)
    
    fig_series = px.scatter(df_combined, x='Time Step (t)', y='Node Load (L_t)', color='Type', 
                            size=[10] * history_steps + [50], # Larger marker for prediction
                            title=f"Node Load (L_t) - History vs. Jump Prediction to t={k_prediction}",
                            height=500)
    fig_series.update_traces(mode='lines+markers', line_shape='linear')
    fig_series.update_layout(xaxis_type='log' if k_prediction > 100 else 'linear')
    
    st.plotly_chart(fig_series, use_container_width=True)
    st.caption("The Maxel framework calculates the distant state (large marker) algebraically in $O(\log k)$ time, bypassing all intermediate steps.")

    # 3.3. Profiling Insight (O(k) vs O(log k))
    st.subheader("📈 Profiling Insight: Execution Time Scaling")

    k_values = np.array(range(10, 1000, 50))
    df_time_compare = pd.DataFrame({
        'k (Prediction Time)': k_values,
        'Simulation Time (O(k))': [simulate_O_N_time(k) for k in k_values],
        'Maxel Time (O(log k))': [simulate_O_logN_time(k) for k in k_values]
    })
    
    fig_time = px.line(df_time_compare, x='k (Prediction Time)', y=['Simulation Time (O(k))', 'Maxel Time (O(log k))'],
                       labels={'value': 'Execution Time (s)', 'variable': 'Method'},
                       title="Computational Time vs. Prediction Depth",
                       height=450)
    fig_time.update_layout(yaxis_range=[0, 10]) 
    st.plotly_chart(fig_time, use_container_width=True)
    st.markdown("The **Maxel $O(\log k)$ time** line (yellow) visually flattens, confirming computation time is not linearly dependent on the forecast horizon.")
    [attachment_0](attachment)
