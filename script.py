import numpy as np
import pandas as pd

# Let me create a comprehensive PageRank example with detailed mathematical calculations
# This will serve as the foundation for a complete mathematical demonstration

# First, let's create a small web graph example (4 pages)
print("=== PAGERANK MATHEMATICAL DEMONSTRATION ===")
print("\n1. Setting up a 4-page web graph example")

# Define the web graph connections
# Page A links to: B, C
# Page B links to: C  
# Page C links to: A, D
# Page D links to: A

print("Web Graph Structure:")
print("Page A → Pages B, C")
print("Page B → Page C")
print("Page C → Pages A, D") 
print("Page D → Page A")

# Create the adjacency matrix A
# A[i,j] = 1 if page j links to page i, 0 otherwise
A = np.array([
    [0, 0, 1, 1],  # Page A receives links from C, D
    [1, 0, 0, 0],  # Page B receives links from A
    [1, 1, 0, 0],  # Page C receives links from A, B
    [0, 0, 1, 0]   # Page D receives links from C
])

print("\n2. Adjacency Matrix A:")
print("(A[i,j] = 1 if page j links to page i)")
print(A)

# Create the transition matrix (column stochastic)
# Normalize each column by the number of outgoing links
outgoing_links = np.sum(A, axis=0)
print(f"\nOutgoing links per page: {outgoing_links}")

# Handle pages with no outgoing links (dangling nodes)
# In this example, all pages have outgoing links
H = A / outgoing_links
print("\n3. Hyperlink Matrix H (column stochastic):")
print("H[i,j] = probability of moving from page j to page i")
print(H)
print(f"Column sums (should be 1): {np.sum(H, axis=0)}")

# Create the Google Matrix with damping factor
d = 0.85
n = 4
e = np.ones((n, n)) / n

G = d * H + (1 - d) * e
print(f"\n4. Google Matrix G with damping factor d={d}:")
print("G = d*H + (1-d)*E, where E is matrix of 1/n")
print(G)
print(f"Column sums: {np.sum(G, axis=0)}")

# Initial PageRank vector (uniform distribution)
r0 = np.ones(n) / n
print(f"\n5. Initial PageRank vector r₀:")
print(f"r₀ = {r0} (uniform distribution)")

# Power iteration method
print("\n6. Power Iteration Method:")
print("rₖ₊₁ = G * rₖ")

iterations = []
r = r0.copy()

for k in range(15):
    r_new = G @ r
    iterations.append({
        'iteration': k,
        'page_A': r[0],
        'page_B': r[1], 
        'page_C': r[2],
        'page_D': r[3],
        'sum': np.sum(r)
    })
    
    if k < 10:  # Print first 10 iterations
        print(f"Iteration {k}: [{r[0]:.4f}, {r[1]:.4f}, {r[2]:.4f}, {r[3]:.4f}] sum={np.sum(r):.4f}")
    
    r = r_new

print("...")
print(f"Iteration {k}: [{r[0]:.4f}, {r[1]:.4f}, {r[2]:.4f}, {r[3]:.4f}] sum={np.sum(r):.4f}")

# Create DataFrame for iterations
df_iterations = pd.DataFrame(iterations)
print(f"\nFinal PageRank scores:")
print(f"Page A: {r[0]:.6f}")
print(f"Page B: {r[1]:.6f}")
print(f"Page C: {r[2]:.6f}")
print(f"Page D: {r[3]:.6f}")

# Verify this is an eigenvector
eigenvector_check = G @ r
print(f"\n7. Eigenvector Verification:")
print(f"G * r = {eigenvector_check}")
print(f"r =     {r}")
print(f"Difference: {np.linalg.norm(eigenvector_check - r):.10f}")

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(G)
print(f"\n8. Eigenvalue Analysis:")
print(f"Eigenvalues: {eigenvalues}")

# Find the eigenvector corresponding to eigenvalue 1
idx = np.argmax(np.real(eigenvalues))
principal_eigenvector = np.real(eigenvectors[:, idx])
# Normalize to sum to 1
principal_eigenvector = principal_eigenvector / np.sum(principal_eigenvector)

print(f"Principal eigenvector (λ=1): {principal_eigenvector}")
print(f"Power method result:        {r}")
print(f"Match: {np.allclose(principal_eigenvector, r)}")

# Save iteration data for visualization
df_iterations.to_csv('pagerank_iterations.csv', index=False)
print(f"\nSaved {len(iterations)} iterations to 'pagerank_iterations.csv'")

# Create damping factor analysis
print("\n9. Damping Factor Analysis:")
damping_factors = [0.5, 0.7, 0.85, 0.9, 0.95]
damping_results = []

for d_val in damping_factors:
    G_temp = d_val * H + (1 - d_val) * e
    
    # Power iteration
    r_temp = np.ones(n) / n
    for _ in range(50):  # More iterations for convergence
        r_temp = G_temp @ r_temp
    
    damping_results.append({
        'damping_factor': d_val,
        'page_A': r_temp[0],
        'page_B': r_temp[1],
        'page_C': r_temp[2], 
        'page_D': r_temp[3]
    })
    
    print(f"d={d_val}: A={r_temp[0]:.4f}, B={r_temp[1]:.4f}, C={r_temp[2]:.4f}, D={r_temp[3]:.4f}")

df_damping = pd.DataFrame(damping_results)
df_damping.to_csv('damping_factor_comparison.csv', index=False)
print(f"\nSaved damping factor analysis to 'damping_factor_comparison.csv'")

print("\n=== MATHEMATICAL INSIGHTS ===")
print("1. PageRank is the stationary distribution of a Markov chain")
print("2. It's computed as the principal eigenvector of the Google matrix")
print("3. The power method converges to this eigenvector")
print("4. Damping factor (0.85) balances authority flow and random jumps")
print("5. Matrix must be stochastic, irreducible, and aperiodic for convergence")