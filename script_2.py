# Let me create a more detailed mathematical example with LaTeX formulations
print("=== COMPREHENSIVE PAGERANK MATHEMATICAL EXAMPLE ===")
print()

# Create a simple but illustrative example
print("EXAMPLE: 3-Page Web Network")
print("="*40)

# Define a 3-page network for simplicity
print("Network Structure:")
print("• Page A links to Page B")
print("• Page B links to Page C") 
print("• Page C links to Page A")
print()

# Show the mathematical construction step by step
import numpy as np

# Adjacency matrix
A_3 = np.array([
    [0, 0, 1],  # A receives from C
    [1, 0, 0],  # B receives from A  
    [0, 1, 0]   # C receives from B
])

print("1. ADJACENCY MATRIX A:")
print("   A[i,j] = 1 if page j links to page i")
print("   A =", A_3.tolist())
print()

# Hyperlink matrix (column stochastic)
H_3 = A_3.astype(float)
print("2. HYPERLINK MATRIX H:")
print("   Since each page has exactly 1 outgoing link:")
print("   H =", H_3.tolist())
print("   (Each column sums to 1)")
print()

# Google matrix with damping
d = 0.85
n = 3
e_3 = np.ones((n, n)) / n
G_3 = d * H_3 + (1 - d) * e_3

print("3. GOOGLE MATRIX G:")
print(f"   G = d·H + (1-d)·E  where d={d}")
print("   G =")
for i, row in enumerate(G_3):
    print(f"       [{row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f}]")
print()

# Show the iterative formula
print("4. ITERATIVE PAGERANK FORMULA:")
print("   r^(k+1) = G · r^(k)")
print("   where r^(k) is the PageRank vector at iteration k")
print()

# Power iteration
r = np.ones(n) / n
print("5. POWER ITERATION STEPS:")
print(f"   Initial: r^(0) = [{r[0]:.4f}, {r[1]:.4f}, {r[2]:.4f}]")

for k in range(8):
    r_new = G_3 @ r
    print(f"   Step {k+1:2d}: r^({k+1}) = [{r_new[0]:.4f}, {r_new[1]:.4f}, {r_new[2]:.4f}]")
    r = r_new

print()

# Eigenvector solution
eigenvals, eigenvecs = np.linalg.eig(G_3)
idx = np.argmax(np.real(eigenvals))
principal_eigenvec = np.real(eigenvecs[:, idx])
principal_eigenvec = principal_eigenvec / np.sum(principal_eigenvec)

print("6. EIGENVECTOR SOLUTION:")
print(f"   Eigenvalues: {np.real(eigenvals)}")
print(f"   Principal eigenvector: [{principal_eigenvec[0]:.4f}, {principal_eigenvec[1]:.4f}, {principal_eigenvec[2]:.4f}]")
print(f"   Power method result:   [{r[0]:.4f}, {r[1]:.4f}, {r[2]:.4f}]")
print()

# Mathematical insights
print("7. KEY MATHEMATICAL INSIGHTS:")
print("   • PageRank vector r satisfies: r = G·r (eigenvector equation)")
print("   • G is column stochastic: each column sums to 1")
print("   • G is primitive: irreducible and aperiodic")  
print("   • By Perron-Frobenius theorem: unique solution exists")
print("   • Power method converges to principal eigenvector")
print("   • Convergence rate depends on second-largest eigenvalue")
print()

# Show the mathematical formulation in standard notation
print("8. STANDARD MATHEMATICAL NOTATION:")
print("   For page i:")
print("   PR(i) = (1-d)/n + d · Σ[PR(j)/L(j)]")
print("   where:")
print("   • PR(i) = PageRank of page i")
print("   • d = damping factor (typically 0.85)")
print("   • n = total number of pages")
print("   • j ranges over all pages linking to i")
print("   • L(j) = number of outbound links from page j")
print()

# Matrix form
print("9. MATRIX FORMULATION:")
print("   r = (1-d)/n · e + d · H · r")
print("   where:")
print("   • r = PageRank vector [PR(1), PR(2), ..., PR(n)]ᵀ")
print("   • e = vector of ones [1, 1, ..., 1]ᵀ")
print("   • H = hyperlink matrix (column stochastic)")
print()

print("10. COMPUTATIONAL COMPLEXITY:")
print("    • Matrix size: n × n where n ≈ 10¹² (web pages)")
print("    • H is sparse: ~10 non-zeros per column")
print("    • Matrix-vector multiply: O(10n) operations")
print("    • Iterations needed: ~50-100 for convergence")
print("    • Total: O(10n) per iteration")
print()

print("="*60)
print("This example demonstrates all key mathematical concepts:")
print("• Matrix construction and properties")
print("• Eigenvalue/eigenvector relationship") 
print("• Power iteration method")
print("• Convergence guarantees")
print("• Computational considerations")
print("="*60)