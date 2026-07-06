import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch

# 1. Set random seed for reproducibility
np.random.seed(2026)

# 2. Generate High-Fidelity Data (Concentrated region)
n_hifi = 45
m_hifi = np.random.uniform(5.7, 6.3, n_hifi)
re_hifi = np.random.uniform(4.5e6, 7.5e6, n_hifi)
alpha_hifi = np.random.uniform(0.04, 0.09, n_hifi)
hifi_points = np.column_stack((m_hifi, re_hifi, alpha_hifi))

# 3. Calculate the 3D Convex Hull
hull = ConvexHull(hifi_points)

# 4. Generate Low-Fidelity Data (Broader range [5, 7], [1e6, 12e6], [0, 0.15])
n_lofi = 200
m_lofi = np.random.uniform(5, 7, n_lofi)
re_lofi = np.random.uniform(1e6, 12e6, n_lofi)
alpha_lofi = np.random.uniform(0, 0.15, n_lofi)

# 5. Define Query Points (Some explicitly inside, some outside)
# query_m = [6.0, 5.2, 6.8, 5.9, 5.5, 6.7, 6.1, 5.1]
# query_re = [6.0e6, 2.5e6, 10.5e6, 5.5e6, 8.2e6, 3.8e6, 6.8e6, 11e6]
# query_alpha = [0.06, 0.12, 0.02, 0.07, 0.01, 0.13, 0.05, 0.08]

# Query Points set 1: 取自High-Fidelity Data的随机子集
num_queries_hifi = 4
query_indices_hifi = np.random.choice(n_hifi, num_queries_hifi, replace=False)
query_m_hifi = m_hifi[query_indices_hifi]
query_re_hifi = re_hifi[query_indices_hifi]
query_alpha_hifi = alpha_hifi[query_indices_hifi]

# Query Points set 2: 取自Low-Fidelity Data的随机子集
num_queries_lofi = 8
query_indices_lofi = np.random.choice(n_lofi, num_queries_lofi, replace=False)
query_m_lofi = m_lofi[query_indices_lofi]
query_re_lofi = re_lofi[query_indices_lofi]
query_alpha_lofi = alpha_lofi[query_indices_lofi]

# Combine both sets of query points
query_m = np.concatenate((query_m_hifi, query_m_lofi))
query_re = np.concatenate((query_re_hifi, query_re_lofi))
query_alpha = np.concatenate((query_alpha_hifi, query_alpha_lofi))


# 6. Set up the 3D plot
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot Low-Fidelity Data
scatter_lofi = ax.scatter(m_lofi, re_lofi, alpha_lofi, marker='^', facecolors='none', 
                          edgecolors='steelblue', s=50, label='Low-Fidelity Data')

# Plot High-Fidelity Data
scatter_hifi = ax.scatter(m_hifi, re_hifi, alpha_hifi, color='forestgreen', 
                          alpha=0.9, s=50, label='High-Fidelity Data')

# Draw the 3D Convex Hull Faces
# Extract the points for each triangle (simplex) that makes up the hull
hull_faces = [hifi_points[s] for s in hull.simplices]
poly3d = Poly3DCollection(hull_faces, alpha=0.3, facecolor='forestgreen', 
                          edgecolor='black', linewidths=0.5)
ax.add_collection3d(poly3d)

# Plot Query Points
scatter_query = ax.scatter(query_m, query_re, query_alpha, marker='x', color='firebrick', 
                           s=120, linewidths=3, label='Query Points')

# 7. Formatting and Labels
ax.set_title('Data Distributions and Convex Hull in M-Re-α Space', fontsize=16)
ax.set_xlabel('Mach Number (M)', fontsize=12, labelpad=10)
ax.set_ylabel('Reynolds Number (Re)', fontsize=12, labelpad=15)
ax.set_zlabel('Angle of Attack (α)', fontsize=12, labelpad=10)

# Set the axis limits based on your constraints
ax.set_xlim([5, 7])
ax.set_ylim([1e6, 12e6])
ax.set_zlim([0, 0.15])

# Use scientific notation for the Reynolds Number axis
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Create a proxy artist for the Convex Hull legend entry (since Poly3DCollection doesn't auto-generate well)
hull_proxy = Patch(facecolor='forestgreen', edgecolor='black', alpha=0.3, label='Convex Hull of Hi-Fi Data')

# Compile custom legend
ax.legend(handles=[scatter_hifi, scatter_lofi, scatter_query, hull_proxy], 
          fontsize=12, loc='upper right', bbox_to_anchor=(1.05, 1))

# Adjust viewing angle for better depth perception
ax.view_init(elev=20, azim=-55)

plt.tight_layout()
plt.show()