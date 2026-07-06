import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

# 1. Set a new random seed for a different data distribution
np.random.seed(2026)

# 2. Generate High-Fidelity Data (Concentrated range)
n_hifi = 35
m_hifi = np.random.uniform(0.7, 1.15, n_hifi)
re_hifi = m_hifi * 5e6 + np.random.normal(0, 0.5e6, n_hifi) # Correlated trend
hifi_points = np.column_stack((m_hifi, re_hifi))

# 3. Calculate the Convex Hull for the High-Fidelity Data
hull = ConvexHull(hifi_points)

# 4. Generate Low-Fidelity Data (Larger range, sparser)
n_lofi = 120
m_lofi = np.random.uniform(0.2, 1.6, n_lofi)
re_lofi = m_lofi * 5e6 + np.random.normal(0, 1.5e6, n_lofi)
re_lofi = np.clip(re_lofi, 1e6, 12e6) # Keep within realistic bounds

# 5. Define Query Points (Some inside the hull, some outside)
query_m = [0.85, 1.0, 0.75, 1.35, 0.5, 1.5, 0.95]
query_re = [4.5e6, 5.1e6, 5.0e6, 4.7e6, 3.3e6, 7.2e6, 2.5e6]

# Shift horizontal coordinates (X-axis) by adding 5.0
m_lofi = m_lofi + 5.0
m_hifi = m_hifi + 5.0
hifi_points[:, 0] = hifi_points[:, 0] + 5.0
query_m = [m + 5.0 for m in query_m]

# 6. Set up the plot
fig, ax = plt.subplots(figsize=(10, 7))

# Plot Low-Fidelity Data
ax.scatter(m_lofi, re_lofi, marker='^', facecolors='none', edgecolors='steelblue', 
           s=70, label='Low-Fidelity Data')

# Plot High-Fidelity Data
ax.scatter(m_hifi, re_hifi, color='forestgreen', alpha=0.8, s=60, label='High-Fidelity Data')

# Draw the Convex Hull
hull_points = hifi_points[hull.vertices]
hull_polygon = Polygon(hull_points, closed=True, facecolor='forestgreen', alpha=0.15, 
                       edgecolor='black', linewidth=1, label='Convex Hull of Hi-Fi Data')
ax.add_patch(hull_polygon)

# Plot Query Points
ax.scatter(query_m, query_re, marker='x', color='firebrick', s=150, linewidths=2.5, 
           label='Query Points')

# 7. Formatting
ax.set_title('Data Distributions and Convex Hull in M-Re Space', fontsize=18)
ax.set_xlabel('Mach Number (M)', fontsize=14)
ax.set_ylabel('Reynolds Number (Re)', fontsize=14)
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.grid(True, linestyle='-', alpha=0.7)
ax.legend(fontsize=12, loc='lower right')

# Adjust layout and show the figure
plt.tight_layout()
plt.show()