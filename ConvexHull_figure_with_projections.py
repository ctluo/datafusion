import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
from matplotlib.path import Path

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

# 6b. Calculate and plot projections for Query Points outside the convex hull
hull_path = Path(hull_points)
query_pts = np.column_stack((query_m, query_re))

# Determine scaling factors based on high-fidelity points to compute visual/normalized distance
x_min, y_min = hifi_points.min(axis=0)
x_max, y_max = hifi_points.max(axis=0)
scale_x = x_max - x_min
scale_y = y_max - y_min

def to_norm(p):
    return np.array([(p[0] - x_min) / scale_x, (p[1] - y_min) / scale_y])

def to_raw(p_norm):
    return np.array([p_norm[0] * scale_x + x_min, p_norm[1] * scale_y + y_min])

outside_queries = []
projections = []

for q in query_pts:
    # If the point is inside the convex hull, skip projection
    if hull_path.contains_point(q):
        continue
    
    # Normalize query point and hull vertices for visual distance calculations
    q_norm = to_norm(q)
    hull_points_norm = np.array([to_norm(p) for p in hull_points])
    
    best_proj_norm = None
    min_dist = float('inf')
    k = len(hull_points_norm)
    for i in range(k):
        a = hull_points_norm[i]
        b = hull_points_norm[(i + 1) % k]
        ab = b - a
        ab_len_sq = np.sum(ab**2)
        if ab_len_sq == 0:
            proj = a
        else:
            aq = q_norm - a
            t = np.dot(aq, ab) / ab_len_sq
            t = np.clip(t, 0.0, 1.0)
            proj = a + t * ab
        
        dist = np.linalg.norm(q_norm - proj)
        if dist < min_dist:
            min_dist = dist
            best_proj_norm = proj
            
    if best_proj_norm is not None:
        best_proj = to_raw(best_proj_norm)
        outside_queries.append(q)
        projections.append(best_proj)

# Plot connections and projection points
for q, proj in zip(outside_queries, projections):
    # Connect with dashed line
    ax.plot([q[0], proj[0]], [q[1], proj[1]], color='purple', linestyle='--', linewidth=1.5)
    # Purple diamond for projection
    ax.scatter(proj[0], proj[1], color='purple', marker='D', s=80, zorder=5)

# Create a proxy artist for the projection legend entry
import matplotlib.lines as mlines
proj_legend_handle = mlines.Line2D([], [], color='purple', linestyle='--', marker='D', 
                                    markersize=8, label='Projection (Outside Hull)')

# 7. Formatting
ax.set_title('Data Distributions and Convex Hull in M-Re Space', fontsize=18)
ax.set_xlabel('Mach Number (M)', fontsize=14)
ax.set_ylabel('Reynolds Number (Re)', fontsize=14)
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.grid(True, linestyle='-', alpha=0.7)

# Retrieve current legend handles and combine them with the projection handle
handles, labels = ax.get_legend_handles_labels()
handles.append(proj_legend_handle)
ax.legend(handles=handles, fontsize=12, loc='lower right')

# Adjust layout, save figure in both PNG and EPS formats, and show the figure
plt.tight_layout()
plt.savefig('projections2D.png', dpi=300)
plt.savefig('projections2D.eps', format='eps')
plt.show()