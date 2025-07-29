
import matplotlib.pyplot as plt
import numpy as np


def plot_point_cloud(train_grid, train_pt_labs):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract x, y, z coordinates from train_grid
    x = train_grid[0, :, 0].cpu().numpy()
    y = train_grid[0, :, 1].cpu().numpy()
    z = train_grid[0, :, 2].cpu().numpy()
    
    # Get labels for coloring (using the first batch item)
    labels = train_pt_labs[0, :, 0].cpu().numpy()
    
    # Plot the points with colors based on labels
    scatter = ax.scatter(x, y, z, c=labels, cmap='viridis', marker='o')
    
    # Add a colorbar to show label values
    plt.colorbar(scatter, ax=ax, label='Point Labels')
    
    # Set equal scaling for all axes to ensure same scale
    max_range = max([
        np.max(x) - np.min(x),
        np.max(y) - np.min(y),
        np.max(z) - np.min(z)
    ])
    mid_x = (np.max(x) + np.min(x)) * 0.5
    mid_y = (np.max(y) + np.min(y)) * 0.5
    mid_z = (np.max(z) + np.min(z)) * 0.5
    ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
    ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
    ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud with Label Colors')

    # Show the plot
    plt.show()
