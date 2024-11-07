# This shows what different trajectories look like the UMAP

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np

# Load the trajectories
trajectories = np.load('coordinatesNorm.npy')

# This is how many frames you want to show
frame_counts = [25, 50, 75, 100] 

fig, axs = plt.subplots(2, 2)
CMAPS = ['Reds', 'Oranges', 'Greens', 'Blues', 'Purples', 'Greys']
color = ['red', 'orange', 'green', 'blue', 'purple', 'black']

axs = axs.flatten()

trajectoryIndices = [179, 100] # Specify the trajectories you want to look at
theTrajectories = trajectories[trajectoryIndices]

for idx, N_FRAMES in enumerate(frame_counts):
    ax = axs[idx]
    
    for i, traj in enumerate(theTrajectories):
        print(len(traj))
        x = traj[:N_FRAMES, 0]
        y = traj[:N_FRAMES, 1]
        z = np.arange(N_FRAMES)
        ax.scatter(x, y, c=z, cmap=CMAPS[i], s=10.0, zorder=10, alpha=0.7)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        cmap = plt.get_cmap(CMAPS[i])
        norm = plt.Normalize(z.min(), z.max())
        lc = LineCollection(segments, cmap=cmap, norm=norm, zorder=9, alpha=0.5)
        lc.set_array(z)
        lc.set_linewidth(2)
        ax.add_collection(lc)
        ax.scatter(x[0], y[0], color='white', s=100, zorder=15)
        ax.scatter(x[-1], y[-1], color=color[i], s=100, zorder=15)
    
    ax.set_facecolor('xkcd:grey')
    ax.set_title(f'First {N_FRAMES} Frames')
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

plt.show()
