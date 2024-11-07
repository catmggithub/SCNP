import argparse
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    # Get user input for the trajectory of interest, the number of frames
    # to include, and whether to normalize the trajectory.
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_id', type=int, default=0)
    parser.add_argument('--n_frames', type=int, default=100)
    parser.add_argument('--norm', action='store_false')
    parser.add_argument('--save', default=None)
    parser.add_argument('--interval', type=int, default=200)
    args = parser.parse_args()

    # Load the trajectories.
    trajectories = np.load('trajectoriesNorm.npy')



    # Isolate the specified trajectory.
    traj = trajectories[args.traj_id]

    # Animate the trajectory, generate figure.
    plt.rcParams['font.family'] = 'Helvetica Neue'
    plt.rcParams['axes.linewidth'] = 1.5
    fig, ax = plt.subplots(1,1,figsize=(5,3))
    bars = ax.bar(range(traj.shape[1]), traj[0], color='#ffca8c', edgecolor='black', linewidth=1.0)
    ax.set_title(f'Frame 1 / {args.n_frames}')
    ax.set_xlabel('Local Density')
    ax.set_ylabel('Frequency')
    def update(frame):
        for bar, height in zip(bars, traj[frame]):
            bar.set_height(height)
        ax.set_title(f'Frame {frame+1} / {args.n_frames}')
    ani = FuncAnimation(fig, update, frames=args.n_frames, repeat=True, interval=args.interval)
    plt.tight_layout()
    plt.show()

    # Save figure to the specified location.
    if args.save is not None:
        ani.save(args.save, writer=PillowWriter(fps=1000 / args.interval))