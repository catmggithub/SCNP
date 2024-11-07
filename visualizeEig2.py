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
    parser.add_argument('--eig3', type=float, default=-0.12)
    parser.add_argument('--n_figs', type=int, default=5)
    args = parser.parse_args()

    # Load the trajectories.
    trajectories = np.load('trajectories.npy')

    # If args.norm is specified, normalize the trajectory and select the
    # specified number of frames.
    if args.norm:
        print("hap")
        trajectories = np.load('trajectories.npy')
        trajectoriesNorm = [np.zeros_like(args.n_frames)]*len(trajectories)
        for i in range(len(trajectories)): # len(trajectories)
            traj = trajectories[i][:args.n_frames]
            print(traj)
            minVal = np.min(traj)
            print(minVal)
            maxVal = np.max(traj)
            print(maxVal)
            trajectoriesNorm[i] = (traj - minVal) / (maxVal - minVal)
        trajectoriesNorm = np.array(trajectoriesNorm)
        trajectories = trajectoriesNorm

    
    # Find trajectories within the specified range.
    eig2 = np.load('eigenvectors/eig2.npy')
    eig3 = np.load('eigenvectors/eig3.npy')
    eigs = np.hstack((eig2.reshape(-1,1), eig3.reshape(-1,1)))
    viable_idx = np.argwhere(np.abs(eigs[:,1] - args.eig3) < 0.005)
    viable_idx = viable_idx.reshape(-1)
    viable_idx = np.array(sorted(viable_idx, key=lambda x: eig2[x]))
    chosen_idx = np.linspace(start=0, stop=len(viable_idx) - 1, num=args.n_figs, endpoint=True).astype(int)
    viable_idx = viable_idx[chosen_idx].tolist()
    print(f'Chosen trajectories: {viable_idx}')

    # Generate figure of selected points.
    plt.rcParams['font.family'] = 'Helvetica Neue'
    plt.rcParams['axes.linewidth'] = 1.5
    fig, ax = plt.subplots(1,1,figsize=(5,4))
    ax.scatter(eigs[:,0], eigs[:,1], s=25.0, color='#91dbff', edgecolors='black', linewidth=1.0)
    ax.scatter(eigs[viable_idx,0], eigs[viable_idx,1], s=25.0, color='#fe7c7c', edgecolors='black', linewidth=1.0)
    ax.set_xlabel(r'$\lambda_{2}$', size=14)
    ax.set_ylabel(r'$\lambda_{3}$', size=14)
    ax.tick_params(axis='both', top=True, bottom=True, left=True, right=True, direction='in', width=1.0, length=5.0)
    plt.tight_layout()
    # plt.savefig('./figures/dmap_2d.png', dpi=500)
    plt.show()

    # Animate the trajectory, generate figure.
    plt.rcParams['font.family'] = 'Helvetica Neue'
    plt.rcParams['axes.linewidth'] = 1.5
    fig, axs = plt.subplots(args.n_figs,1,figsize=(5,8), sharex=True)
    bars = []
    for fig_idx, idx in enumerate(viable_idx):
        traj = trajectories[idx]
        bars_ = axs[fig_idx].bar(range(traj.shape[1]), traj[0], color='#ffca8c', edgecolor='black', linewidth=1.0, label= r'$\lambda_{2}$ ='+ f'{eig2[idx]:.3f}')
        bars.append(bars_)
        axs[fig_idx].set_ylabel('Frequency')
        axs[fig_idx].set_ylim(ymin=0.0, ymax=1.0) # Change this for height (0, 250) or (0, 1)
        axs[0].set_title(f'Frame 1 / {args.n_frames}')
        axs[fig_idx].legend()
    axs[-1].set_xlabel('Local Density')
    def update(frame):
        for fig_idx, idx in enumerate(viable_idx):
            for bar, height in zip(bars[fig_idx], trajectories[idx][frame]):
                bar.set_height(height)
            axs[0].set_title(f'Frame {frame+1} / {args.n_frames}')
    ani = FuncAnimation(fig, update, frames=args.n_frames, repeat=True, interval=args.interval)
    plt.tight_layout()
    plt.show()

    # Save figure to the specified location.
    if args.save is not None:
        ani.save(args.save, writer=PillowWriter(fps=1000 / args.interval))
