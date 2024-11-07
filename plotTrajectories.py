import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Plots the trajectories based on second largest eigenvector
def plotTrajectoriesBy1Eig():
    coordinates = np.load('coordinates.npy')
    print(coordinates)
    eig2 = np.load('eigenvectors/eig2.npy')
    eig3 = np.load('eigenvectors/eig3.npy')
    fig = plt.figure()
    
    ax1 = fig.add_subplot(121, projection='3d')
    norm2 = plt.Normalize(vmin=np.min(eig2), vmax=np.max(eig2))
    cmap = LinearSegmentedColormap.from_list("pink_teal", ["#008080", "#FFB6C1"])
    # Plot each trajectory
    for i in range(len(coordinates)):
        x = coordinates[i][:, 0]
        y = coordinates[i][:, 1]
        z = np.arange(len(x))  
        color = cmap(norm2(eig2[i]))
        ax1.plot(x, y, z, c = color, alpha = .3)
        
    ax1.set_xlabel("x-coordinate")
    
    mappable2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm2)
    mappable2.set_array(eig2)
    fig.colorbar(mappable2, ax=ax1)
 
    ax2 = fig.add_subplot(122, projection='3d')
    norm3 = plt.Normalize(vmin=np.min(eig3), vmax=np.max(eig3))
    # Plot each trajectory
    for i in range(len(coordinates)):
        x = coordinates[i][:, 0]
        y = coordinates[i][:, 1]
        z = np.arange(len(x))  
        color = cmap(norm3(eig3[i]))
        ax2.plot(x, y, z, c = color, alpha = .3)
        
    mappable3 = plt.cm.ScalarMappable(cmap=cmap, norm=norm3)
    mappable3.set_array(eig3)
    fig.colorbar(mappable3, ax=ax2)   
    
    plt.show()
plotTrajectoriesBy1Eig()
"""def plotTrajectoriesUMAPEndstate():
    coordinates = np.load("coordinates.npy")
    fig, ax = plt.subplots()
    print(len(coordinates[2]))
    for i in range(210):
        xCord = coordinates[i][-1][0]
        yCord = coordinates[i][-1][1]
        imagePath = "SCNPForms/endstate" + str(i) + ".png"
        if yCord-xCord > .4:
            image = OffsetImage(plt.imread(imagePath), zoom=.01)
        else:
            image = OffsetImage(plt.imread(imagePath), zoom=.02)
        ab = AnnotationBbox(image, (coordinates[i][-1][0], coordinates[i][-1][1]), frameon=False)
        ax.add_artist(ab)
    plt.show()"""
def plotTrajectoriesUMAPEndstate():
    coordinates = np.load("coordinates.npy")
    fig, ax = plt.subplots()
    
    # Loop through the coordinates and add text labels instead of images
    for i in range(210):
        xCord = coordinates[i][-1][0]
        yCord = coordinates[i][-1][1]
        
        # Plot the index number at each coordinate
        ax.text(xCord, yCord, " ", fontsize=8, ha='center', va='center')

    plt.show()


def plotTrajectoriesBy2Eig():

    eig2 = np.load('eigenvectors/eig2.npy')
    eig3 = np.load('eigenvectors/eig3.npy')


    morphology = np.load('morphologies.npy')
    
    
    # Define color mapping for each morphology type
    color_map = {
        "globular": "#191970",
        "rod": "#008080",
        "tadpole": "#FFB6C1",
        "pearl necklace": "#9D00FF",
    }

    # Get colors for each point based on morphology
    colors = [color_map[morph] for morph in morphology]

    # Scatter plot
    plt.scatter(eig2, eig3, c=colors, alpha=1)
    
    # Create legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=morph)
               for morph, color in color_map.items()]
    plt.legend(handles=handles, title='Morphology')

    plt.show()
plotTrajectoriesBy2Eig()

# Plots the trajectories based on the second, third, and fourth largest eigenvector
def plotTrajectoriesBy3Eig():
    eig2 = np.load('eigenvectors/eig2.npy')
    eig3 = np.load('eigenvectors/eig3.npy')
    eig4 = np.load('eigenvectors/eig4.npy')
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    pink_teal_cmap = LinearSegmentedColormap.from_list("pink_teal", ["#008080", "#ff69b4"])
    sc = ax.scatter(eig2, eig3, eig4, c=eig4, cmap = pink_teal_cmap, alpha=1)
    plt.show()

