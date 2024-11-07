import numpy as np
import umap


# Function that cuts off trajectories when they appear stagnant 
def cutOffTrajectory(vectors, treshold):
    vectorsNum = len(vectors)
    endVector = np.array(vectors[-1])
    differences = [np.linalg.norm(np.array(vectors[i]) - endVector) for i in range(vectorsNum)]
    differences = np.array(differences)
    for i in range(10, vectorsNum):
        movingAverage = np.sum(differences[i-10:i])/10
        if movingAverage < treshold:
            return vectors[:i+1].tolist()
    return vectors.tolist()

# Cut off the trajectories to prepare to create the UMAP
trajectories = np.load('trajectories.npy')
umapList = cutOffTrajectory(trajectories[0], 30)
for i in range(len(trajectories)-1):
    umapList += cutOffTrajectory(trajectories[i+1], 30)
print(len(umapList))
umapList = np.array(umapList)

# Run the UMAP
reducer = umap.UMAP(n_neighbors = 200, min_dist = 1, n_components = 2, metric = 'euclidean', verbose=True)
reducer.fit(umapList)
reducedTrajectories = reducer.transform(trajectories)
with open('coordinates.npy', 'wb') as f:
    np.save(f, np.array(reducedTrajectories))

