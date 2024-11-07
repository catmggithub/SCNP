import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# Calculating Some Metrics

# Calculate topological descriptors for initial bead
positions = np.load("initialPositions.npy")
rg = []
c = []
b = []
k = []
for position in positions:  
    com = np.mean(position, axis = 0)
    positionCenter = position - com
    gyrationTensor = np.zeros((3, 3))
    for pos in positionCenter:
        gyrationTensor += np.outer(pos, pos)
    gyrationTensor = gyrationTensor/400
    eigenvalues = np.linalg.eigvalsh(gyrationTensor)
    pos1 = eigenvalues[0]
    pos2 = eigenvalues[1]
    pos3 = eigenvalues[2]
    rg.append(np.sqrt(pos1**2 + pos2**2 + pos3**2))
    c.append(pos2**2 - pos1**2)
    b.append(pos3**2-1/2*(pos1**2 + pos2**2))
    k.append(3/2*(pos1**4 + pos2**4 + pos3**4)/(pos1**2+pos2**2+pos3**2)**2-1/2)

# Calculate average velocity/standard deviation velocity
velocities = np.load("initialVelocities.npy")
averageVelocity = []
standardDevVelocity = []
for velocity in velocities:
    velocity = velocity.flatten()
    averageVelocity.append(np.mean(velocity))
    standardDevVelocity.append(np.std(velocity))


# Calculates stuff for sequence
freq = pd.read_csv("trajectory_info.csv")['fraction'].tolist()
bloc = pd.read_csv("trajectory_info.csv")['fraction'].tolist()
sequenceInfo = pd.read_csv("trajectory_info.csv")['sequence'].tolist()
for i in range(210):
    sequence = np.array([int(j) for j in sequenceInfo[i]])

# Calculate topological descriptors for endstates
finalPositions = np.load("finalPositions.npy")
rgEnd = []
cEnd = []
bEnd = []
kEnd = []
for endstate in finalPositions:  
    com = np.mean(endstate, axis = 0)
    endstateCenter = endstate - com
    gyrationTensor = np.zeros((3, 3))
    for pos in endstateCenter:
        gyrationTensor += np.outer(pos, pos)
    gyrationTensor = gyrationTensor/400
    eigenvalues = np.linalg.eigvalsh(gyrationTensor)
    end1 = eigenvalues[0]
    end2 = eigenvalues[1]
    end3 = eigenvalues[2]
    rgEnd.append(np.sqrt(end1**2 + end2**2 + end3**2))
    cEnd.append(end2**2 - end1**2)
    bEnd.append(end3**2-1/2*(end1**2 + end2**2))
    kEnd.append(3/2*(end1**4 + end2**4 + end3**4)/(end1**2+end2**2+end3**2)**2-1/2)

rgEnd = np.array(rgEnd)
cEnd = np.array(cEnd)
bEnd = np.array(bEnd)
kEnd = np.array(kEnd)

morphology = np.load("morphologies.npy")

morphologyNumber = []
for morph in morphology:
    if morph == "globular":
        morphologyNumber.append(0)
    elif morph == "rod":
        morphologyNumber.append(1)
    elif morph == "tadpole":
        morphologyNumber.append(2)
    elif morph == "pearl necklace":
        morphologyNumber.append(3)
    else:
        print("warning")

X = []
eig2 = np.load('eigenvectors/eig2.npy')
sequenceInfo = pd.read_csv("trajectory_info.csv")['sequence'].tolist()
for i in range(210):
    initialPositionStuff = [rg[i], c[i], b[i], k[i]]
    initialVelocityStuff = [averageVelocity[i], standardDevVelocity[i]]
    sequence = np.array([int(j) for j in sequenceInfo[i]])
    sequenceStuff = [freq[i], bloc[i]]
    endStuff = [rgEnd[i], cEnd[i], bEnd[i], kEnd[i],  morphologyNumber[i]]
    X.append([eig2[i]])
    # X.append(np.concatenate((initialPositionStuff, initialVelocityStuff, sequenceStuff, endStuff))) # Here's where you change what the random forest gets
print("a")
print(X[0])
print(X[-1])
y = np.load('eigenvectors/eig3.npy').tolist() #Here's where you change the eigenvector number
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)
print("b")
def train_and_evaluate(XTrain, XTest, yTrain, yTest):
    model = RandomForestRegressor(n_estimators=100, random_state=50)
    model.fit(XTrain, yTrain)
    predictions = model.predict(XTest)
    r2 = r2_score(yTest, predictions)
    featureImportance = model.feature_importances_
    return predictions, r2, featureImportance
yPred, r2, featureImportance = train_and_evaluate(XTrain, XTest, yTrain, yTest)


print("c")
plt.scatter(yTest, yPred, c = "#008080")
plt.text(0.05, 0.95, f'$R^2 = {r2:.2f}$', fontsize=12, transform=plt.gca().transAxes)
plt.plot([min(yPred), max(yPred)], [min(yPred), max(yPred)], 'k--')
print("d")
plt.show()