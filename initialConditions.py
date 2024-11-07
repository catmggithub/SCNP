import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap



# Trains a random forest model to see if there's a relationship between anythingn and eigenvectors
def randomForest(eigNum):
    X = []
    positions = np.load("initialPositions.npy")
    velocities = np.load("initialVelocities.npy")
    endstates = np.load("endstates.npy")
    sequenceInfo = pd.read_csv("trajectory_info.csv")['sequence'].tolist()
    for i in range(210):
        sequence = np.array([int(j) for j in sequenceInfo[i]])
        X.append(np.concatenate((positions[i].flatten(), velocities[i].flatten(), sequence, endstates[i].flatten())))
    print("a")
    print(X[0])
    print(X[-1])
    y = np.load('eig' + str(eigNum) + '.npy').tolist()
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)
    print("b")
    def train_and_evaluate(XTrain, XTest, yTrain, yTest):
        model = RandomForestRegressor(n_estimators=100, random_state=100)
        model.fit(XTrain, yTrain)
        predictions = model.predict(XTest)
        r2 = r2_score(yTest, predictions)
        featureImportance = model.feature_importances_
        return predictions, r2, featureImportance
    yPred, r2, featureImportance = train_and_evaluate(XTrain, XTest, yTrain, yTest)

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(featureImportance)), featureImportance)
    plt.title(f'Feature Importance Distribution for All Information')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance Score')
    plt.show()
    
    print("c")
    plt.scatter(yTest, yPred)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values From Random Forest')
    plt.title('2nd Eigenvector Prediction with All Information')
    plt.plot([min(yPred), max(yPred)], [min(yPred), max(yPred)], 'k--')
    print("d")
    plt.show()


def hope():
    positions = np.load('initialPositions.npy')
    velocities = np.load('initialVelocities.npy')
    eig2 = np.load('eig2.npy')
    eig3 = np.load('eig3.npy')
    averageVelocity = []
    for velocity in velocities:
        averageVelocity.append(np.mean(velocity.flatten()))
    norm_rg = plt.Normalize(vmin=min(averageVelocity), vmax = max(averageVelocity))
    plt.scatter(eig2, eig3, c=averageVelocity, cmap='viridis', norm=norm_rg)
    plt.title("Average Velocity vs Eigenvectors")
    plt.show()

def frac():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    eig2 = np.load('eigenvectors/eig2.npy')
    eig3 = np.load('eigenvectors/eig3.npy')
    fraction = pd.read_csv("trajectory_info.csv")['fraction'].tolist()
    blockiness = pd.read_csv("trajectory_info.csv")['blockiness'].tolist()


    cmapf = {
        .2: "#191970",
        .3: "#008080",
        .4: "#9D00FF",
        .7: "#FA8072",
        .8: "#FFB6C1"
    }
    
    cmapb = {
        .2: "#191970",
        .4: "#008080",
        .6: "#9D00FF",
        .8: "#FFB6C1",
    }
    
    colorsf = [cmapf[frac] for frac in fraction]
    colorsb = [cmapb[bloc] for bloc in blockiness]
    
    sc1 = ax1.scatter(eig2, eig3, c=colorsf)
    sc2 = ax2.scatter(eig2, eig3, c=colorsb)

    # Create legend for the first subplot
    handlesf = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmapf[key], markersize=10) for key in cmapf]
    labelsf = [f'Fraction: {key}' for key in cmapf]
    ax1.legend(handlesf, labelsf, loc='lower right')

    # Create legend for the second subplot
    handlesb = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmapb[key], markersize=10) for key in cmapb]
    labelsb = [f'Blockiness: {key}' for key in cmapb]
    ax2.legend(handlesb, labelsb , loc='lower right')

    plt.show()

def michelleCurious():
    eig2 = np.load('eigenvectors/eig2.npy')
    fraction = pd.read_csv("trajectory_info.csv")['fraction'].tolist()
    blockiness = pd.read_csv("trajectory_info.csv")['blockiness'].tolist()
    pink_teal_cmap = LinearSegmentedColormap.from_list("pink_teal", ["#008080", "#FFB6C1"])
    norm = plt.Normalize(vmin=min(eig2), vmax=max(eig2))
    plt.scatter(fraction, blockiness, c=eig2, cmap=pink_teal_cmap, norm=norm)
    plt.show()
michelleCurious()
    
def metrics():
    # Get data
    positions = np.load('finalPositions.npy')
    eig2 = np.load('eigenvectors/eig2.npy')
    eig3 = np.load('eigenvectors/eig3.npy')

    pink_teal_cmap = LinearSegmentedColormap.from_list("pink_teal", ["#008080", "#FFB6C1"])
    # Calculate topological descriptors
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

    # Convert lists to arrays for plotting
    rg = np.array(rg)
    c = np.array(c)
    b = np.array(b)
    k = np.array(k)

    # Plot results
    figure, axis = plt.subplots(2, 2, figsize=(10, 8))

    # Use colormap normalization (optional: adjust the range using vmin and vmax)
    norm_rg = plt.Normalize(vmin=min(rg), vmax=100)
    norm_c = plt.Normalize(vmin=min(c), vmax=1500)
    norm_b = plt.Normalize(vmin=min(b), vmax=10000)
    norm_k = plt.Normalize(vmin=min(k), vmax=max(k))

    # Apply colormap and scatter
    sc1 = axis[0, 0].scatter(eig2, eig3, c=rg, cmap=pink_teal_cmap, norm=norm_rg)
    sc2 = axis[0, 1].scatter(eig2, eig3, c=c, cmap=pink_teal_cmap, norm=norm_c)
    sc3 = axis[1, 0].scatter(eig2, eig3, c=b, cmap=pink_teal_cmap, norm=norm_b)
    sc4 = axis[1, 1].scatter(eig2, eig3, c=k, cmap=pink_teal_cmap, norm=norm_k)

    # Add colorbars
    figure.colorbar(sc1, ax=axis[0, 0])
    figure.colorbar(sc2, ax=axis[0, 1])
    figure.colorbar(sc3, ax=axis[1, 0])
    figure.colorbar(sc4, ax=axis[1, 1])

    # Set plot titles

    plt.show()
    print(b)
    
