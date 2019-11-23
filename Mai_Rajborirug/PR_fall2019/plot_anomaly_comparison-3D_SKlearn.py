"""
============================================================================
Comparing anomaly detection algorithms for outlier detection on 3D toy datasets
============================================================================

This example is an extension of anomaly detection comparison from `sckit-learn 2.7.1`. These experiment applied sixalgorithm on 3D toy dataset. The six algorithms are :class:`sklearn.covariance.EllipticEnvelope`, :class:`sklearn.svm.OneClassSVM`, :class:`sklearn.ensemble.IsolationForest`, and :class:`sklearn.neighbors.LocalOutlierFactor`. The classification performance is compared by AUC score :class:`sklearn.metrics.roc_auc_score`.

For the simulation setting, there are 500 samples with outlier fraction = 0.15, 3 information dimension, and adjustable noise dimension (in `D_noise`). The data in information dimension followed figure 1 in the paper: Madhyastha, Meghana, et al. "Geodesic Learning via Unsupervised Decision Forests." arXiv preprint arXiv:1907.02844 (2019). There are six datasets, which are linear, helix, sphere, gaussian mixture, misaligned gaussian misture.

The AUC score shows us that in low dimensional noise (`D_noise=0`), :class:`sklearn.neighbors.LocalOutlierFactor` exceeds other algorithm.  In high dimensional noise (`D_noise=10`), :class:`sklearn.covariance.EllipticEnvelope` and then perform best. The one concern is that :class:`sklearn.covariance.EllipticEnvelope`creates a hyperplane elliptical envolope to differentiate inliers from outliers. Thus its performance decrease when the inlier data is not in elliptical shape. 
"""

# Author: Pharuj Rajborirug <mai.rajborirug.gmail@com> <prajbor1@jhu.edu>

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

import time
import matplotlib
from sklearn import svm
from sklearn.datasets import make_moons, make_blobs
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import roc_auc_score

# Example settings
D_noise = 0                                                             # number of uniform noise dimension
n_samples = 500
outliers_fraction = 0.15
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers
sd= 0.06

# Define datasets
## 1: Linear
t_lin = np.transpose(np.linspace(-1, 1,n_inliers))
X_lin = np.c_[0.4*t_lin + sd * np.random.randn(n_inliers) ,
              0.6*t_lin + sd * np.random.randn(n_inliers),t_lin+ sd * np.random.randn(n_inliers)]
## 2: Helix
t_hex = np.transpose(np.linspace(2*np.pi, 9*np.pi,n_inliers))
xline = t_hex*np.cos(t_hex) # before rescale
xline = xline/(max(xline)-min(xline))*2 + sd * np.random.randn(n_inliers)
yline = t_hex*np.sin(t_hex) # before rescale
yline = yline/(max(yline)-min(yline))*2 + sd * np.random.randn(n_inliers)
zline = (t_hex-(max(t_hex)+min(t_hex))/2)/(max(t_hex)-min(t_hex))*2 + sd * np.random.randn(n_inliers)
X_hex=np.c_[xline,yline,zline]
## 3: Sphere, equally distribution
def fibonacci_sphere(samples=1,randomize=True):
    rnd = 1.
    if randomize:
        rnd = np.random.random() * samples
    points = []
    offset = 2./samples
    increment = np.pi * (3. - np.sqrt(5.));
    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2);
        r = np.sqrt(1 - pow(y,2))
        phi = ((i + rnd) % samples) * increment
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        points.append([x,y,z])
    return points
X_sph=np.array(fibonacci_sphere(samples=n_inliers))
## 4: Gaussian Mixture
blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=3)
X_gau=make_blobs(centers=[[-0.7, -0.7,-0.7], [0, 0,0],
                          [0.7,0.7,0.7]], cluster_std=[0.2, 0.2,0.2],**blobs_params)[0]
## 5: Misaligned Gaussian Mixture
blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=3)
X_misaligned=make_blobs(centers=[[-0.7, -0.7,-0.7], [0.7, 0.7,-0.7],
                          [-0.7, 0.7,0.7]], cluster_std=[0.2, 0.2,0.2],**blobs_params)[0]
## 6: Whole dataset
datasets3D = [X_lin,X_hex,X_sph,X_gau,X_misaligned]

# Define to data label
labels = np.concatenate([np.ones(n_inliers),-np.ones(n_outliers)], axis=0)  
# lbel 1 as inliers, -1 as outliers

# define outlier/anomaly detection methods to be compared
anomaly_algorithms = [
    ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
    ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                      gamma=0.1)),
    ("Isolation Forest (IF)", IsolationForest(n_estimators=500,behaviour='new',
                                         contamination=outliers_fraction,
                                         random_state=42)),
    ("Local Outlier Factor", LocalOutlierFactor(
        n_neighbors=35, contamination=outliers_fraction))] 

plt.figure(figsize=(14,15))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.98, wspace=.05,hspace=.01)
plot_num = 1
rng = np.random.RandomState(42)
noise_params = dict(random_state=0, n_samples=n_samples, n_features=3)
AUC=np.array([])
for i_dataset3D, X in enumerate(datasets3D):
    # Add outliers (i = index (0,1,..), X = data set)
    X = np.concatenate([X, rng.uniform(low=-1.2, high=1.2,
                       size=(n_outliers, 3))], axis=0)                  # add uniform outlier
    X_noise =rng.uniform(low=-1.2, high=1.2,size=(n_samples, D_noise))  # add uniform noise dimension
    X = np.append(X, X_noise, axis=1)
    AUC_row=np.array([])
    for name, algorithm in anomaly_algorithms: 
        t0 = time.time()
        algorithm.fit(X)
        t1 = time.time()

        # fit the data and tag outliers
        if name == "Local Outlier Factor":
            y_pred = algorithm.fit_predict(X)
        else:
            y_pred = algorithm.fit(X).predict(X)
            
        #calculate % accuracy score
        auc = roc_auc_score(y_pred, labels)
        t2 = time.time()    
        ax=plt.subplot(len(datasets3D), len(anomaly_algorithms), plot_num, projection='3d')   
        ax.axis('on')
        
        if i_dataset3D == 0:
            plt.title(name, size=14, color = 'red') # function "name" to be title     
        ax.text2D(.01, .85, ('AUC %.3f' % auc).lstrip('0'),
                  transform=plt.gca().transAxes, size=15,
                  horizontalalignment='left')        
        colors = np.array(['#377eb8', '#ff7f00'])
        ax.scatter3D(X[:, 0], X[:, 1],X[:,2], s=5, color=colors[((y_pred + 1) // 2)]) # label color
        ax.text2D(.99, .01, ('%.2fs' % (t2 - t0)).lstrip('0'),
                  transform=plt.gca().transAxes, size=15,
                  horizontalalignment='right')                         
        ax.set_xticks([])
        ax.set_yticks([])     
        ax.zaxis.set_ticklabels([])
        plot_num += 1
        
        AUC_row=np.append(AUC_row,auc)
        
    AUC = np.append(AUC,AUC_row)
AUC=AUC.reshape(len(datasets3D),len(anomaly_algorithms))
print("D_noise = ", str(D_noise))
plt.show()
print('AUC',AUC)