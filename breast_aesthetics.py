import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import numpy as np
from keras.applications.densenet import preprocess_input
from keras.models import load_model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import pickle 
import scipy.interpolate as interpolate

from cnn_model import CNN_classifier

def spline(points, n_points=9999):
    t = np.arange(0, 1.0000001, 1/n_points)
    x = points[:,0]
    y = points[:,1]
    tck, u = interpolate.splprep([x, y], s=0)
    out = interpolate.splev(t, tck)
    return out

def y_spline(image_coord, breast='right'): 
    points = np.array(image_coord)
    if(breast=='left'):
        y_points = points[1:35:2]
        x_points = points[0:34:2]  
    else: 
        y_points = points[35:69:2]
        x_points = points[34:68:2]
    points = np.array([i for i in zip(x_points, y_points)])
    points = np.array(spline(points)).transpose()
    y_points = points[1:1001:2]
    x_points = points[0:1000:2]

    return y_points
    
    
with open("images.pickle",'rb') as fp:
        X = pickle.load(fp)
X = np.array(X, dtype='float')
X = X[:143]

with open("heatmaps.pickle",'rb') as fp:
        heatmaps = pickle.load(fp)
heatmaps = np.array(heatmaps, dtype='float')
heatmaps = heatmaps[:143]

with open("keypoints.pickle",'rb') as fp:
        keypoints = pickle.load(fp)
keypoints = np.array(keypoints, dtype='float')
keypoints = keypoints[:143]

with open("labels_MJ.pickle",'rb') as fp:
        labels = pickle.load(fp)
labels = np.array(labels, dtype='int')
labels = labels[:143]

X = preprocess_input(X)
labels = np.array(labels)    

y_min_r = [] 
y_min_l = [] 

for i in range(X.shape[0]):
    y_min_r.append(np.min(y_spline(keypoints[i], 'right')))
    y_min_l.append(np.min(y_spline(keypoints[i], 'left')))

y_min_r = np.reshape(np.array(y_min_r), (-1,1))
y_min_l = np.reshape(np.array(y_min_l), (-1,1))
y_nipple_right = np.reshape(keypoints[:,73], (-1,1))
y_nipple_left = np.reshape(keypoints[:,71], (-1,1))
y_sternal = np.reshape(keypoints[:,69], (-1,1))
x_nipple_right = np.reshape(keypoints[:,72], (-1,1))
x_nipple_left = np.reshape(keypoints[:,70], (-1,1))
x_sternal = np.reshape(keypoints[:,68], (-1,1))

fts = np.concatenate([y_min_r, y_min_l, y_nipple_right, y_nipple_left, y_sternal, \
                      x_nipple_right, x_nipple_left, x_sternal], axis=1)
fts = fts/np.max(fts)


labels_cpy = labels.copy()
labels[np.where(labels_cpy<2)]=0
labels[np.where(labels_cpy>=2)]=1


predictions = [] 
gnd = [] 

sk_init = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for init_idx, val_idx in sk_init.split(X, labels):
    X_after, X_test =  X[init_idx], X[val_idx]
    fts_after, fts_test = fts[init_idx], fts[val_idx]
    y_after, y_test = labels[init_idx], labels[val_idx]
        
    break

model = CNN_classifier()
model.fit(X_after, fts_after, y_after)
preds = model.predict(X_test)

print(preds[1])
print(np.round(preds[1],0))
print(y_test)

p = np.array(np.round(preds[1],0), dtype='int')

predictions.append(p.tolist())
gnd.append(y_test)
    
predictions = [item for sublist in predictions for item in sublist]
predictions = [item for sublist in predictions for item in sublist]
gnd = [item for sublist in gnd for item in sublist]
print('Predictions: ', predictions)
print('Ground-truth: ', gnd)

print(accuracy_score(gnd, predictions))
print(balanced_accuracy_score(gnd, predictions))




