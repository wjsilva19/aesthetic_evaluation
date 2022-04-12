""" Retrieval study """

import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import matplotlib.pyplot as plt
from keras.applications.densenet import preprocess_input
from keras.models import Model, load_model
from keras import losses
from sklearn.model_selection import StratifiedKFold
import keras.backend as K
from sklearn.metrics import accuracy_score
import pickle

def to_show(img):
    img = img - np.min(img)
    img = img/np.max(img)    
    
    return img
    
with open("images.pickle",'rb') as fp:
        X = pickle.load(fp)
X_show = np.array(X, dtype='int')
X = np.array(X, dtype='float')
X = X[:143]

with open("labels_MJ.pickle",'rb') as fp:
        labels = pickle.load(fp)
labels = np.array(labels, dtype='float')
labels = labels[:143]

# preprocess to feed the network 
X = preprocess_input(X)

aesthetic_labels = labels.copy()
aesthetic_labels[np.where(labels<2)]=0
aesthetic_labels[np.where(labels>=2)]=1

sk_init = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for init_idx, test_idx in sk_init.split(X, aesthetic_labels):
    X_after, X_test =  X[init_idx], X[test_idx]
    y_after, y_test = aesthetic_labels[init_idx], aesthetic_labels[test_idx]
    labels_after, labels_test = labels[init_idx], labels[test_idx]    
    break

sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in sk.split(X_after, y_after):
    X_train, X_val = X_after[train_idx], X_after[val_idx]
    y_train, y_val = y_after[train_idx], y_after[val_idx]
    labels_train, labels_val = labels_after[train_idx], labels_after[val_idx]
    
    break


# load final saved model 
model = load_model('model_clf_new.hdf5')  
model.summary()

train_preds = model.predict(X_train)
val_preds = model.predict(X_val)
test_preds = model.predict(X_test)

print('Training accuracy: ', accuracy_score(y_train, np.array(np.round(train_preds[1],0), dtype='int')))
print('Validation accuracy: ', accuracy_score(y_val, np.array(np.round(val_preds[1],0), dtype='int')))
print('Test accuracy: ', accuracy_score(y_test, np.array(np.round(test_preds[1],0), dtype='int')))

int_model = Model(inputs=model.input, outputs=model.layers[-1].output)
int_model.compile(optimizer='adadelta', loss=losses.binary_crossentropy, metrics=['accuracy'])
int_model.summary()

int_model.load_weights('model_clf_new.hdf5')

get_penultimate_output = K.function([int_model.layers[0].input],\
                                    [int_model.layers[-2].output])

from scipy.spatial.distance import euclidean 
def l2_embedding_distance(l_test, img): 
    img = np.reshape(img, (-1,256,384,3))
    emb = get_penultimate_output([img])[0]
    emb = [item for sublist in emb for item in sublist]
    dist = euclidean(l_test, emb)
    
    return dist

train_to_show = to_show(np.reshape(X_train[2], (256,384,3)))
plt.imshow(train_to_show)
plt.show()

retrieval_mistakes = 0

for i in range(X_test.shape[0]):
    query_image = np.reshape(X_test[i], (-1,256,384,3))
    query_to_show = to_show(np.reshape(X_test[i], (256,384,3)))
    
    print('Test image number: ', i)
    print('Label: ', y_test[i])
    print('Original label: ', labels_test[i])
    
    plt.imshow(query_to_show) 
    plt.show()
    
    l_out = get_penultimate_output([query_image])[0]    
    l_out = [item for sublist in l_out for item in sublist]
    
    dist = [] 
    index = [] 
        
    # For all image in training compute distance to test_image
    for n in range(X_train.shape[0]): 
        index.append(n) # append index to do the retrieval afterwards 
        dist.append(l2_embedding_distance(l_out, X_train[n]))
        
    index = np.asarray(index)
    index = np.reshape(index, (index.shape[0], 1))
    dist = np.asarray(dist)
    dist = np.reshape(dist, (dist.shape[0], 1))
        
    results = np.concatenate([index, dist], axis=1)
    
    results = results[results[:,1].argsort()] 
    
    print('Top-3 Most similar images: ')
    
    count=0
    for j in range(3): 
        print('Original_index: ', results[j,0])
        print('Label: ', y_train[int(results[j,0])])
        print('Original label: ', labels_train[int(results[j,0])])
        
        if((labels_train[int(results[j,0])] - labels_test[i]) > 1):
            retrieval_mistakes+=1
            print('Retrieval mistake!')
            break
        
        img_to_show = to_show(np.reshape(X_train[int(results[j,0])], (256,384,3)))
        plt.imshow(img_to_show)
        plt.show()

print(retrieval_mistakes)
