""" Model """

from train_generator_fts_clf import Generator

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

import keras.backend as K
from keras.layers import Dense, Conv2D, MaxPooling2D,\
Flatten, Lambda, Reshape, concatenate, Dropout, Input
from keras.models import Model, load_model
from keras.constraints import non_neg
from keras.losses import mean_squared_error, binary_crossentropy
from keras.callbacks import ModelCheckpoint

class CNN_classifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        return None
        
    def build_model(self, X):
        rows = X.shape[1]
        cols = X.shape[2]
        channels = X.shape[3]
        
        inputs = Input((rows, cols, channels))
        # feature extractor
        def conv_net(inputs):
            conv1 = Conv2D(4, kernel_size=(3,3), activation='relu',\
                           name='conv_1', \
                           )(inputs)
            conv1 = Conv2D(4, kernel_size=(3,3), activation='relu',\
                           name='conv_2', \
                          )(conv1)
            conv1 = MaxPooling2D((2,2))(conv1)
            conv1 = Conv2D(8, kernel_size=(3,3), activation='relu',\
                           name='conv_3', \
                           )(conv1)
            conv1 = Conv2D(8, kernel_size=(3,3), activation='relu',\
                           name='conv_4', \
                           )(conv1)
            conv1 = MaxPooling2D((2,2))(conv1)
            conv1 = Conv2D(16, kernel_size=(3,3), activation='relu',\
                           name='conv_5', \
                           )(conv1)
            conv1 = Conv2D(16, kernel_size=(3,3), activation='relu',\
                           name='conv_6', \
                           )(conv1)
            conv1 = MaxPooling2D((2,2))(conv1)
            conv1 = Conv2D(32, kernel_size=(3,3), activation='relu',\
                           name='conv_7', \
                           )(conv1)
            conv1 = Conv2D(32, kernel_size=(3,3), activation='relu',\
                           name='conv_8', \
                           )(conv1)
            conv1 = MaxPooling2D((2,2))(conv1)
            conv1 = Conv2D(64, kernel_size=(3,3), activation='relu',\
                           name='conv_9', \
                           )(conv1)
            conv1 = Conv2D(64, kernel_size=(3,3), activation='relu',\
                           name='conv_10', \
                           )(conv1)
            conv1 = MaxPooling2D((2,2))(conv1)
               
            return conv1         
        
        # keypoint detection part 
        def fts_computation(inputs):
            flat = Flatten()(inputs)
            dense1 = Dense(100, activation='relu', name='fts_1', \
                           )(flat)
            dense1 = Dropout(0.3)(dense1)
            dense1 = Dense(20, activation='relu', name='fts_2', \
                           )(dense1)
            dense1 = Dense(8, activation='linear', name='features',\
                           )(dense1)
           
            return dense1        
        
        def mlp(input_fts):
            """ difference between lower breast contours """
            lbc = \
            Lambda(lambda input_fts: \
                   # |y_l_min - y_r_min|
                   (K.abs(input_fts[:,1] - input_fts[:,0])) / \
                   # |y_sternal - y_nipple_l|
                   ((\
                     K.abs(input_fts[:,4] - input_fts[:,3]) + \
                   # |y_sternal - y_nipple_r|  
                     K.abs(input_fts[:,4] - input_fts[:,2]) + \
                   # |y_nipple_l - y_l_min|  
                     K.abs(input_fts[:,3] - input_fts[:,1]) + \
                   # |y_nipple_r - y_r_min|  
                     K.abs(input_fts[:,2] - input_fts[:,0])\
                     )/2)\
                     )(input_fts)
            lbc = Reshape((1,))(lbc)
            
            """ difference between inframammary fold distances - |NI1 - NI2| """
            bce = \
            Lambda(lambda input_fts: \
                   np.abs(\
                          # |y_nipple_l - y_l_min|
                          K.abs(input_fts[:,3] - input_fts[:,1]) - \
                          # |y_nipple_r - y_r_min|
                          K.abs(input_fts[:,2] - input_fts[:,0])\
                          )/\
                          (\
                          # |y_nipple_l - y_l_min|
                          (K.abs(input_fts[:,3] - input_fts[:,1]) + \
                          # |y_nipple_r - y_r_min|
                          K.abs(input_fts[:,2] - input_fts[:,0])\
                           )/2)\
                           )(input_fts)
            bce = Reshape((1,))(bce)
            
            """ difference between nipple levels - |Y1 - Y2|"""
            unr = \
            Lambda(lambda input_fts: \
                   np.abs(\
                          # |y_sternal - y_nipple_l|
                          K.abs(input_fts[:,4] - input_fts[:,3]) - \
                          # |y_sternal - y_nipple_r|
                          K.abs(input_fts[:,4] - input_fts[:,2])\
                          )/\
                          (\
                           # |y_sternal - y_nipple_l|
                           (K.abs(input_fts[:,4] - input_fts[:,3]) + \
                           # |y_sternal - y_nipple_r|
                           K.abs(input_fts[:,4] - input_fts[:,2]))\
                           /2)\
                            )(input_fts)
            unr = Reshape((1,))(unr)
            
            """ Breast Retraction Assessment """
            bra = \
            Lambda(lambda input_fts: \
                   K.sqrt(\
                           K.square(\
                                     # |x_sternal - x_nipple_l|
                                     K.abs(input_fts[:,7] - input_fts[:,6]) - \
                                     # |x_sternal - x_nipple_r|
                                     K.abs(input_fts[:,7] - input_fts[:,5])) + \
                           K.square(\
                                     # |y_sternal - y_nipple_l|
                                     K.abs(input_fts[:,4] - input_fts[:,3]) - \
                                     K.abs(input_fts[:,4] - input_fts[:,2])\
                                     )\
                           )\
                   /((\
                      K.sqrt(\
                              # |x_sternal - x_nipple_l|
                              K.square(input_fts[:,7] - input_fts[:,6]) + \
                              # |y_sternal - y_nipple_l|
                              K.square(input_fts[:,4] - input_fts[:,3])\
                              ) + \
                      K.sqrt(\
                              # |x_sternal - x_nipple_r|
                              K.square(input_fts[:,7] - input_fts[:,5]) + \
                              # |y_sternal - y_nipple_r|
                              K.square(input_fts[:,4] - input_fts[:,2])\
                              )\
                      )/2))(input_fts)
            bra = Reshape((1,))(bra)
            
            input_fts = concatenate([unr, lbc, bce, bra])
            dense1 = Dense(25, activation='relu', name='first_dense', \
                           kernel_constraint=non_neg(),\
                           )(input_fts)
            dense1 = Dense(10, activation='relu', name='second_dense', \
                           kernel_constraint=non_neg(),\
                           )(dense1)
            dense1 = Dense(1, activation='sigmoid', name='classification',\
                           )(dense1)
            return dense1        
            
        fts_ = conv_net(inputs)
        raw_fts = fts_computation(fts_)
        clf = mlp(raw_fts)
       
        model = Model(inputs=inputs, outputs=[raw_fts, clf])    
        model.summary()
        
        return model
            
    def fit(self, X, fts, y): 
        # split data into train and validation
        sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, test_idx in sk.split(X, y):
            X_train, X_val = X[train_idx], X[test_idx]
            fts_train, fts_val = fts[train_idx], fts[test_idx]
            y_train, y_val = y[train_idx], y[test_idx]
            
            break
        
        class_weights = compute_class_weight('balanced', \
                      np.unique(y_train), y_train)
        
        self.model = self.build_model(X)
        
        for layer in self.model.layers: 
            if(layer.name=='first_dense' or layer.name=='second_dense' \
               or layer.name=='classification'):
                layer.trainable=False
        
        self.model.compile(optimizer='adadelta', loss=[mean_squared_error, \
                                  binary_crossentropy], \
                                  loss_weights = [1,0],\
                                  metrics={'classification':'accuracy'})
        
        checkpoint = ModelCheckpoint('model__.hdf5', \
                                     monitor='val_features_loss', \
                                     save_best_only=True, verbose=True)
        
        my_generator = Generator(X_train,
                         fts_train,
                         y_train,
                         batchsize=16,
                         flip_ratio=0.5,
                         translation_ratio=0.5,
                         rotate_ratio=0)


        self.model.fit_generator(my_generator.generate(), \
                            steps_per_epoch = X_train.shape[0]/16, epochs=350,\
                            verbose=2,\
                            validation_data = (X_val, [fts_val, y_val]),\
                            callbacks=[checkpoint])   
        
        self.model = load_model('model__.hdf5')
        
        for layer in self.model.layers:
            if(layer.name=='first_dense' or layer.name=='second_dense' or\
               layer.name == 'third_dense' or layer.name=='classification'):
                layer.trainable=True
        
        for layer in self.model.layers[:13]:
            print(layer.name)
            layer.trainable = False
            
        self.model.compile(optimizer='adadelta', loss=[mean_squared_error, binary_crossentropy], \
              loss_weights = [1000,1],\
              metrics={'classification':'acc'})  
   
        checkpoint = ModelCheckpoint('model_clf__.hdf5', \
                                     monitor='val_classification_acc', \
                                     save_best_only=True, verbose=True, mode='max')
        
        my_generator = Generator(X_train,
                                 fts_train,
                                 y_train,
                                 batchsize=16,
                                 flip_ratio=0.5,
                                 translation_ratio=0.5,\
                                 rotate_ratio = 0)
        
        self.model.fit_generator(my_generator.generate(), \
                        steps_per_epoch = X_train.shape[0]/16, epochs=250,\
                        verbose=2, \
                        class_weight={'classification':class_weights},\
                        validation_data = (X_val, [fts_val, y_val]),\
                        callbacks=[checkpoint])
                
        return self
        
        
    def predict(self, X): 
        path = 'model_clf__.hdf5'
        model = load_model(path)
        preds = model.predict(X)
        
        return preds 

        
    
    