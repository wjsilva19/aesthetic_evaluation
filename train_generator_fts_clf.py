
import math
import numpy as np
import cv2
#import matplotlib.pyplot as plt
#from skimage.transform import rotate as rotate_

def translate_points(point,translation): 
    point = point + translation 
    
    return point

def rotate_points(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

np.random.seed(42)


class Generator(object):
    def __init__(self,
                 X_train,
                 features_train,
                 y_train,
                 batchsize=2,
                 flip_ratio=0.1,
                 translation_ratio=0.1,
                 rotate_ratio=0.1):
        """
        Arguments
        ---------
        """
        self.X_train = X_train
        self.features_train = features_train
        self.y_train = y_train
        self.size_train = X_train.shape[0]
        self.batchsize = batchsize
        self.flip_ratio = flip_ratio
        self.translation_ratio = translation_ratio
        self.rotate_ratio = rotate_ratio
    

    def _random_indices(self, ratio):
        """Generate random unique indices according to ratio"""
        size = int(self.actual_batchsize * ratio)
        return np.random.choice(self.actual_batchsize, size, replace=False)
    
    def flip(self):
        """Flip image batch"""
        indices = self._random_indices(self.flip_ratio)
        self.inputs[indices] = self.inputs[indices,:,::-1,:]
        
        flip_indices = [(0,1), (2,3), (5,6)]
        self.features[indices, 5:8] = 1 - self.features[indices, 5:8]
        for a, b in flip_indices:
            self.features[indices, a], self.features[indices, b] = \
            (self.features[indices, b], self.features[indices, a])
                    
            
    def translation(self): 
        """Translation"""        
        indices = self._random_indices(self.translation_ratio)
        tx = np.random.randint(-30, 30)
        ty = np.random.randint(-20, 20)
        
        x_t = self.features[indices, 5:8]
        y_t = self.features[indices, 0:5]
        
        for i in range(np.shape(y_t)[0]):
            y_t[i] = translate_points(y_t[i],ty/384)
            x_t[i] = translate_points(x_t[i],tx/384)            

        self.features[indices, 5:8] = x_t                
        self.features[indices, 0:5] = y_t 
                
        image = self.inputs[indices,:,:,:]
                     
        for i in range(np.shape(image)[0]):
            for j in range(3): #alterar para 3 com rgb
                image[i,:,:,j] = cv2.warpAffine(image[i,:,:,j],np.float32([[1,0,tx],\
                                            [0,1,ty]]),(384,256))            
        
        self.inputs[indices,:,:,:] = image[:,:,:,:]


    def rotate(self):
        """Rotate slighly the image and the targets."""
        indices = self._random_indices(self.rotate_ratio)
        angle = np.random.randint(-5, 5)

        M = cv2.getRotationMatrix2D((384/2,256/2),angle,1)
        
        for i in indices: 
            for j in range(3): #alterar para 3 com rgb 
                self.inputs[i,:,:,j] = cv2.warpAffine(self.inputs[i,:,:,j], M, (384,256))
                        
        x_r = []
        y_r = [] 
        
        j = 0 
        for i in indices: 
            x_r.append(self.features[i][5:7])
            y_r.append(self.features[i][2:4])
            x_r[j], y_r[j] = rotate_points((192/384,128/384),(x_r[j], y_r[j]),(-angle * 2 * np.pi)/360) 
            self.features[i][2:4] = y_r[j]
            self.features[i][5:7] = x_r[j]
            j += 1
            
        y_r_n = [] 
        
        j = 0 
        for i in indices: 
            y_r_n.append(self.features[i][0:2])
            _, y_r_n[j] = rotate_points((192/384,128/384),(0.1, y_r_n[j]),(-angle * 2 * np.pi)/360) 
            self.features[i][0:2] = y_r_n[j]
            j += 1
        
        for i in indices: 
            self.features[i][7], _ = rotate_points((192/384,192/384),\
                         (self.features[i][7], 0.1),(-angle * 2 * np.pi)/360)
          
        for i in indices: 
            _, self.features[i][4] = rotate_points((192/384,192/384),\
                         (0.1, self.features[i][4]),(-angle * 2 * np.pi)/360)
            
            
    def generate(self, batchsize=32): 
        print(self.batchsize,'batchsize')
        """Generator"""
        while True:
            cuts = [(b, min(b + self.batchsize, self.size_train)) for b in range(0, self.size_train, self.batchsize)]
            for start, end in cuts:
                self.inputs = self.X_train[start:end].copy()
                self.features = self.features_train[start:end].copy()
                self.y = self.y_train[start:end].copy()
                self.actual_batchsize = self.inputs.shape[0]  # Need this to avoid indices out of bounds
                self.flip()
                self.translation()
                self.rotate()
                
                
                yield (self.inputs, [self.features, self.y])
    