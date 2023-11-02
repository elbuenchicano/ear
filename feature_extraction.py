import numpy as np
import tensorflow as tf
import pandas as pd

from skimage import io
from skimage.morphology import label, dilation, ball
from skimage.measure    import regionprops
from skimage.transform  import resize, rotate, rescale
from skimage.color      import gray2rgb, rgb2hsv
from skimage.exposure   import rescale_intensity
from skimage.filters    import gaussian, threshold_otsu

from sklearn.preprocessing import normalize

from scipy.ndimage import gaussian_filter1d
from scipy.spatial import distance_matrix

# Bring this code here and make it self-contained
from ear_utils import heatmaps2coords, spline_and_resample, rotate_to_major_axis

import matplotlib.pyplot as plt


def flatten(l):
    return [item for sublist in l for item in sublist]

class BaseFeatureExtractor(object):
    """ """
    
    parts = [slice(0,20),slice(20,35),slice(35,50),slice(50,55)]
    
    def __init__(self, detector_model, **kwargs):
        self.detector_model = detector_model
        
    def get_crop(self, img, pad_width=((15,15),(15,15),(0,0)), output_size='same', remove_background=False):
        r,c,l   = img.shape
        pred    = self.detector_model.predict(np.array([img]), verbose=0)[0]
        pred_th = pred > threshold_otsu(pred)        
        mask    = label(pred_th)
        
        if remove_background:
            img  = img*mask
            mode = 'constant'
        else:
            mode = 'edge'
        
        rprops = regionprops(mask)
        r0,c0,l0,r1,c1,l1 = rprops[np.argmax([e.area for e in rprops])].bbox
        cropped = img[r0:r1,c0:c1,:]
        ret = np.pad(cropped, pad_width, mode)
        if output_size == 'same': ret = resize(ret, (r,c,l))
        return ret


class InterpretableFeatureExtractor(BaseFeatureExtractor):
    """ """
    def __init__(self, detector_model, landmark_model, append_color=False, **kwargs):
        self.landmark_model = landmark_model
        super().__init__(detector_model, **kwargs)
        self.append_color = append_color
        self.baseline_edm = None

    def _get_landmarks_from_image(self, img, n_realign, return_image, remove_background):
        # For landmarks, it makes no sense to remove background
        img_crop = self.get_crop(img, remove_background=False)
        
        # Predict and realign
        img_aligned = np.array([img_crop])
        for i in range(n_realign):
            pred = self.landmark_model.predict(img_aligned, verbose=0)[-1]
            x,y = heatmaps2coords(pred, mode='centroid')
            img_aligned = rotate_to_major_axis(img_aligned[0], x, y)[0]
            img_aligned = np.array([img_aligned])

        # Predict landmarks for aligned image
        pred = self.landmark_model.predict(img_aligned, verbose=0)[-1]
        x,y = heatmaps2coords(pred, mode='centroid')

        # Low-pass filter by parts
        for i,p in enumerate(self.parts):
            x[p],y[p] = [gaussian_filter1d(e, sigma=1) for e in [x[p],y[p]]]

        # Resample from spline approximation
        x,y = zip(*[spline_and_resample(np.array(x[p]), np.array(y[p])) for p in self.parts])
        x,y = flatten(x),flatten(y)

        ret = np.array(x, dtype=object),np.array(y, dtype=object)
        if return_image: ret = img_aligned[-1],*ret
        return ret
        
    def _get_landmarks(self, img_list, n_realign=0, return_image=False, remove_background=False):
        if img_list.ndim < 4:
            ret = self._get_landmarks_from_image(img_list, n_realign, return_image, remove_background)
        else:
            ret = [self._get_landmarks_from_image(img, n_realign, return_image, remove_background)
               for img in img_list]
        return ret

    def _set_standard_shape(self, X):
        distances = self.get_features(X, n_realign=0, bins=256, remove_background=False)
        self.baseline_edm = np.mean(distances, axis=0)
        return self.baseline_edm
        
    def _get_color_histograms_from_image(self, img, bins):
        # Get non-zero values
        mask = np.zeros(img.shape, dtype=bool)
        mask[img > 0] = True

        img_hsv = rgb2hsv(img)
        hue = np.histogram(img_hsv[:,:,0].flatten()[mask[:,:,0].flatten()], range=(0,1), bins=bins)[0]
        return hue
    
    def _get_color_histograms(self, img_list, bins=256):
        if img_list.ndim < 4:
            ret = self._get_color_histograms_from_image(img_list, bins=bins)
        else:
            ret = [self._get_color_histograms_from_image(img, bins=bins) for img in img_list]
        return ret
    
    def get_features(self, X, n_realign=0, bins=256, remove_background=False, difference_to_baseline=False):
        X = np.array(X)
        
        landmarks = self._get_landmarks(X, n_realign=n_realign, remove_background=remove_background)
        points    = [np.swapaxes(e, 0, 1) for e in landmarks]
        distances = normalize([distance_matrix(e, e, p=2).flatten() for e in points])
        if difference_to_baseline:
            distances = [e-self.baseline_edm for e in distances]
        if self.append_color:
            histograms = self._get_color_histograms(X, bins=bins)
            ret = [np.concatenate([distances[i], histograms[i]])
                  for i in range(len(distances))]
        else:
            ret = distances
        return ret
