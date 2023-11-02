import numpy as np
import tensorflow as tf
import scipy
import pandas as pd

import os
import math

import matplotlib.pyplot as plt
# import tqdm.notebook as tqdm
import tqdm as tqdm
import albumentations as A
import skimage
import sklearn
import pickle

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from glob import glob
from joblib import Parallel, delayed
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA

from skimage import io
from skimage.morphology import label, dilation, ball
from skimage.measure import regionprops
from skimage.transform import resize, rotate, rescale
from skimage.color import gray2rgb, rgb2hsv
from skimage.exposure import rescale_intensity
from skimage.filters import gaussian, threshold_otsu

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, KFold, cross_val_score, GridSearchCV
from sklearn.cross_decomposition import PLSRegression

import models

def rotate_coordinates(x, y, angle, skimage_convention=True):
    """
    Rotate 2d-coordinates around their centroid.

    Parameters
    ----------
    x : float
        x-axis values of the input coordinates.
    y : float
        y-axis values of the input coordinates.
    angle : float
        Angle of rotation.
    skimage_convention : bool
        Whether to rotate counter-clockwise (skimage convention).

    Returns
    -------
    x : float
        x-axis values of the rotated coordinates.
    y : float
        y-axis values of the rotated coordinates.
    """
    # Move to origin before rotating
    x_mid = np.max(x) - round((np.max(x)-np.min(x))/2)
    y_mid = np.max(y) - round((np.max(y)-np.min(y))/2)
    x = [round(e-x_mid) for e in x]
    y = [round(e-y_mid) for e in y]

    # Scikit-image rotates counter-clockwise
    if skimage_convention:
        angle = -angle

    m = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    coords = [np.array([x[i],y[i]]) for i in range(len(x))]
    coords = [np.dot(m,e.T) for e in coords]

    # Return to original position
    x = [e[0]+x_mid for e in coords]
    y = [e[1]+y_mid for e in coords]
    return x,y

def rotate_to_major_axis(image, x, y, angle=None):
    """ Rotate image and landmark coordinates to major axis. """
    pts = [[x[i],y[i]] for i in range(len(x))]
    pca = PCA(n_components=2)
    pca.fit(pts)
    ang_0 = np.arctan2(pca.components_[1],np.array([0.,1.])) # atan of first PC w.r.t. y axis
    ang_0 = np.degrees(ang_0[1])
    
    x,y = rotate_coordinates(x,y, ang_0)
    x_center = np.min(x)+round((np.max(x)-np.min(x))/2)
    y_center = np.min(y)+round((np.max(y)-np.min(y))/2)
    image = rotate(image, ang_0, mode='edge', center=(x_center,y_center))
    return image,x,y

def im_warp(src_im, src_lm, dst_lm, corners=True):
    """
    Usage:
    x_src,y_src,im_src = im2lm(src_impath, src_ptspath, model)
    x_dst,y_dst,im_dst = im2lm(src_impath, src_ptspath, model)
    im_warped = im_warp(im_src, [x_src,y_src], [x_dst,y_dst])
    """
    
    src_lm = np.dstack(np.array(src_lm, dtype=float))[0]
    dst_lm = np.dstack(np.array(dst_lm, dtype=float))[0]
        
    if corners:
        corner_lm = [[0,0], [0,src_im.shape[0]], [src_im.shape[1],0], [src_im.shape[1],src_im.shape[0]]]
        src_lm  = np.concatenate([src_lm, corner_lm])
        dst_lm  = np.concatenate([dst_lm, corner_lm])
    
    warp_trans = skimage.transform.PiecewiseAffineTransform()
    warp_trans.estimate(dst_lm, src_lm)
    im_warped = skimage.transform.warp(src_im, warp_trans)
    return im_warped

def heatmaps2coords(heatmaps, threshold=.9, mode='centroid'):
    """ Returns coordinates of the max value in each of a list of heatmaps """
    x_list,y_list = [],[]
    for i in range(0,heatmaps.shape[2]):
        p = heatmaps[:,:,i]
        if mode == 'centroid':
            # Set the landmark coordinate as the centroid of the largest
            # connected component of the heatmap thresholded at 90%
            heatmap = rescale_intensity(heatmaps[:,:,i], out_range=(0,1))
            label_img = np.zeros(heatmap.shape, dtype=int)
            label_img[heatmap>threshold] = 1
            regions = regionprops(label_img)
            areas   = [props.area for props in regions] 
            y,x     = regions[np.argmax(areas)].centroid
        else:
            # Set the landmark coordinate as the position with the largest
            # value. Might be innacurate if there are several pixels with 
            # the largest value
            idx = p.argmax()
            y,x = np.unravel_index(p.argmax(), p.shape)
        x_list.append(x)
        y_list.append(y)
    return x_list,y_list

def spline_and_resample(x_in, y_in, n_points=None):
    """ Fits a cubic spline and resamples uniformly from it.

    Parameters
    ----------
    x_in : list of float
        x-axis values of the input coordinates.
    y_in : list of float
        y-axis values of the input coordinates.

    Returns
    -------
    x_out : list of float
        x-axis values of the coordinates samples from the spline.
    y_out : list of float
        y-axis values of the coordinates samples from the spline.

    Notes
    -----
    x_out and y_out have the same length as x_in and y_in.
    """

    assert len(x_in) == len(y_in)
    points = np.array([x_in,y_in]).T

    # Parameterize the curve w.r.t. to distance
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    distance = np.insert(distance, 0, 0)/distance[-1]

    # Fit spline
    spline_order = 3
    splines = [UnivariateSpline(distance, coords, k=spline_order, s=0.5*len(distance)) # s=None
              for coords in points.T]

    # Resample points from spline
    alpha       = np.linspace(0, 1, len(points))
    resampled   = np.vstack([spl(alpha) for spl in splines]).T
    x_out,y_out = zip(*resampled)
    return x_out,y_out

# class EarAnalyzer(object):
#     def __init__(self, ear_model_path, landmark_model_path):
#         self.parts = [slice(0,20),slice(20,35),slice(35,50),slice(50,55)]
#         self.detector_model = tf.keras.models.load_model(ear_model_path)
#         self.landmark_model = tf.keras.models.load_model(landmark_model_path)
    
#     def detect_ear(self, im):
#         pred = self.detector_model.predict(np.array([im]), verbose=0)[0]
#         return pred > threshold_otsu(pred)
       
#     def get_landmarks(self, img, n_realign=0, return_image=False):
#         def _flatten(l):
#             return [item for sublist in l for item in sublist]
                
#         if img.ndim < 4:
#             img = np.array([img])
        
#         img = np.array([self.get_ear_crop(img[0])])

        
#         # Predict and realign
#         img_aligned = img
#         for i in range(n_realign):
#             pred = self.landmark_model.predict(img_aligned, verbose=0)[-1]
#             x,y = heatmaps2coords(pred, mode='centroid')
#             img_aligned = rotate_to_major_axis(img_aligned[0], x, y)[0]
#             img_aligned = np.array([img_aligned])

#         # Predict landmarks for aligned image
#         pred = self.landmark_model.predict(img_aligned, verbose=0)[-1]
#         x,y = heatmaps2coords(pred, mode='centroid')

#         # Low-pass filter by parts
#         for i,p in enumerate(self.parts):
#             x[p],y[p] = [gaussian_filter1d(e, sigma=1) for e in [x[p],y[p]]]

#         # Resample from spline approximation
#         x,y = zip(*[spline_and_resample(np.array(x[p]), np.array(y[p])) for p in self.parts])
#         x,y = _flatten(x),_flatten(y)

#         ret = np.array(x, dtype=object),np.array(y, dtype=object)
#         if return_image:
#             ret = img_aligned[-1],*ret
#         return ret
    
#     def get_appearance(self, img, x_dst, y_dst, n_realign=0, corners=False, close_fit=False, histogram=False, bins=8):
#         img,x_src,y_src = self.get_landmarks(img, n_realign=n_realign, return_image=True)
#         warped = im_warp(img, [x_src, y_src], [x_dst,y_dst], corners=corners)

#         if close_fit:
#             h,w,c = warped.shape
#             x_min,x_max = int(np.min(x_dst)),int(np.max(x_dst))
#             y_min,y_max = int(np.min(y_dst)),int(np.max(y_dst))
#             warped = warped[y_min:y_max,x_min:x_max,:]
#             warped = resize(warped, (h,w))

#         if histogram:        
#             # Get non-zero values
#             mask = np.zeros(warped.shape, dtype=bool)
#             mask[warped > 0] = True

#             img_hsv = rgb2hsv(warped)
#             hue = np.histogram(img_hsv[:,:,0].flatten()[mask[:,:,0].flatten()], range=(0,1), bins=bins)[0]
#             sat = np.histogram(img_hsv[:,:,1].flatten()[mask[:,:,1].flatten()], range=(0,1), bins=bins)[0]
#             val = np.histogram(img_hsv[:,:,2].flatten()[mask[:,:,2].flatten()].flatten(), range=(0,1), bins=bins)[0]
#             return np.concatenate([hue,sat,val])
#         return warped
    
#     # --- 
    
#     def get_ear_crop(self, img, pad_width=((15,15),(15,15),(0,0)), mode='edge', output_size='same'):
#         r,c,l = img.shape
#         mask = label(dilation(self.detect_ear(img), ball(3)))
#         rprops = regionprops(mask)
#         r0,c0,l0,r1,c1,l1 = rprops[np.argmax([e.area for e in rprops])].bbox
#         cropped = img[r0:r1,c0:c1,:]
#         ret = np.pad(cropped, pad_width, mode=mode)
#         if output_size == 'same': ret = resize(ret, (r,c,l))
#         return ret
    
#     def feature_extractor(self, img, featext_fun, n_realign=0, close_fit=True, from_image=False, remove_background=False):
#         img_aligned,x,y = self.get_landmarks(
#             np.array([self.get_ear_crop(img)]),
#             n_realign=n_realign,
#             return_image=True
#         )
#         if from_image:
#             # ---
#             if remove_background:
#                 img_aligned = im_warp(img_aligned, [x,y], [x,y], corners=False)
                
#             if close_fit:
#                 h,w,c = img_aligned.shape
#                 x_min,x_max = int(np.min(x)),int(np.max(x))
#                 y_min,y_max = int(np.min(y)),int(np.max(y))
#                 img_aligned = img_aligned[y_min:y_max,x_min:x_max,:]
#                 img_aligned = resize(img_aligned, (h,w))
#             # plt.imshow(img_aligned)
#             # plt.show()
#             # ---
#             return featext_fun(img_aligned).flatten()
#         else:
#             return featext_fun([x,y]).flatten()
        
        
        
        
class EarAnalyzer(object):
    def __init__(self, ear_model_path, landmark_model_path):
        self.parts = [slice(0,20),slice(20,35),slice(35,50),slice(50,55)]
        self.detector_model = tf.keras.models.load_model(ear_model_path)
        self.landmark_model = tf.keras.models.load_model(landmark_model_path)
    
    def detect_ear(self, im):
        pred = self.detector_model.predict(np.array([im]), verbose=0)[0]
        return pred > threshold_otsu(pred)
       
    def get_landmarks(self, img_list, n_realign=0, return_image=False):
        def _flatten(l):
            return [item for sublist in l for item in sublist]
                
        if img_list.ndim < 4:
            img_list = np.array([img_list])
        
        crops = [self.get_ear_crop(e) for e in img_list]
        
        ret_list = []
        for e in crops:

            # Predict and realign
            img_aligned = np.array([e])
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
            x,y = _flatten(x),_flatten(y)

            ret = np.array(x, dtype=object),np.array(y, dtype=object)
            if return_image:
                ret = img_aligned[-1],*ret
            
            ret_list.append(ret)
        return ret_list
    
    def get_appearance(self, img, x_dst, y_dst, n_realign=0, corners=False, close_fit=False, histogram=False, bins=8):
        
        def crop(im):
            h,w,c = im.shape
            x_min,x_max = int(np.min(x_dst)),int(np.max(x_dst))
            y_min,y_max = int(np.min(y_dst)),int(np.max(y_dst))
            im = im[y_min:y_max,x_min:x_max,:]
            return resize(im, (h,w))
        
        def get_hsv(im):
            # Get non-zero values
            mask = np.zeros(im.shape, dtype=bool)
            mask[im > 0] = True

            img_hsv = rgb2hsv(im)
            hue = np.histogram(img_hsv[:,:,0].flatten()[mask[:,:,0].flatten()], range=(0,1), bins=bins)[0]
            sat = np.histogram(img_hsv[:,:,1].flatten()[mask[:,:,1].flatten()], range=(0,1), bins=bins)[0]
            val = np.histogram(img_hsv[:,:,2].flatten()[mask[:,:,2].flatten()].flatten(), range=(0,1), bins=bins)[0]
            return np.concatenate([hue,sat,val])
        
        # img,x_src,y_src = self.get_landmarks(img, n_realign=n_realign, return_image=True)
        img_aligned,x_src,y_src = zip(*self.get_landmarks(
            img,
            n_realign=n_realign,
            return_image=True
        ))
        
        warped = [im_warp(e, [x_src[i], y_src[i]], [x_dst,y_dst], corners=corners)
                  for i,e in enumerate(img_aligned)]

        if close_fit:
            warped = [crop(e) for e in warped]

        if histogram:
            hsv_hist = [get_hsv(e) for e in warped]
            return hsv_hist

        return warped
    
    # --- 
    
    def get_ear_crop(self, img, pad_width=((15,15),(15,15),(0,0)), mode='edge', output_size='same'):
        r,c,l = img.shape
        mask = label(dilation(self.detect_ear(img), ball(3)))
        rprops = regionprops(mask)
        r0,c0,l0,r1,c1,l1 = rprops[np.argmax([e.area for e in rprops])].bbox
        cropped = img[r0:r1,c0:c1,:]
        ret = np.pad(cropped, pad_width, mode=mode)
        if output_size == 'same': ret = resize(ret, (r,c,l))
        return ret
    
    def get_foreground(self, img):
        img  = self.get_ear_crop(img)
        mask = self.detect_ear(img)
        return img*mask
    
    def feature_extractor(self, img_list, featext_fun, n_realign=0, close_fit=True, from_image=False, remove_background=False, append_hsv=False):
        
        def _get_fit_crop(im,x,y):
            h,w,c = im.shape
            x_min,x_max = int(np.min(x)),int(np.max(x))
            y_min,y_max = int(np.min(y)),int(np.max(y))
            
            if ((x_max-x_min) < 10) or ((y_max-y_min) > 10):
                return im
            im = im[y_min:y_max,x_min:x_max,:]
            im = resize(im, (h,w))
            
            return im
        
        # print('Predicting landmarks')
        img_aligned,x,y = zip(*self.get_landmarks(
            img_list,
            n_realign=n_realign,
            return_image=True
        ))
        
        # print('Extracting features')
        if remove_background:
            img_aligned = [self.get_foreground(e) for e in tqdm.tqdm(img_aligned)]
            # img_aligned = [im_warp(e, [x[i],y[i]], [x[i],y[i]], corners=False) for i,e in tqdm.tqdm(enumerate(img_aligned))]
            
        if close_fit:
            img_aligned = [_get_fit_crop(e, x[i], y[i]) for i,e in enumerate(tqdm.tqdm(img_aligned))]
            
        if from_image:
            ret_list = [e.flatten() for e in tqdm.tqdm(featext_fun(np.array(img_aligned)))]
        else:
            ret_list = [featext_fun([x[i],y[i]]).flatten() for i in tqdm.tqdm(range(len(img_aligned)))]
        
        if append_hsv:
            hsvhist = self.get_appearance(
                    np.array(img_aligned),
                    x, y,
                    close_fit=close_fit,
                    histogram=True,
                    bins=256
            )
            concat_feature = [np.concatenate([ret_list[i], hsvhist[i]])
                for i in range(len(ret_list))]
            return concat_feature
            
#         for e in img_aligned:
#             plt.imshow(e)
#             plt.show()
            
        return ret_list