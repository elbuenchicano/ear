import numpy as np
import tensorflow as tf
import scipy
import pandas as pd
import skimage
import matplotlib.pyplot as plt

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from DBs import *

import models
from feature_extraction import InterpretableFeatureExtractor
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
def earfeatures(general, individual):
    ''' Initializing datasets
    ''' 
    path        = general['prefix_path'][general['path_op']]
    directory   = path + '/' + general['directory']
    
    #.....................................................................................


    
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
def initdb(general, individual):
    ''' Initializing datasets
    ''' 
    path        = general['prefix_path'][general['path_op']]
    directory   = path + '/' + general['directory']
    
    #.....................................................................................
    datasets    = individual['datasets']
    to_init     = individual['to_init']

    #.....................................................................................
    for db in to_init:
        db_path         = datasets[db]
        dbInit(db, db_path, directory)
