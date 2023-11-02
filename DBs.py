import  os
 
import  matplotlib.pyplot   as      plt
import  pandas              as      pd       

from    utils               import  *
from    tqdm                import  tqdm
import  numpy as np
import  tensorflow as tf
import  scipy
import  pandas as pd
import  skimage
import  matplotlib.pyplot as plt

from    tensorflow.compat.v1 import ConfigProto
from    tensorflow.compat.v1 import InteractiveSession

import  models
from    feature_extraction import InterpretableFeatureExtractor


# Ensure reproducibility
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

tf_config = ConfigProto()
tf_config.gpu_options.allow_growth = True
session = InteractiveSession(config=tf_config)

def switchBase(data, str1, str2):
    for i in data:
        data[i].replace(str1, str2)

###################################################################################################
###################################################################################################
###################################################################################################
def dbInit(name, db_pt, local_pt):
    '''It selects the dataset'''
    dbs = {
        "Cox"        : CoxDbEar
    }
    local_pt    += '/data/db/' + name 
    db          = dbs[name]()
    db.getConfs(db_pt, local_pt)

###################################################################################################
###################################################################################################
###################################################################################################
def dbSelect(name, local_pt, db_confs):
    '''It selects the dataset'''
    dbs = {
        "Cox"        : CoxDbEar,
        }
    local_pt    += '/data/db/' + name 
    return dbs[name](local_pt, db_confs)

################################################################################
################################################################################
################################################################################
class CoxDbEar():
    ''' This class must be initialized first with initdb function
        confs must be in data folder
    ''' 
    def __init__(self, local_pt=None, db_confs=None):
        
        if local_pt == None: return
        
        paths           = u_loadJson(local_pt + '/cox_ear_paths.json')
        self.db_pt      = paths['db_pts'][u_whichOS()]
        self.transforms = db_confs['transforms']
       
        # items are arranged in this function 
        modes       = { "byset": self._byset }                  
        
        #.......................................................................
        mode    = db_confs["mode"]
        modes[mode](paths, db_confs[mode], local_pt)
    #..............................................................................................
    #..............................................................................................
    def __getitem__(self, index):
        id, pt, bb  = self.items[index]
        im  = read_image(u_joinPath([self.db_pt, pt]))
       
        
        return im, id
        
    #..............................................................................................
    #..............................................................................................
    def __len__(self):
        return self.items.shape[0]
    #..............................................................................................
    #..............................................................................................
    def _byset(self, paths, confs, local_pt):
        sets    = confs[confs['set']] # set must be inserted manually by code
        arrs    = []
        db_pt   = paths['db_pts'][u_whichOS()]

        for s in sets:
            pt = local_pt + paths[s + '_bbox_pt']
            arr = pd.read_json(pt).to_numpy()
            arrs.append(arr)
            
        self.items  = np.concatenate(arrs, axis=0)

    #..............................................................................................
    #..............................................................................................
    def getConfs(self, db_pts, local_pt):
        confs           = {}
        db_pt           = db_pts[u_whichOS()]

        ## important local files 
        conf_pt         = '/cox_ear_paths.json'
        item_pt         = '/item_list.json'   
        ids_int_pt      = '/ids_int.json'  
        wbdet_pt        = '/wbdet.json'  
        
        st_bbox_pt      = '/st_bbox.json'  # pandas format
        c1_bbox_pt      = '/c1_bbox.json'
        c2_bbox_pt      = '/c2_bbox.json'
        c3_bbox_pt      = '/c3_bbox.json'
        
        c1_frames_pt    = '/c1_frames.json'
        c2_frames_pt    = '/c2_frames.json'
        c3_frames_pt    = '/c3_frames.json'

        #......................................................................................
        # item list loading
        print('Local path is :', local_pt)
        u_mkdir(local_pt)
        if not os.path.isfile(local_pt + item_pt):
            item_list   = u_listFileAllDic_2(db_pt, '', 'jpg')
            u_saveDict2File(local_pt + item_pt, local_pt + item_list)

        #......................................................................................
        ## bbox list loading
        wrong_det   = [] 
        # device      = "cuda" if torch.cuda.is_available() else "cpu"
        
        item_list   = u_loadJson(local_pt + item_pt)

        #......................................................................................
        
        #ids_int     = self.__still(item_list, db_pt, wrong_det, 
        #                           local_pt + ids_int_pt, local_pt + st_bbox_pt)

        ids_int = u_loadJson('ids_int.json')

        self.__cameras(item_list, db_pt, wrong_det, ids_int, 
                       [local_pt + i for i in [c1_bbox_pt, c2_bbox_pt, c3_bbox_pt]], 
                       [local_pt + i for i in [c1_frames_pt, c2_frames_pt, c3_frames_pt]])

        #u_saveList2File(local_pt + wbbox_pt, wrong_bbox)

        #.........................................
        #.........................................
        confs['item_pt'     ] = item_pt
        confs['ids_int_pt'  ] = ids_int_pt
        confs['wbbox_pt'    ] = wbbox_pt
        confs['st_bbox_pt'  ] = st_bbox_pt
        confs['c1_bbox_pt'  ] = c1_bbox_pt
        confs['c2_bbox_pt'  ] = c2_bbox_pt
        confs['c3_bbox_pt'  ] = c3_bbox_pt
        confs['c1_frames_pt'] = c1_frames_pt
        confs['c2_frames_pt'] = c2_frames_pt
        confs['c3_frames_pt'] = c3_frames_pt
        confs['db_pts'      ] = db_pts
        
        u_saveDict2File(local_pt + conf_pt, confs)

    #..........................................................................................
    #..........................................................................................
    def __getFeatures(self, img_pt, feature_extractor, wrong_det): 
        input_image = skimage.io.imread(img_pt)
        input_image = skimage.transform.resize(input_image, (1024,1024))

        plt.imshow(input_image)
        plt.show()
        
        norm_image,x,y = feature_extractor._get_landmarks(input_image, return_image=True)
        plt.imshow(norm_image)
        plt.plot(x,y,'.')
        plt.axis(False)
        plt.show()

        #feature = feature_extractor.get_features(np.array([input_image]))

        #plt.matshow(np.reshape(feature, (55,55)))
        #plt.show()

        #edm = np.reshape(feature, (55,55))
        #features_compact = edm[np.triu_indices_from(edm, k=1)]

        #print(f'Full matrix size: {len(edm.flatten())}')
        #print(f'Compacted size:   {len(features_compact)}')

        return []
        return features_compact

    #..........................................................................................
    #..........................................................................................
    def __still(self, item_list, db_pt, wrong_det, ids_int_pt, st_bbox_pt):
        '''Getting feats for still folder 
        '''
        ids_int     = {}
        still_files = item_list['still']['_files']
        still_pt    = item_list['still']['_path']

        ids         = []
        pts         = []
        bbox        = []
        
        #.........................................................................................
        ear_detector_path       = 'ear_detector.h5'
        landmark_detector_path  = 'landmark_detector.h5'

        ear_detector_model      = tf.keras.models.load_model(ear_detector_path)
        landmark_detector_model = tf.keras.models.load_model(landmark_detector_path)
        feature_extractor       = InterpretableFeatureExtractor(ear_detector_model, 
                                                                landmark_detector_model, 
                                                                append_color=False)


        #.........................................................................................                
        for it, pt in enumerate(still_files):
            id          = pt[:12]
            ids_int[id] = it
        
            bb      = self.__getFeatures(db_pt + still_pt + '/'+ pt, feature_extractor, wrong_det)
            
            if len(bb):
                ids.append(it)
                pts.append(still_pt +'/'+ pt)
                bbox.append(bb)
            else:
                pass
    
        #......................................................................................
        df = pd.DataFrame({'id':ids, 'pt_file':pts, 'bbox':bbox})
        df.to_json(st_bbox_pt) 
        u_saveDict2File(ids_int_pt, ids_int)
        return ids_int
    #..........................................................................................
    #..........................................................................................
    def __cameras(self, item_list, db_pt, wrong_det, ids_int, c_bbox_pts, c_frames_pts):
        # for each camera
        for i, camera_pt in enumerate(c_bbox_pts):
            cam     = 'cam' + str(i+1) 
            ids     = []
            pts     = []
            bbox    = []
            frames  = {}
            p_frm   = 0

            #.........................................................................................
            ear_detector_path       = 'ear_detector.h5'
            landmark_detector_path  = 'landmark_detector.h5'

            ear_detector_model      = tf.keras.models.load_model(ear_detector_path)
            landmark_detector_model = tf.keras.models.load_model(landmark_detector_path)
            feature_extractor       = InterpretableFeatureExtractor(ear_detector_model, 
                                                                    landmark_detector_model, 
                                                                    append_color=False)

            #.........................................................................................                

            for id in ids_int:
                frames[ids_int[id]] = []
                for file in item_list['video'][cam][id]['_files'][::-1]:
                    path    = item_list['video'][cam][id]['_path']
                    

                    bb      = self.__getFeatures(db_pt + path + '/'+ file, feature_extractor, wrong_det)

                    if len(bb):
                        ids.append(ids_int[id])
                        pts.append(path + '/'+ file)
                        bbox.append(bb)
                        frames[ids_int[id]].append(p_frm)
                        p_frm += 1
                    else:
                        frames[ids_int[id]].append(-1)


                    break 
                print(f'camera: {i+1}, id: {ids_int[id]}')

            df = pd.DataFrame({'id':ids, 'pt_file':pts, 'bbox':bbox})
            df.to_json(camera_pt) 
            u_saveDict2File(c_frames_pts[i], frames)

