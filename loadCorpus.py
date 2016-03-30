import os
import json

import numpy as np
from gensim.models import Doc2Vec
from tqdm import tqdm
import logging

def loadModel(dim=600):
    '''
    Read model from disk. Assumes files are in a subdir called cache. 
    Conventions for the files is allDocs<DIM>D.model, where DIM is the
    dimensionality of the embedding. DIM is determined by the parameter 
    dim. 600 is the default dimensionality.
    '''
    logging.getLogger().setLevel(logging.INFO)

    if(dim not in (100, 300, 600)):
        raise ValueError('dim must be 100, 300 or 600')
                        
    modelName = 'allDocs' + str(dim) + 'D.model'
    modelBasePath = 'cache'
    modelPath = os.path.join(os.getcwd(), modelBasePath, modelName)

    logging.info('start loading the model')
    model = Doc2Vec.load(modelPath)
    logging.info('loading completed')

    return model

def loadCorpus(dim=600, model=None, regression=False):
    '''
    Returns corpus prepared in a way appropriate for gensim/scipy models 
    to fit to. The first element of the tuple returned is a matrix of numpy
    integers representing the doc embedding. The second elements indicates 
    whether a document was cited at all. If regression is set to True it will 
    be represented by either [0,1] (was not cited) or [1,0] (was cited). 
    Otherwise it is 0 and 1.
    '''
    JSONFILESDIR = 'data/json'
    X = []
    y = []
    if not isinstance(model, Doc2Vec):
        model = loadModel(dim=dim)
                    
    if regression:
        zero = np.array([0,1], dtype=np.int32)
        one = np.array([1,0], dtype=np.int32)
    else:
        zero = np.int32(0)
        one = np.int32(1)

    logging.info('building corpus...')
    filenames = model.docvecs.doctags.keys()
    for k in tqdm(filenames):
        with open(os.path.join(JSONFILESDIR, k + '.json')) as fh:
            jsonFile = json.load(fh)

        if jsonFile['lang'] != 'en' or jsonFile['citedBy'] is None:
            logging.debug('{f} discarded from corpus.'.format(f=k))
            continue

        X.append(model.docvecs[k])
        isSuccessfull = one if int(jsonFile['citedBy']) > 0 else zero
        y.append(isSuccessfull)
        logging.debug('{f} absorbed into corpus.'.format(f=k))

    # transform to numpy arrays
    X = np.array(X)
    y = np.array(y)

    logging.info('corpus complete')
    return X, y

