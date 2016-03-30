import json
import os
from os.path import isfile, join
from scandir import scandir
import re
from textblob import TextBlob
from nltk.corpus import stopwords
from gensim.models.doc2vec import LabeledSentence
import multiprocessing
from gensim.models.doc2vec import TaggedDocument
from concurrent.futures import ProcessPoolExecutor
from collections import OrderedDict
import logging

logging.getLogger().setLevel(logging.INFO)

class loadItems(object):
    def __init__(self, wd = 'samples'):
        self.wd = wd

    def __iter__(self):
        for doc in scandir(self.wd):
            if doc.is_file() and doc.name.endswith('.json'):
                with open(os.path.join(self.wd, doc.name), 'r') as f:
                    yield json.load(f) 


def normalize(text, lang='en'):
    # remove some symbols
    text = re.sub(r'[,.;:]', r' ', text)

    # stopword removal
    langMapping = [('de', 'german'), ('en', 'english'), ('es', 'spanish'), ('fr', 'french')]
    langFound = None
    for map in langMapping:
        if lang == map[0]:
            langFound = map[1]
        if langFound is not None:
            wordlist = [w for w in text.split() if w not in stopwords.words(langFound)]
            text = " ".join(wordlist)

        # drop too long and too short words
        lower = 4
        upper = 20
        wordlist = [w for w in text.split() if len(w) >= lower and len(w) < upper]
        text = " ".join(wordlist)   

        # singularize
        try:
            wordlist = TextBlob(text)
            text = ' '.join(wordlist.words.singularize())
        except:
            pass

        return text


class LabeledDocument(object):
    def __init__(self, lang='en', normalizeText=True, workers=1, wd='samples'):
        self.docGenerator = loadItems(wd=wd)
        self.lang = lang
        self.normalizeText = normalizeText
        self.workers = workers
            
    def mkDir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
            logging.debug(dir + ' has been created.')
                    
    def __iter__(self):
        p = ProcessPoolExecutor(self.workers)
        cacheDir = os.path.join('cache', 'normalizedDocs')
        cacheDict = {}
        futures = OrderedDict()
            
        if os.path.exists(cacheDir):
            keySet = set(os.listdir(cacheDir))
            logging.debug('Read keySet from cache directory')
        else:
            keySet = set()
            self.mkDir(cacheDir)
            logging.debug('Cache is empty. Begin with empty keySet')
                    
        for doc in self.docGenerator:
            if 'lang' not in doc or 'filename' not in doc \
            or doc['lang'] != self.lang or 'plaintext' not in doc:
                logging.debug('Omitting document. Important parameters missing')
                continue
                        
            text = doc['plaintext']
            filename = doc['filename']
                        
            if not self.normalizeText:
                yield TaggedDocument(words=text.split(), tags=filename)
            else:
                if filename in keySet and os.path.exists(os.path.join(cacheDir, filename)):
                    # the file has already been normalized, let's
                    # read the cache
                    with open(os.path.join(cacheDir, filename)) as fh:
                        logging.debug('Yielded from Cache')
                        yield LabeledSentence(words=json.load(fh).split(), tags=[filename])
                else:
                    futures[filename] = p.submit(normalize, text, lang=doc['lang'])

            if self.normalizeText:
                for k, v in futures.items():
                    v = v.result()
                    keySet.add(k)
                    with open(os.path.join(cacheDir, k), 'w') as fh:
                        json.dump(v, fh)
                        logging.debug('Yielded from Calculation')
                        yield LabeledSentence(words=v.split(), tags=[k])
