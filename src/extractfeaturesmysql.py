from os import listdir,remove
from os.path import join,isfile
from mysqlstuffs import Database
from six import iteritems
import librosa.feature as lrft
import librosa.core as lrco
import numpy as np
import sys

def absolutepromt(prompt="Enter y/n?",choices=["Y","y","n","N"],yeschoices=['Y','y'],error="Invalid choice"):
    while True:
        result = input(prompt)
        if result in choices:
            return result in yeschoices
        print(error)

class ExtractMonoAudioFiles:
    #static vars to be manually set
    ##the natural byterate of the examples here but we can consider modifying it
    sr = 44100
    nblabels = 88
    batchsize = 1000
    #featurefunc = lambda y, sr: lrft.mfcc(y, sr, n_mfcc=1).T
    #featurefunc = lambda *x: x
    #featurefunc = lambda y, sr: lrft.melspectrogram(y, sr).T
    featurefunc = lrft.melspectrogram
    inpath = '../simple-wdir'
    outdb = 'psc'
    
    ## for feeder
    featuremutation = lambda y, sr: lrft.melspectrogram(y, sr).T

#    @staticmethod
#    def labelmutation(pitch):
#        labelvect = np.zeros(shape=(1, ExtractMonoAudioFiles.nblabels))
#        labelvect[:, int(pitch)] = np.ones(nbsamples)
#        return labelvect
    ##

    def __init__(self, inpath=None, outdb=None):
        if inpath is None:
            self.inpath = ExtractMonoAudioFiles.inpath
        else:
            self.inpath = inpath
        if outdb is None:
            self.outdb = ExtractMonoAudioFiles.outdb
        else:
            self.outdb = outdb
        self.tablecontext = self.featurefunc.__name__
#        if featurefunc is None:
#            self.featurefunc = ExtractMonoAudioFiles.featurefunc
#        else:
#            self.featurefunc = featurefunc

    def __call__(self, inpath=None, outdb=None):
        if inpath is None:
            inpath = self.inpath
        if outdb is None:
            outdb = self.outdb
        self.extract(inpath, outdb)

    def extract(self, inpath, outdb):
        files = [f for f in listdir(inpath) if f.endswith('.txt')]

        nbfiles = len(files)
        if nbfiles <= 0:
            raise ValueError('you must provide a least one file.')

        loopit = 1
        for smd in self.__extractmetadata(inpath, files):
            self.__storedata(outdb, self.__computedata(inpath, smd), loopit)

            if loopit in [1, nbfiles] or not loopit % 100:
                print('loading file %d/%d' % (loopit, nbfiles))
            loopit += 1

    def __extractmetadata(self, path, filelist):
        samplesmetadata = []

        for f in filelist:
            with open(join(path, f), 'r') as fs:
                #get the the onset and offset plus midi pitch
                onset, offset, midipitch = tuple(map(float, fs.readlines()[1][:-1].split('\t')))
            #first is the midi pitch rescaled into [0,87] range
            #second is a tuple of wav path, onset and duration
            samplesmetadata.append(((f[:-3] + 'wav', onset, offset-onset), int(midipitch - 21)))
        return samplesmetadata

    def __computedata(self, path, samplemetadata):
        meta, pitch = samplemetadata

        audiodat = lrco.load(join(path, meta[0]), sr=self.sr,
                             offset=meta[1], duration=meta[2])
        audiodat = ExtractMonoAudioFiles.featurefunc(*audiodat).T

#        pitchvect = np.array([pitch] * audiodat.shape[0])

        #return (audiodat, pitch)
        return {'features': audiodat, 'label': pitch}

    def __storedata(self, outdb, data, sampleid):
        pass
#        assert len(outpaths) == len(data) and len(data) == 2
#        datasize = len(data)
        db = Database(outdb)
        #TODO: check if glue corresponds to new set of tables and alter it if not and check every table so and etc
        #create table if not exists
#        if not db.istable('glue'):
#            glue = {'id': 'INT(11) NOT NULL AUTO_INCREMENT'}
#            for k,v in iteritems(data):
#                tmpstruct = {'id': 'INT(11) NOT NULL AUTO_INCREMENT'}
#                if len(v.shape) > 1:
#                    nbfeat = v.shape[1]
#                else:
#                    nbfeat = 1
#                for i in range(nbfeat):
#                    tmpstruct['item_' + str(i+1)] = 'FLOAT DEFAULT NULL'
#                tmpstruct['PRIMARY KEY'] = '(id)'
#                db.createtable(k, tmpstruct)
#                glue[k+'_id'] = 'INT(11) DEFAULT NULL'
#
#            glue['PRIMARY KEY'] = '(id)'
#            db.createtable('glue', glue)
        features = data['features']
        label = data['label']

        if not db.istable(self.tablecontext):
            tmpstruct = {'id': 'INT(11) NOT NULL AUTO_INCREMENT',
                         'label_id': 'INT(11) DEFAULT NULL',
                         'sample_id': 'INT(11) DEFAULT NULL'}
            if len(features.shape) > 1:
                nbfeat = features.shape[1]
            else:
                nbfeat = 1

            for i in range(nbfeat):
                tmpstruct['feature_' + str(i+1)] = 'FLOAT DEFAULT NULL'
            tmpstruct['PRIMARY KEY'] = '(id)'
            db.createtable(self.tablecontext, tmpstruct)

        #fill table
        rowsnb = features.shape[0]
        if len(features.shape) > 1:
            colsnb = features.shape[1]
        else:
            colsnb = 1

        formateddatas = []
        for row in range(rowsnb):
            formatedrow = {'label_id': label, 'sample_id': sampleid}
            if colsnb > 1:
                for col in range(colsnb):
                    formatedrow['feature_'+str(col+1)] = features[row,col]
            else:
                formatedrow['feature_1'] = features[row]
            formateddatas.append(formatedrow)
        db.insert(self.tablecontext, formateddatas)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        ex = ExtractMonoAudioFiles(*tuple(sys.argv[1:]))
    else:
        ex = ExtractMonoAudioFiles()
    ex()
