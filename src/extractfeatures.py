from os import listdir,remove
from os.path import join,isfile
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

class FeaturesExtractor:
    #static vars to be manually set
    ##the natural byterate of the examples here but we can consider modifying it
    sr = 44100
    nblabels = 88
    batchsize = 1000
    #featurefunc = lrft.melspectrogram
    featurefunc = lambda y, sr: lrft.mfcc(y, sr, n_mfcc=1)
    workingpath = '../simple-wdir'

    def __init__(self, inpath=None, featurefunc=None, outpath=None):
        if inpath is not None:
            self.workingpath = inpath
        if featurefunc is not None:
            self.featurefunc = featurefunc
        self.outpath = outpath

    def __call__(self, inpath=None, featurefunc=None, outpath=None):
        if inpath is None:
            inpath = self.workingpath
        if featurefunc is None:
            featurefunc = self.featurefunc
        if outpath is None:
            outpath = self.outpath
        FeaturesExtractor.extract(inpath, featurefunc, outpath)

    @staticmethod
    def extract(inpath=None, featurefunc=None, outpath=None):
        if inpath is None:
            inpath = FeaturesExtractor.workingpath
        if featurefunc is None:
            featurefunc = FeaturesExtractor.featurefunc
        if outpath is None:
            outpath = inpath

        outdatapaths = FeaturesExtractor.getdatapaths(outpath)
        for odp in outdatapaths:
            if isfile(odp):
                if absolutepromt("File %s exists in filesystem. Do you want to erase it? (y/n) " % (odp,)):
                    ##erase file
                    #with open(odp, 'w') as fs:
                    #    pass

                    #remove file; will be recreated after
                    remove(odp)
                else:
                    print("Then please consider resolving this file conflict before relaunching.")
                    return

        files = [f for f in listdir(inpath) if f.endswith('.txt')]

        nbfiles = len(files)
        if nbfiles <= 0:
            raise ValueError('you must provide a least one file.')

        loopit = 1
        for smd in FeaturesExtractor.__extractsamplesmetadata(inpath, files):
            FeaturesExtractor.__storedata(outdatapaths,
                                          FeaturesExtractor.__computefeatureddata(inpath, featurefunc, smd))

            if loopit in [1, nbfiles] or not loopit % 100:
                print('loading file %d/%d' % (loopit, nbfiles))
            loopit += 1

    @staticmethod
    def __computefeatureddata(path, featurefunc, samplemetadata):
        meta, pitch = samplemetadata

        audiodat = lrco.load(join(path, meta[0]), sr=FeaturesExtractor.sr,
                             offset=meta[1], duration=meta[2])
        audiodat = featurefunc(*audiodat).T

        pitchvect = np.zeros(shape=(audiodat.shape[0], FeaturesExtractor.nblabels))
        pitchvect[:, pitch] = np.ones(audiodat.shape[0])

        return (audiodat, pitchvect)

    @staticmethod
    def __storedata(outpaths, data):
        assert len(outpaths) == len(data) and len(data) == 2

        for i in range(2):
            with open(outpaths[i], 'ab') as fs:
                np.savetxt(fs, data[i])

    @staticmethod
    def getdatapaths(outpath):
        return (join(outpath, 'features.dat'), join(outpath, 'labels.dat'))

    @staticmethod
    def __extractsamplesmetadata(path, filelist):
        samplesmetadata = []
        for f in filelist:
            with open(join(path, f), 'r') as fs:
                #get the the onset and offset plus midi pitch
                onset, offset, midipitch = tuple(map(float, fs.readlines()[1][:-1].split('\t')))
            #first is the midi pitch rescaled into [0,87] range
            #second is a tuple of wav path, onset and duration
            samplesmetadata.append(((f[:-3] + 'wav', onset, offset-onset), int(midipitch - 21)))
        return samplesmetadata

if __name__ == '__main__':
    if len(sys.argv) > 2:
        FeaturesExtractor.extract(sys.argv[1])
    else:
        FeaturesExtractor.extract()
