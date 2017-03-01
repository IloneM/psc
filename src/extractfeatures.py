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
    def __init__(self, inpath, computedatafunc, extractmetadatafunc, outpath=None):
        self.inpath = inpath
        self.computedatafunc = computedatafunc
        self.extractmetadatafunc = extractmetadatafunc
        self.outpath = outpath

    def __call__(self, inpath=None, computedatafunc=None, extractmetadatafunc=None, outpath=None):
        if inpath is None:
            inpath = self.inpath
        if computedatafunc is None:
            computedatafunc = self.computedatafunc
        if extractmetadatafunc is None:
            extractmetadatafunc = self.extractmetadatafunc
        if outpath is None:
            outpath = self.outpath
        FeaturesExtractor.extract(inpath, computedatafunc, extractmetadatafunc, outpath)

    @staticmethod
    def extract(inpath, computedatafunc, extractmetadatafunc, outpath=None):
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
        for smd in extractmetadatafunc(inpath, files):
            FeaturesExtractor.__storedata(outdatapaths, computedatafunc(inpath, smd))

            if loopit in [1, nbfiles] or not loopit % 100:
                print('loading file %d/%d' % (loopit, nbfiles))
            loopit += 1

    @staticmethod
    def __storedata(outpaths, data):
        assert len(outpaths) == len(data) and len(data) == 2

        for i in range(2):
            with open(outpaths[i], 'ab') as fs:
                np.savetxt(fs, data[i])

    @staticmethod
    def getdatapaths(outpath):
        return (join(outpath, 'features.dat'), join(outpath, 'labels.dat'))

class ExtractMonoAudioFiles(FeaturesExtractor):
    #static vars to be manually set
    ##the natural byterate of the examples here but we can consider modifying it
    sr = 44100
    nblabels = 88
    batchsize = 1000
    #featurefunc = lambda y, sr: lrft.mfcc(y, sr, n_mfcc=1).T
    featurefunc = lambda x: x
    inpath = '../simple-wdir'
    
    #for feeder
    featuremutation = lambda y, sr: lrft.melspectrogram(y, sr).T

    @staticmethod
    def labelmutation(pitch, nbsamples):
        labelvect = np.zeros(shape=(nbsamples, ExtractMonoAudioFiles.nblabels))
        labelvect[:, pitch] = np.ones(nbsamples)


    def __init__(self, inpath=None):
        if inpath is None:
            self.inpath = ExtractMonoAudioFiles.inpath
        else:
            self.inpath = inpath
#        if featurefunc is None:
#            self.featurefunc = ExtractMonoAudioFiles.featurefunc
#        else:
#            self.featurefunc = featurefunc
        super().__init__(self.inpath, ExtractMonoAudioFiles.computefeatureddata, ExtractMonoAudioFiles.extractsamplesmetadata)

    @staticmethod
    def extractsamplesmetadata(path, filelist):
        samplesmetadata = []

        for f in filelist:
            with open(join(path, f), 'r') as fs:
                #get the the onset and offset plus midi pitch
                onset, offset, midipitch = tuple(map(float, fs.readlines()[1][:-1].split('\t')))
            #first is the midi pitch rescaled into [0,87] range
            #second is a tuple of wav path, onset and duration
            samplesmetadata.append(((f[:-3] + 'wav', onset, offset-onset), int(midipitch - 21)))
        return samplesmetadata

    @staticmethod
    def computefeatureddata(path, samplemetadata):
        meta, pitch = samplemetadata

        audiodat = lrco.load(join(path, meta[0]), sr= ExtractMonoAudioFiles.sr,
                             offset=meta[1], duration=meta[2])
        audiodat = ExtractMonoAudioFiles.featurefunc(*audiodat)

        pitchvect = np.array([pitch] * audiodat.shape[0])

        return (audiodat, pitchvect)

if __name__ == '__main__':
    if len(sys.argv) > 2:
        ex = ExtractMonoAudioFiles(sys.argv[1])
    else:
        ex = ExtractMonoAudioFiles()
    ex()
