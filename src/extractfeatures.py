from os import listdir,remove
from os.path import join,isfile
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

def fileremoved(outfile):
    if isfile(outfile):
        if absolutepromt("File %s exists in filesystem. Do you want to erase it? (y/n) " % (outfile,)):
            remove(outfile)
        else:
            return False
    return True

class FeaturesExtractor:
    def savelabel(self, outfile, dat):
        pass

    def savefeature(self, outfile, dat):
        pass

    def __init__(self, inpath, outpath, computedatafunc, extractmetadatafunc):
        self.inpath = inpath
        self.computedatafunc = computedatafunc
        self.extractmetadatafunc = extractmetadatafunc
        self.outpath = outpath

    def __call__(self, inpath=None, outpath=None, computedatafunc=None, extractmetadatafunc=None):
        if inpath is None:
            inpath = self.inpath
        if computedatafunc is None:
            computedatafunc = self.computedatafunc
        if extractmetadatafunc is None:
            extractmetadatafunc = self.extractmetadatafunc
        if outpath is None:
            outpath = self.outpath
        self.extract(inpath, outpath, computedatafunc, extractmetadatafunc)

    def extract(self, inpath, outpath, computedatafunc, extractmetadatafunc):
        files = [f for f in listdir(inpath) if f.endswith('.txt')]

        nbinfiles = len(files)
        if nbinfiles <= 0:
            raise ValueError('you must provide a least one file.')

        metadats = extractmetadatafunc(inpath, files)
        nbfiles = len(metadats)
        loopit = 1
        self.metas = np.zeros(shape=(nbfiles, 2), dtype=int)
        for smd in metadats:
            self.__storedata(outpath, computedatafunc(inpath, smd), loopit)

            if loopit in [1, nbfiles] or not loopit % 100:
                print('loading file %d/%d' % (loopit, nbfiles))
            loopit += 1
        np.savetxt(join(outpath, 'meta.dat'), self.metas, header='nbitems', fmt='%i')

    def __storedata(self, outpath, data, it):
        self.metas[it-1, 0] = data[0].shape[0]

        outfile = join(outpath, 'feature_%d' % (it,))
        if fileremoved(outfile):
            self.savefeature(outfile, data[0])

        outfile = join(outpath, 'label_%d' % (it,))
        if fileremoved(outfile):
            self.savelabel(outfile, data[1])

class ExtractMonoAudioFiles(FeaturesExtractor):
    #static vars to be manually set
    ##the natural byterate of the examples here but we can consider modifying it
    sr = 44100
#TOCHANGE
    nblabels = 88
#    nblabels = 89
    batchsize = 1000
    #featurefunc = lambda y, sr: lrft.mfcc(y, sr).T
    #outpath = 'outdata/mfcc_20'
    #featurefunc = lambda y, sr: lrft.mfcc(y, sr, n_mfcc=5).T
    #outpath = 'outdata/mfcc_5'
    #featurefunc = lambda y, sr: lrft.mfcc(y, sr, n_mfcc=1).T
    #outpath = 'outdata/mfcc_1'
    featurefunc = lambda y, sr: lrft.melspectrogram(y, sr).T
    #outpath = 'outdata/melspectrogramwithfewsilence_128'
    outpath = 'outdata/melspectrogram_128'
    inpath = '../simple-wdir'
    
    #for feeder
    featuremutation = lambda y, sr: lrft.melspectrogram(y, sr).T

    @staticmethod
    def labelmutation(pitch, nbsamples):
        labelvect = np.zeros(shape=(nbsamples, ExtractMonoAudioFiles.nblabels))
        labelvect[:, int(pitch)] = np.ones(nbsamples)
        return labelvect

    def __init__(self, inpath=None, outpath=None):
        if inpath is None:
            self.inpath = ExtractMonoAudioFiles.inpath
        else:
            self.inpath = inpath

        if outpath is None:
            self.outpath = ExtractMonoAudioFiles.outpath
        else:
            self.outpath = outpath
        self.savefeature = np.savetxt


        super().__init__(self.inpath, self.outpath, ExtractMonoAudioFiles.computefeatureddata, ExtractMonoAudioFiles.extractsamplesmetadata)

    @staticmethod
    def extractsamplesmetadata(path, filelist):
        samplesmetadata = []

        for f in filelist:
            with open(join(path, f), 'r') as fs:
                #get the the onset and offset plus midi pitch
                onset, offset, midipitch = tuple(map(float, fs.readlines()[1][:-1].split('\t')))
            #first is the midi pitch rescaled into [0,87] range
            #second is a tuple of wav path, onset and duration
#TOCHANGE
#            samplesmetadata.append(((f[:-3] + 'wav', 0, onset), 88))
            samplesmetadata.append(((f[:-3] + 'wav', onset, offset-onset), int(midipitch - 21)))
            #samplesmetadata.append(((f[:-3] + 'wav', offset, None), 89))
        return samplesmetadata

    @staticmethod
    def computefeatureddata(path, samplemetadata):
        meta, pitch = samplemetadata

        #audiodat0 = lrco.load(join(path, meta[0]), sr= ExtractMonoAudioFiles.sr,
        #                     duration=meta[1] + meta[2])
        audiodat = lrco.load(join(path, meta[0]), sr= ExtractMonoAudioFiles.sr,
                             offset=meta[1], duration=meta[2])
        #try:
        #    assert audiodat0[0].shape[0] == int(np.ceil(ExtractMonoAudioFiles.sr * (meta[1] + meta[2])))
        #    assert audiodat[0].shape[0] == int(np.ceil(ExtractMonoAudioFiles .sr * meta[2]))
        #except AssertionError:
        #    print(audiodat0[0].shape[0])
        #    print(int(np.ceil(ExtractMonoAudioFiles.sr * (meta[1] + meta[2]))))
        #    print(audiodat[0].shape[0])
        #    print(int(np.ceil(ExtractMonoAudioFiles .sr * meta[2])))
        #    print()

        audiodat = ExtractMonoAudioFiles.featurefunc(*audiodat)

        #pitchvect = np.array([pitch] * audiodat.shape[0])

        return (audiodat, pitch)
        #return (audiodat, pitchvect)

    def savelabel(self, outfile, dat):
        with open(outfile, 'w') as fs:
            fs.write(str(dat))

if __name__ == '__main__':
    if len(sys.argv) > 2:
        ex = ExtractMonoAudioFiles(sys.argv[1])
    else:
        ex = ExtractMonoAudioFiles()
    ex()
