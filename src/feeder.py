import numpy as np

class Feeder:
    def __init__(self, featurespath, labelspath=None, opts={}):

        opts.update({'examplesratio': 0.95, })
        self.opts = opts

        if labelspath is None:
            from extractfeatures import FeaturesExtractor as fe
            featurespath, labelspath = fe.getdatapaths(featurespath)
        self.labelspath = labelspath
        self.featurespath = featurespath

        print('loading features')
        self.features = np.loadtxt(featurespath)
        print('loading labels')
        self.labels = np.loadtxt(labelspath)

        assert self.features.shape[0] == self.labels.shape[0]

        if 'featuremutation' in opts:
            print('processing features')
            tmpfeaturessize = self.features.shape[0]
            tmpfeatures = [None] * tmpfeaturessize
            featuressizeslist = [0] * tmpfeaturessize
            for i in range(self.features.shape[0]):
                tmpfeatures[i] = opts['featuremutation'](self.features[i])
                featuressizeslist[i] = tmpfeatures[i].shape[0]
            
            featuressize = sum(featuressizeslist)
            if featuressize > tmpfeaturessize:
                if len(tmpfeatures[0].shape) == 1:
                    self.features = np.zeros((featuressize,))
                else:
                    self.features = np.zeros((featuressize, tmpfeatures[0].shape[0]))

                featuressizeit = 0
                for i in range(tmpfeaturessize):
                    beg = featuressizeit
                    featuressizeit += featuressizeslist[i]
                    end = featuressizeit
                    self.features[beg:end] = tmpfeatures[i]
            else:
                self.features = np.array(tmpfeatures)

        if 'labelmutation' in opts:
            #here the parameters passed to labelmutation must be modified every time while a standard isn't found/fixed
            firstres = opts['labelmutation'](self.labels[0], 1)

            assert firstres.size == firstres.shape[0]

            tmplabels = np.zeros((self.labels.shape[0], firstres.size))
            if featuressize > tmpfeaturessize:
                featuressizeit = 0
                for i in range(self.labels.shape[0]):
                    beg = featuressizeit
                    featuressizeit += featuressizeslist[i]
                    end = featuressizeit
                    #here the parameters passed to labelmutation must be modified every time while a standard isn't found/fixed
                    tmplabels[beg:end] = opts['labelmutation'](self.labels[i], featuressizeslist[i])
            else:
                for i in range(self.labels.shape[0]):
                    #here the parameters passed to labelmutation must be modified every time while a standard isn't found/fixed
                    tmplabels[i] = opts['labelmutation'](self.labels[i], 1)
                self.labels = np.array(tmplabels)

        assert self.features.shape[0] == self.labels.shape[0]

        #prevents to give scalar instead of a vector
        if len(self.features.shape) == 1:
            self.features = np.array([[nb] for nb in self.features])
        if len(self.labels.shape) == 1:
            self.labels = np.array([[nb] for nb in self.labels])

        if 'batchsize' in opts:
            self.batchsize = opts['batchsize']
        else:
            self.batchsize = None
        self.nbsamples = self.features.shape[0]
        self.nbexamples = int(np.ceil(self.nbsamples * opts['examplesratio']))
        self.nbtests = self.nbsamples - self.nbexamples

        #random permutation of the samples in way to have an otpimal learning
        permut = np.random.permutation(self.nbsamples)
        self.features = self.features[permut]
        self.labels = self.labels[permut]

        self.examplemode = True

    def getbatch(self, batchfeatures=None, batchlabels=None, batchsize=None):
        if batchsize is None:
            batchsize = self.batchsize
        if batchsize is None:
            raise ValueError("You must provide a valid batchsize.")
        if self.examplemode:
            if batchsize > self.nbexamples:
                raise IndexError
            choice = np.random.choice(self.nbexamples, batchsize, False)
        else:
            if batchsize > self.nbtests:
                raise IndexError
            choice = np.random.choice(np.arrange(self.nbexamples, self.nbsamples), batchsize, False)
        batchfeatures, batchlabels = (self.features[choice], self.labels[choice])
        return (batchfeatures, batchlabels)

    def switchmode(self):
        self.examplemode = not self.examplemode

    def __call__(self, batchfeatures=None, batchlabels=None, batchsize=None):
        return self.getbatch(batchfeatures, batchlabels, batchsize)

    def __next__(self):
        if self.examplemode:
            choice = np.random.randint(self.nbexamples)
        else:
            choice = np.random.randint(self.nbexamples, self.nbsamples)
        return (self.features[choice], self.labels[choice])

    def __iter__(self):
        return self


class AudioFeeder(Feeder):
    def __init__(self, featurespath, labelspath=None, opts={}):
        import extractfeatures as ef
        opts.update({'featuremutation': ef.ExtractMonoAudioFiles.featuremutation, 'labelmutation': ef.ExtractMonoAudioFiles.labelmutation})
