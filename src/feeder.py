import numpy as np

class Feeder:
    def __init__(self, featurespath, labelspath=None, examplesratio=0.95, batchsize=None):
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

        #prevents to give scalar instead of a vector
        if len(self.features.shape) == 1:
            self.features = np.array([[nb] for nb in self.features])
        if len(self.labels.shape) == 1:
            self.labels = np.array([[nb] for nb in self.labels])

        self.batchsize = batchsize
        self.nbsamples = self.features.shape[0]
        self.nbexamples = int(np.ceil(self.nbsamples * examplesratio))
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
