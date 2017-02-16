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
        self.exampleit = 0
        self.testit = 0

    def getbatch(self, batchfeatures=None, batchlabels=None, batchsize=None):
        if batchsize is None:
            batchsize = self.batchsize
        if batchsize is None:
            raise ValueError("You must provide a valid batchsize.")
        if self.examplemode:
            if self.exampleit + batchsize > self.nbexamples:
                raise IndexError
            beg = self.exampleit
            self.exampleit += batchsize
            end = self.exampleit
        else:
            if self.testit + batchsize > self.nbtests:
                raise IndexError
            beg = self.nbexamples + self.testit
            self.testit += batchsize
            end = self.nbexamples + self.testit
        batchfeatures, batchlabels = (self.features[beg:end], self.labels[beg:end])
        return (batchfeatures, batchlabels)

    def switchmode(self):
        self.examplemode = not self.examplemode

    def __call__(self, batchfeatures=None, batchlabels=None, batchsize=None):
        return self.getbatch(batchfeatures, batchlabels, batchsize)

    def __next__(self):
        if self.examplemode:
            if self.exampleit >= self.nbexamples:
                raise StopIteration
            it = self.exampleit
            self.exampleit += 1
        else:
            if self.testit >= self.nbtests:
                raise StopIteration
            it = self.nbexamples + self.testit
            self.testit += 1
        return (self.features[it], self.labels[it])

    def __iter__(self):
        return self
