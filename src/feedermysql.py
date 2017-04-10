import numpy as np
from extractfeaturesmysql import ExtractMonoAudioFiles as emaf
from mysqlstuffs import Database

class Feeder:
    def __init__(self, opts={}):
        opts.update({'examplesratio': 0.95, 'dbname': emaf.outdb})
        self.opts = opts

        self.db = Database(self.opts['dbname'])

#        if labelspath is None:
#            from extractfeatures import FeaturesExtractor as fe
#            featurespath, labelspath = fe.getdatapaths(featurespath)
#        self.labelspath = labelspath
#        self.featurespath = featurespath

#        if len(self.features.shape) == 1:
#            self.features = np.array([[nb] for nb in self.features])
#        if len(self.labels.shape) == 1:
#            self.labels = np.array([[nb] for nb in self.labels])

        if 'batchsize' in opts:
            self.batchsize = opts['batchsize']
        else:
            self.batchsize = None
        self.nbsamples = self.__countsamples()
        #self.nbsamples = self.db.count(emaf.tablecontext)
        self.nbexamples = int(np.ceil(self.nbsamples * opts['examplesratio']))
        self.nbtests = self.nbsamples - self.nbexamples

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
        #batchfeatures, batchlabels = (self.features[choice], self.labels[choice])
        return self.__choiceintosamples(choice)
#        batchfeatures,batchlabels = self.__choiceintosamples(choice)
        #tmpres = self.db.get(choice)
#        batchfeatures, batchlabels = (self.db.get , self.labels[choice])
#        return (batchfeatures, batchlabels)

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

#    @abstractmethod
    def __countsamples(self):
        pass

#    @abstractmethod
    def __choiceintosamples(self, choice):
        pass


class AudioFeeder(Feeder):
    def __init__(self, featurespath, labelspath=None, opts={}):
#        import extractfeatures as ef
        #opts.update({'featuremutation': ef.ExtractMonoAudioFiles.featuremutation, 'labelmutation': ef.ExtractMonoAudioFiles.labelmutation})
        #opts.update({'featuremutation': ef.ExtractMonoAudioFiles.featurefunc, 'labelmutation': ef.ExtractMonoAudioFiles.labelmutation})
        #opts.update({'labelmutation': ef.ExtractMonoAudioFiles.labelmutation})

        self.nbfeatures = None
        self.nblabels = emaf.nblabels

        super().__init__(opts)

    def __countsamples(self):
        return self.db.count(emaf.tablecontext)

    def __choiceintosamples(self, choice):
        nbsamples = self.batchsize
        nblabels = self.nblabels
        tmpres = self.db.get(choice)

        if self.nbfeatures is None:
            self.nbfeatures = len(tmpres[0]) - 3
        nbfeatures = self.nbfeatures

        features = np.zeros(shape=(nbsamples, nbfeatures))
        labels = np.zeros(shape=(nbsamples, nblabels))

        for i in range(nbsamples):
            line = list(tmpres[i])
            features[i] = line[3:]
            labels[i, line[1]] = 1.

        return (features, labels)
