import numpy as np
from extractfeaturesmysql import ExtractMonoAudioFiles as emaf
from mysqlstuffs import Database
#from abc import ABCMeta
from abc import *

#TODO: translate ids by rand permut

class Feeder(metaclass=ABCMeta):
    def __init__(self, opts={}):
        opts.update({'examplesratio': 0.95, 'dbname': emaf.outdb})
        self.opts = opts

        self.db = Database(self.opts['dbname'])

        if 'batchsize' in opts:
            self.batchsize = opts['batchsize']
        else:
            self.batchsize = None
        self.nbsamples = self.countsamples()
        self.nbexamples = self.countexamples()
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
            return self.choiceintosamples(choice)
        else:
            return self.controlprocess(batchsize)

    def switchmode(self):
        self.examplemode = not self.examplemode

    def __call__(self, batchfeatures=None, batchlabels=None, batchsize=None):
        return self.getbatch(batchfeatures, batchlabels, batchsize)

    def __next__(self):
        return self.getbatch(batchsize=1)

    def __iter__(self):
        return self

    def controlprocess(self, batchsize):
        if batchsize > self.nbtests:
            raise IndexError
        choice = np.random.choice(np.arange(self.nbexamples, self.nbsamples), batchsize, False)
        return self.choiceintosamples(choice)

    @abstractmethod
    def countsamples(self):
        pass

    @abstractmethod
    def choiceintosamples(self, choice):
        pass

    def countexamples(self):
        return int(np.ceil(self.nbsamples * self.opts['examplesratio']))

class AudioFeeder(Feeder):
    def __init__(self, featurespath, labelspath=None, opts={}):
        self.nbfeatures = None
        self.nblabels = emaf.nblabels

        super().__init__(opts)

        #nbfeatures definition
        self.__next__()

    def countsamples(self):
        return self.db.count(emaf.tablecontext)

    def choiceintosamples(self, choice):
        nbsamples = choice.size
        nblabels = self.nblabels
        tmpres = self.db.get(emaf.tablecontext, choice)

        if self.nbfeatures is None:
            self.nbfeatures = len(tmpres[0]) - 3
        nbfeatures = self.nbfeatures

        features = np.zeros(shape=(nbsamples, nbfeatures))
        labels = np.zeros(shape=(nbsamples, nblabels))

        try:
            for i in range(nbsamples):
                line = list(tmpres[i])
                features[i] = line[3:]
                labels[i, line[1]] = 1.
        except IndexError:
            if tmpres is not None:
                nbreturned = len(tmpres)
            else:
                nbreturned = 0
            print("%d/%d returned" % (nbreturned, choice.size))
        return (features, labels)

class AudioFeederContext(AudioFeeder):
    def countexamples(self):
        approxres = super().countexamples()
        itempersample = self.itempersample = self.db.count(emaf.tablecontext, 'sample_id')
        totitems = 0
        itsample = 0
        while totitems < approxres:
            totitems += itempersample[itsample][0]
            itsample += 1

        self.examplefirstitem = itsample
        return totitems

    def controlprocess(self, batchsize):
        res = []
        nbgroupedsamples = len(self.itempersample)
        for sampleit in range(self.examplefirstitem, nbgroupedsamples):
            ids = np.array([idt[0] for idt in self.db.get(emaf.tablecontext, sampleit, ['id'], 'sample_id')])
            res.append(self.choiceintosamples(ids))
        return res

class AudioFeederFullContext(AudioFeederContext):
    def __init__(self, featurespath, labelspath=None, opts={}):
        opts.update({'nbaverage': 3})

        self.nbaverage = opts['nbaverage']
        
        super().__init__(opts)

    def choiceintosamples(self, choice):
        nbsamples = choice.size
        nblabels = self.nblabels

        smartchoice = [None] * (nbsamples * self.nbaverage)

        for i in range(nbsamples):
            for j in range(self.nbaverage):
                smartchoice[i*self.nbaverage + j] = choice[i] +j

        tmpres = self.db.get(emaf.tablecontext, smartchoice)

        if self.nbfeatures is None:
            self.nbfeatures = len(tmpres[0]) - 3
        nbfeatures = self.nbfeatures

        features = np.zeros(shape=(nbsamples, nbfeatures))
        labels = np.zeros(shape=(nbsamples, nblabels))

        restoparse = {feat[0]: (feat[2], feat[3:]) for feat in features}

        try:
            res = []
            for i in range(nbsamples):
                batch = [None] * self.nbaverage
                batch[0] = restoparse[smartchoice[i*self.nbaverage]][1]
                sample = restoparse[smartchoice[i*self.nbaverage]][0]
                for j in range(1,self.nbaverage):
                    batch[j] = restoparse[smartchoice[i*self.nbaverage + j]]
                    if batch[j][0] == sample:
                        batch[j] = batch[j][1]
                    else:
                        batch = None
                        break
                if batch is not None:
                    res.append(fonctionmontal(batch))
        except IndexError:
            if restoparse is not None:
                nbreturned = len(restoparse)
            else:
                nbreturned = 0
            print("%d/%d returned" % (nbreturned, choice.size * self.nbaverage))
        return (features, labels)

