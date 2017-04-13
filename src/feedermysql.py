import numpy as np
from extractfeaturesmysql import ExtractMonoAudioFiles as emaf
from mysqlstuffs import Database
#from abc import ABCMeta
from abc import *

#TODO: translate ids by rand permut

class Feeder(metaclass=ABCMeta):
    def __init__(self, opts={}):
        defopts = {'examplesratio': 0.95, 'dbname': emaf.outdb}
        defopts.update(opts)
        self.opts = opts = defopts

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
            #add use arange from 1 to nbexamples+1 because of the diff of indexiation between mysql and arrays in python (from 1 vs from 0)
            choice = np.random.choice(np.arange(1,self.nbexamples+1) , batchsize, False)
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
        #add 1 to nbexamples and nbsamples because of the diff of indexiation between mysql and arrays in python (from 1 vs from 0)
        choice = np.random.choice(np.arange(self.nbexamples+1, self.nbsamples+1), batchsize, False)
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
    def __init__(self, opts={}):
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
            features = features[:nbreturned]
            labels = labels[:nbreturned]
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
            #add 1 to sampleit because of the diff of indexiation between mysql and arrays in python (from 1 vs from 0)
            ids = np.array([idt[0] for idt in self.db.get(emaf.tablecontext, sampleit+1, ['id'], 'sample_id')])
            res.append(self.choiceintosamples(ids))
        return res

class AudioFeederFullContext(AudioFeederContext):
    def __init__(self, opts={}):
        defopts = {'nbaverage': 3}
        defopts.update(opts)
        opts = defopts

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

        restoparse = {feat[0]: (feat[2], feat[3:], feat[1]) for feat in tmpres}

        nberrors = 0
        for i in range(nbsamples):
            batch = np.zeros(shape=(self.nbaverage, nbfeatures))
            try:
                batch[0] = restoparse[smartchoice[i*self.nbaverage]][1]
                sample = restoparse[smartchoice[i*self.nbaverage]][0]
                for j in range(1, self.nbaverage):
                    tupleres = restoparse[smartchoice[i*self.nbaverage + j]]
                    if tupleres[0] == sample:
                        batch[j] = tupleres[1]
                    else:
                        batch = None
                        #print("%d is not in sample range. Passing" % (i,))
                        nberrors += 1
                        break
            except IndexError:
                print("%d has not all features available. Passing." % (i,))
                nberrors += 1
            #occurs when we are at the end of table
            except KeyError:
                batch = None
                nberrors += 1
            if batch is not None:
                features[i-nberrors] = np.mean(batch, 0)
                labels[i-nberrors, restoparse[smartchoice[i*self.nbaverage]][2]] = 1.

        if nberrors > 0:
            nbreturned = nbsamples - nberrors
            print("%d/%d returned" % (nbreturned, nbsamples))
            features = features[:nbreturned]
            labels = labels[:nbreturned]

        #print(labels)
        #print(features)
        feattocmp = super().choiceintosamples(choice)

#        assert np.array_equal(np.array(smartchoice), choice)
#        assert np.allclose(feattocmp[1], labels)
#        assert np.allclose(feattocmp[0], features)

        return (features, labels)

    def lissage(self, u, i):
        shape = np.asarray(np.shape(u))
        n = shape[0]
        shape[0] = n-i
        up = np.zeros(shape)
        for j in range(n-i):
            up[j] = np.mean(u[j:j+i+1], 0)
        return up

