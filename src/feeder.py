import numpy as np
from threading import Thread,Lock,Event
cpu_count = lambda: 4
#from multiprocessing import Process,Lock,Event,cpu_count
#Thread = Process
from os.path import join
from extractfeaturesnew import ExtractMonoAudioFiles as emaf

#We have several (~2000) files which are here refered as "samples".
#Each file has several lines of featured content which are refered as "items".
#And finally each of this lines has several columns of features which are simply refered as "features".

class Feeder:
    def loadlabel(self, it):
        nbitems = self.siii[it]
        nblabels = self.nblabels
        with open(join(self.inpath, 'label_%d' % (it+1,)), 'r') as fs:
            pitch = int(fs.read())
        res = np.zeros(shape=(nbitems, nblabels))
        res[:,pitch] = [1.] * nbitems
        return res

    def loadfeature(self, it):
        return np.loadtxt(join(self.inpath, 'feature_%d' % (it+1,)), ndmin=2)

    def loadsample(self, sampleit, threadsreslist, threadid):
        features = self.loadfeature(sampleit)
        labels = self.loadlabel(sampleit)
        threadsreslist[threadid] = (sampleit, features, labels)

    def updateloadeddata(self, newsampledata):
        newsampleid = newsampledata[0]
        newfeatures = newsampledata[1]
        newlabels = newsampledata[2]

        begin = self.nbitemsavailable
        end = self.nbitemsavailable + self.siii[newsampleid]
        self.mergeditems[begin:end] = newfeatures
        self.mergedlabels[begin:end] = newlabels

        self.samplesready.append(newsampleid)
        self.nbitemsavailable = end

        nbsamplesready = self.nbsamplesready = len(self.samplesready)
        if nbsamplesready <= self.nbexamplessamples:
            self.learningitemsavailable = end

            if not self.learnable.is_set() and nbsamplesready >= self.beginlearnceil:
                self.learnable.set()
        else:
            self.newtestavailable.set()

    def loaddata(self):
        nbthreads = cpu_count()
        ids = np.random.permutation(self.nbsamples)
        threads = [None] * nbthreads
        threadswork = [None] * nbthreads

        updatelock = Lock()

        for tit in range(min(nbthreads,self.nbsamples)):
            threads[tit] = Thread(target=self.loadsample, args=(ids[tit], threadswork, tit))
            threads[tit].start()

        tit = 0
        for sit in ids[nbthreads:]:
            while threadswork[tit] is None:
                tit = (tit + 1) % nbthreads

            with updatelock:
                self.updateloadeddata(threadswork[tit])

            threadswork[tit] = None
            threads[tit] = Thread(target=self.loadsample, args=(sit, threadswork, tit))
            threads[tit].start()

        for tit in range(nbthreads):
            if threads[tit] is None:
                break
            threads[tit].join()
            with updatelock:
                self.updateloadeddata(threadswork[tit])
            
    def __init__(self, inpath, opts={}):
        defopts = {'examplesratio': 0.95, 'beginlearningratio': 0.1}
        defopts.update(opts)
        self.opts = opts = defopts

        self.inpath = inpath

        self.metadatas = np.loadtxt(join(inpath, 'meta.dat'), dtype=int)
        #sample indexes in items i.e the first item of a sample considering agregaeded items
        self.siii = self.metadatas

        self.nbsamples = self.siii.size
        self.nbexamplessamples = int(np.ceil(self.nbsamples * opts['examplesratio']))
        self.beginlearnceil = int(np.ceil(self.nbexamplessamples * opts['beginlearningratio']))
#        self.nbtestsamples = self.nbsamples - self.nbexamplessamples
        self.nbitems = sum(self.siii)
        self.approxnbexamplesitems = int(np.ceil(self.nbitems * opts['examplesratio']))
        self.nbfeatures = int(inpath.split('_')[-1])
        self.nblabels = emaf.nblabels

        self.mergeditems = np.zeros(shape=(self.nbitems, self.nbfeatures))
        self.mergedlabels = np.zeros(shape=(self.nbitems, self.nblabels))

        self.samplesready = []
        self.nbsamplesready = 0
        self.nbitemsavailable = 0
        self.learningitemsavailable = 0
        self.learnable = Event()
        self.newtestavailable = Event()

        self.loadthread = Thread(target=self.loaddata)
        self.loadthread.start()

        #prevents to give scalar instead of a vector
        ##if len(self.mergeditems.ndim) == 1:
        ##    self.features = np.array([[nb] for nb in self.features])
        ##if len(self.mergedlabels.ndim) == 1:
        ##    self.labels = np.array([[nb] for nb in self.labels])

        if 'batchsize' in opts:
            self.batchsize = opts['batchsize']
        else:
            self.batchsize = None

        self.testitemsit = None
        self.testsampleit = None

    def getbatch(self, batchfeatures=None, batchlabels=None, batchsize=None):
        self.learnable.wait()

        if batchsize is None:
            batchsize = self.batchsize
        if batchsize is None:
            raise ValueError("You must provide a valid batchsize.")
        batchsize = min(batchsize, self.learningitemsavailable)

        choice = np.random.choice(self.learningitemsavailable, batchsize, False)
        print(self.learningitemsavailable)

        batchfeatures, batchlabels = (self.mergeditems[choice], self.mergedlabels[choice])
        return (batchfeatures, batchlabels)

    def __iter__(self):
        self.testitemsit = self.learningitemsavailable
        self.testsampleit = self.nbexamplessamples
        return self

    def __next__(self):
        if self.testsampleit >= self.nbsamples:
            raise StopIteration
        while self.testsampleit >= self.nbsamplesready:
            self.newtestavailable.clear()
            self.newtestavailable.wait(1.)
        begin = self.testitemsit
        end = self.testitemsit + self.siii[self.samplesready[self.testsampleit]]

        self.testsampleit += 1
        self.testitemsit = end

        return (self.mergeditems[begin:end], self.mergedlabels[begin:end])

    def __call__(self, batchfeatures=None, batchlabels=None, batchsize=None):
        return self.getbatch(batchfeatures, batchlabels, batchsize)

#class AudioFeeder(Feeder):
#    def __init__(self, featurespath, labelspath=None, opts={}):
#        import extractfeatures as ef
#        super().__init__(featurespath, labelspath, opts)
