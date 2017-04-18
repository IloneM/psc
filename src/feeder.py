import numpy as np
from threading import Thread,Lock,Event
cpu_count = lambda: 20
#from multiprocessing import Process,Lock,Event,cpu_count
#Thread = Process
from os.path import join
from extractfeatures import ExtractMonoAudioFiles as emaf
from extractfeaturesmysql import ExtractMonoAudioFiles as emafms
import feedermysql as fm

#We have several (~2000) files which are here refered as "samples".
#Each file has several lines of featured content which are refered as "items".
#And finally each of this lines has several columns of features which are simply refered as "features".

class Feeder:
    def loadlabel(self, it):
        nbitems = self.nbitemsinsample[it]
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
        nbitemsinnewsample = self.nbitemsinsample[newsampleid]

        assert nbitemsinnewsample == newfeatures.shape[0] == newlabels.shape[0]

        begin = self.nbitemsready
        end = self.nbitemsready + self.nbitemsinsample[newsampleid]

        self.mergeditems[begin:end] = newfeatures
        self.mergedlabels[begin:end] = newlabels

        self.samplesready.append(newsampleid)
        self.nbitemsinsampleasready.append(nbitemsinnewsample)
        self.siii.append(begin)
        self.nbitemsready = end

        nbsamplesready = self.nbsamplesready = len(self.samplesready)
        if nbsamplesready <= self.nbexamplessamples:
            self.nblearningsamplesready = nbsamplesready
            self.nblearningitemsready = end

            if not self.learnable.is_set() and nbsamplesready >= self.beginlearnceil:
                self.learnable.set()
        else:
            self.newtestready.set()

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

        self.dataloaded.set()
            
    def __init__(self, inpath, opts={}):
        defopts = {'examplesratio': 0.95, 'beginlearningratio': .75, 'deep': False, 'deepbatchsize': 64}
        defopts.update(opts)
        self.opts = opts = defopts

        self.inpath = inpath

        self.metadatas = np.loadtxt(join(inpath, 'meta.dat'), dtype=int, ndmin=2)
        self.nbitemsinsample = self.metadatas[:, 0]
        #self.siii = self.metadatas[:, 1]

        self.nbsamples = self.nbitemsinsample.size
        self.nbexamplessamples = int(np.ceil(self.nbsamples * opts['examplesratio']))
        self.beginlearnceil = int(np.ceil(self.nbexamplessamples * opts['beginlearningratio']))
#        self.nbtestsamples = self.nbsamples - self.nbexamplessamples
        self.nbitems = sum(self.nbitemsinsample)
        self.approxnbexamplesitems = int(np.ceil(self.nbitems * opts['examplesratio']))
        self.nbfeatures = int(inpath.split('_')[-1])
        self.nblabels = emaf.nblabels

        self.mergeditems = np.zeros(shape=(self.nbitems, self.nbfeatures))
        self.mergedlabels = np.zeros(shape=(self.nbitems, self.nblabels))

        #sample indexes in items i.e the first item of a sample considering agregaeded items
        self.siii = []
        self.samplesready = []
        self.nbitemsinsampleasready = []
        self.nbsamplesready = 0
        self.nbitemsready = 0
        self.nblearningsamplesready = 0
        self.nblearningitemsready = 0

        self.learnable = Event()
        self.dataloaded = Event()
        self.newtestready = Event()

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
        if 'deepbatchsize' in opts:
            self.deepbatchsize = opts['deepbatchsize']
        else:
            self.deepbatchsize = None

        self.testsampleit = None

        self.myslqtest = fm.AudioFeederContext(opts)

    def getdeepbatch(self, batchsize=None, deepbatchsize=None):
        self.learnable.wait()

        if batchsize is None:
            batchsize = self.batchsize
        if batchsize is None:
            raise ValueError("You must provide a valid batchsize.")
        if deepbatchsize is None:
            deepbatchsize = self.deepbatchsize
        if deepbatchsize is None:
            raise ValueError("You must provide a valid deepbatchsize.")

        batchsize = min(batchsize, self.nblearningitemsready)
        if self.nblearningitemsready == self.nbitemsready:
            print("Learning items ready: %d/~%d" % (self.nblearningitemsready, self.approxnbexamplesitems))

        batchfeatures = np.zeros(shape=(batchsize, self.nbfeatures, deepbatchsize))
        batchlabels = np.zeros(shape=(batchsize, self.nblabels, deepbatchsize))

        for i in range(batchsize):
            samplechoice = np.random.choice(self.nblearningsamplesready)
            samplesize = self.nbitemsinsampleasready[samplechoice]
            deepbatchsize = min(deepbatchsize, samplesize)
            relativeindexchoice = np.random.choice(samplesize-deepbatchsize)

            begin = self.siii[samplechoice] + relativeindexchoice
            end = begin + deepbatchsize

            batchfeatures[i] = self.mergeditems[begin:end].T
            batchlabels[i] = self.mergedlabels[begin:end].T
        return (batchfeatures, batchlabels)

    def getbatch(self, batchsize=None):
        self.learnable.wait()

        if batchsize is None:
            batchsize = self.batchsize
        if batchsize is None:
            raise ValueError("You must provide a valid batchsize.")
        batchsize = min(batchsize, self.nblearningitemsready)

        choice = np.random.choice(self.nblearningitemsready, batchsize, False)
        if self.nblearningitemsready == self.nbitemsready:
            print("Learning items ready: %d/~%d" % (self.nblearningitemsready, self.approxnbexamplesitems))

        batchfeatures, batchlabels = (self.mergeditems[choice], self.mergedlabels[choice])
        return (batchfeatures, batchlabels)

    def challengethis(self):
        choice = np.random.choice(self.nbsamplesready)
        frommysql = self.myslqtest.db.get(emafms.tablecontext, self.samplesready[choice], idfield='sample_id')

        nbsamples = len(frommysql)
        features = np.zeros(shape=(nbsamples, self.nbfeatures))
        labels = np.zeros(shape=(nbsamples, self.nblabels))

        try:
            for i in range(nbsamples):
                line = list(frommysql[i])
                features[i] = line[3:]
                labels[i, line[1]] = 1.
        except IndexError:
            if frommysql is not None:
                nbreturned = len(frommysql)
            else:
                nbreturned = 0
            print("%d/%d returned" % (nbreturned, choice.size))

        try:
            begin = self.siii[choice]
            end = begin + self.nbitemsinsample[choice]

            assert nbsamples == end - begin
            assert np.allclose(features, self.mergeditems[begin:end])
            assert np.allclose(labels, self.mergedlabels[begin:end])
        except AssertionError:
            print("nbs mysql: %d\tnbs new:%d" % (nbsamples, end-begin))
            print(features)
            print(self.mergeditems[begin:end])
            print()
            print(labels)
            print(self.mergedlabels[begin:end])
            print('T' if np.allclose(labels, self.mergedlabels[begin:end]) else 'F')
            print()

    def getfulltests(self):
        self.dataloaded.wait()
        begin = self.nblearningitemsready
        end = self.nbitems

        return (self.mergeditems[begin:end], self.mergedlabels[begin:end])

    def __iter__(self):
        self.testsampleit = self.nbexamplessamples
        return self

    def __next__(self):
        if self.testsampleit >= self.nbsamples:
            raise StopIteration
        while self.testsampleit >= self.nbsamplesready:
            self.newtestready.clear()
            self.newtestready.wait(1.)
        begin = self.siii[self.testsampleit]
        end = begin + self.nbitemsinsampleasready[self.testsampleit]

        self.testsampleit += 1

        return (self.mergeditems[begin:end], self.mergedlabels[begin:end])

    def __call__(self, batchsize=None):
        if self.opts['deep']:
            return self.getdeepbatch(batchsize)
        return self.getbatch(batchsize)

#class AudioFeeder(Feeder):
#    def __init__(self, featurespath, labelspath=None, opts={}):
#        import extractfeatures as ef
#        super().__init__(featurespath, labelspath, opts)
