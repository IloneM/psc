import numpy as np
import feeder as fe

feed = fe.Feeder(fe.emaf.outpath, opts={'batchsize': 100})

nbsamples = 2112

def compute(mysqlanswer):
    nbitems = len(mysqlanswer)

    features = np.zeros(shape=(nbitems, 128))
    labels = np.zeros(shape=(nbitems, 88))

    for i in range(nbitems):
        line = list(mysqlanswer[i])
        features[i] = np.array(line[3:])
        labels[i, line[1]] = 1.

    return(features, labels)

def getlistofind():
    mysqldb = feed.myslqtest.db
    tablecontext = fe.emafms.tablecontext
    reslist = [0] * nbsamples
    
    feed.dataloaded.wait()
        
    for i in range(nbsamples):
        ressql = compute(mysqldb.get(tablecontext, i+1, idfield='sample_id'))
        nbitems = len(ressql[0])
        
        minind = 0
        
        while minind < nbsamples and feed.nbitemsinsampleasready[minind] != nbitems:
            minind += 1

        if minind == nbsamples:
            print(nbitems)
            print(i)
            raise ValueError

        begin = feed.siii[minind]
        end = begin + feed.nbitemsinsampleasready[minind]
        
        diff = ressql[0] - feed.mergeditems[begin:end]
        minnorm = (diff*diff).sum()
        for j in range(minind+1, nbsamples):
            if feed.nbitemsinsampleasready[j] != nbitems:
                continue
            begin = feed.siii[j]
            end = begin + feed.nbitemsinsampleasready[j]
            diff = ressql[0] - feed.mergeditems[begin:end]

            if (diff*diff).sum() < minnorm:
                minind = j
                minnorm = (diff*diff).sum()

        reslist[i] = (minind, minnorm)

        print("%d/%d" % (i+1, nbsamples))

    return reslist

def assert_is_all_correct():
    invsamplesready = [0] * nbsamples

    feed.dataloaded.wait()

    for i in range(nbsamples):
        invsamplesready[feed.samplesready[i]] = i

    for i in range(nbsamples):
        msql = compute(feed.myslqtest.db.get(fe.emafms.tablecontext, i+1, idfield='sample_id'))
        begin = feed.siii[invsamplesready[i]]
        end = begin + feed.nbitemsinsampleasready[invsamplesready[i]]
        assert np.allclose(msql[0], feed.mergeditems[begin:end])
        assert np.allclose(msql[1], feed.mergedlabels[begin:end])

        print("%d/%d okay!" % (i+1, nbsamples))

assert_is_all_correct()

