import sys
import scipy.spatial #used for hierarchical clustering
import scipy.cluster #used for hierarchical clustering
import sklearn.metrics #used ONLY for calculating pairwise distance function
import matplotlib.pyplot as plt
import itertools
import numpy as np
np.set_printoptions(threshold='nan')

#Global settings
isVisualize = 0
UsualSetting = 1
AnalysisB = 0
AnalysisC = 0
Bonus = 0

NUMITERS = 50
Ks = [2,4,8,16,32]
# Ks = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
##############################Utility functions ########################
def readCSVFile(fileName):
    csv_delimiter = ','
    csvContent = np.loadtxt(open(fileName,'rb'),dtype=float,delimiter=csv_delimiter,skiprows=0)
    return csvContent

def sortOn2ndCol(content):
    return content[content[:, 1].argsort()]

def visualizeImages(content,classLabels,k):
    for i in range(0,k):
        indexImage = np.where(classLabels==i)[0][0]
        print indexImage
        image = content[indexImage][2:]
        image = image.reshape((28,28))
        print image.shape
        plt.imshow(image,cmap='gray')
        # plt.show()
        plt.savefig('image' + str(i) + '.png',cmap='gray')

def visualizeClusters(content,clusters):
    numClusters = len(clusters)
    # print numClusters
    for i in range(0,numClusters):
        plt.scatter(content[clusters[i],2],content[clusters[i],3],s=10,label=str(i))
    plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.)
    plt.title('Random clusters of pca embedding')
    plt.savefig('bonus/randomCluster.png')
    plt.close()

def visualizeClustersWithMean(content,clusters,means,iter):
    numClusters = len(clusters)
    for i in range(0,numClusters):
        plt.scatter(content[clusters[i],2],content[clusters[i],3],s=10,label=str(i))
    plt.scatter(means[:,0],means[:,1],s=30,marker='*',edgecolor='k',label='means')
    plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.)
    # plt.show()
    plt.title('Clusters using kmeans at iteration :' + str(iter))
    plt.savefig('results/clusteringIter' + str(iter) + '.png')
    plt.close()

def getRandomClusters(content,k):
    randExIds = np.random.choice(np.arange(0,len(content)),1000,replace=False)
    clusters = []
    for i in range(0,k):
        ids = np.where(content[randExIds,1]==i)
        clusters.append(randExIds[ids])
    return clusters

def visualize(content,k):
    if len(content[0]) > 5:
        classLabels = content[:,1]
        visualizeImages(content,classLabels,k)
    else:
        clusters = getRandomClusters(content,k)
        visualizeClusters(content,clusters)

def getSimiMatrix(content):
    simiMat = np.zeros((len(content),len(content)),dtype=float)
    for i in range(0,len(content)):
        print str(i) + ' / ' + str(len(content))
        for j in range(i,len(content)):
            dist = getEucDist(content[i,:],content[j,:])
            simiMat[i,j] = simiMat[j,i] = dist
    return simiMat

def initClusters(k,sizeContent):
    allExIds = np.arange(0,sizeContent)
    return np.random.choice(allExIds,k,replace=False)

def getEucDist(x,y):
    return np.linalg.norm(x-y)

def getSSD(x,y):
    return np.sum(np.square(x-y))

def plotDendogram(Z,x):
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical clustering dendrogram ' + x)
    plt.xlabel('index')
    plt.ylabel('distance')
    scipy.cluster.hierarchy.dendrogram(
        Z,
        leaf_rotation=90.,
        leaf_font_size=8.,
    )
    plt.savefig('analysisC/'+ x +'Link.png')
    plt.close()

def average(listNums):
    return float(sum(listNums)) / max(len(listNums), 1)

###########################k means functions#########################
def getNearestMean(x,means):
    leastDist = 1e20
    nearestMeanId = 0

    for i in range(0,len(means)):
        dist = getEucDist(x,means[i,:])
        if dist < leastDist:
            nearestMeanId = i
            leastDist = dist
    return nearestMeanId

def doKMeans(k, content):
    meansIds = initClusters(k,len(content))
    means = content[meansIds,:]
    for iter in range(0,NUMITERS):
        # print iter
        clusters = [[] for i in range(k)]
        for i in range(0,len(content)):
            nearestMeanId = getNearestMean(content[i,2:],means[:,2:])
            clusters[nearestMeanId].append(i)

        for c in range(0,len(clusters)):
            if len(clusters[c]) == 0:
                indexes = range(0,len(content))
                means[c,2:] = content[np.random.choice(indexes,1),2:]
            else:
                means[c,2:] = np.mean(content[clusters[c],2:],axis=0,dtype=float)
    return clusters, means

############################Evaluation functions###################
def findSCFast(X, labels):
    As = findAs(X, labels)
    Bs = findBs(X, labels)
    indivSC = (Bs - As) / np.maximum(As, Bs)
    indivSC = np.nan_to_num(indivSC)
    return np.mean(indivSC)

def findAvgDistsAs(X):
    dists = sklearn.metrics.pairwise.pairwise_distances(X, metric='euclidean')
    avgDists = dists.sum(axis=1) / (dists.shape[0]-1)
    return avgDists

def findAs(X, labels):
    As = np.zeros(labels.size, dtype=float)
    avgdists = []
    for label in np.unique(labels):
        tempX = X[np.where(labels == label)[0]]
        avgdists.append(findAvgDistsAs(tempX))
    uniqueLabels = np.unique(labels)
    for label, dists in zip(uniqueLabels, avgdists):
        tempIds = np.where(labels == label)[0]
        As[tempIds] = dists
    return As


def findAvgDistsBs(X, Y):
    dist = sklearn.metrics.pairwise.pairwise_distances(X, Y, metric='euclidean')
    dist_a = dist.mean(axis=1)
    dist_b = dist.mean(axis=0)
    return dist_a, dist_b

def findBs(X, labels):
    Bs = np.zeros(labels.size, dtype=float)
    Bs[:] = 1e100
    uniqueLabels = np.unique(labels)
    avgdists = []
    for lX,lY in itertools.combinations(uniqueLabels, 2):
        tempX = X[np.where(labels == lX)[0]]
        tempY = X[np.where(labels == lY)[0]]
        avgdists.append(findAvgDistsBs(tempX,tempY))
    for (lX, lY), (avgdistsX, avgdistsY) in zip(itertools.combinations(uniqueLabels, 2), avgdists):
        tempIdsX = np.where(labels == lX)[0]
        Bs[tempIdsX] = np.minimum(avgdistsX, Bs[tempIdsX])
        tempIdsY = np.where(labels == lY)[0]
        Bs[tempIdsY] = np.minimum(avgdistsY, Bs[tempIdsY])
    return Bs


def findWCSSD(content, clusters, means):
    WCSSD = 0.0
    for i in range(0, len(means)):
        for j in range(0, len(clusters[i])):
            means = np.asarray(means)
            WCSSD += getSSD(content[clusters[i][j],2:], means[i,2:])
    return WCSSD

def getSecondNearestMean(x,means):

    dists = np.zeros((len(means),1),dtype=float)

    for i in range(0,len(means)):
        dists[i] = getEucDist(x,means[i,:])
    sortedDistsIds = np.argsort(dists)

    return sortedDistsIds[1]

def findSCslow(content, clusters, means):
    indivSC = np.zeros((len(content),1),dtype=float)
    counter = 0
    for i in range(0,len(clusters)):
        print i
        for j in range(0,len(clusters[i])):
            x = content[clusters[i][j],2:]
            nearestMeanId = getSecondNearestMean(x,means[:,2:])
            A = 0.0
            B = 0.0
            for c in range(0,len(clusters[nearestMeanId])):
                B += np.linalg.norm(content[clusters[nearestMeanId][c],2:]-x)
            B = B / float(len(clusters[nearestMeanId]))
            for c in range(0,len(clusters[i])):
                A += np.linalg.norm(content[clusters[i][c],2:]-x)
            A = A / float(len(clusters[i]))
            indivSC[counter] = (B-A) / max(A,B)
            counter += 1
    SC = np.sum(indivSC) / float(len(indivSC))
    return SC

def findSCGivenSimi(content, clusters, means, simi):
    indivSC = np.zeros((len(content),1),dtype=float)
    counter = 0
    for i in range(0,len(clusters)):
        # print i
        for j in range(0,len(clusters[i])):
            x = content[clusters[i][j],2:]
            nearestMeanId = getSecondNearestMean(x,means[:,2:])
            A = 0.0
            B = 0.0
            for c in range(0,len(clusters[nearestMeanId])):
                B += simi[clusters[nearestMeanId][c],clusters[i][j]]
            B = B / float(len(clusters[nearestMeanId]))
            for c in range(0,len(clusters[i])):
                A += simi[clusters[i][c],clusters[i][j]]
            A = A / float(len(clusters[i]))
            indivSC[counter] = (B-A) / max(A,B)
            counter += 1
    SC = np.sum(indivSC) / float(len(indivSC))
    return SC

def getClasses(content,labels):
    classes = []
    for i in range(0,len(labels)):
        ids = np.where(content[:,1] == labels[i])
        classes.append(ids[0])
    return classes

def getPCorGs(C,lenContent):
    pcs = np.zeros((len(C),1),dtype=float)
    for i in range(0,len(C)):
        pcs[i] = float(len(C[i])) / float(lenContent)
    return pcs

def getPCGs(classes,clusters,lenContent):
    pcgs = np.zeros((len(classes),len(clusters)),dtype=float)
    for i in range(0,len(classes)):
        for j in range(0,len(clusters)):
            pcgs[i,j] = float(len(np.intersect1d(classes[i],clusters[j]))) / float(lenContent)
    return pcgs

def findNMI(content, clusters):
    classes = getClasses(content,np.unique(content[:,1]))
    pcs = getPCorGs(classes, len(content))
    pgs = getPCorGs(clusters, len(content))
    pcgs = getPCGs(classes,clusters,len(content))
    NMI = 0.0
    X = 0.0
    Y = 0.0

    for i in range(0,len(pcs)):
        X -= pcs[i] * np.log(pcs[i])
    for i in range(0,len(pgs)):
        Y -= pgs[i] * np.log(pgs[i])

    for i in range(0,len(pcs)):
        for j in range(0,len(pgs)):
            if (pcgs[i,j]!=0.0):
                NMI += pcgs[i,j] * np.log(float(pcgs[i,j] / (pcs[i]*pgs[j])))

    NMI = NMI / (X+Y)
    return NMI[0]

def assignLabels(clusters,lenC):
    labels = np.zeros((1,lenC))[0]
    for i in range(0,len(clusters)):
        for j in range(0,len(clusters[i])):
            labels[clusters[i][j]] = i
    return labels

############################Analysis B###########################
def formdata2467(content):
    indexes = []
    indexes.append(np.where(content[:,1] == 2)[0])
    indexes.append(np.where(content[:,1] == 4)[0])
    indexes.append(np.where(content[:,1] == 6)[0])
    indexes.append(np.where(content[:,1] == 7)[0])
    indexes = np.concatenate(indexes).ravel()
    return content[indexes,:]

def formdata67(content):
    indexes = []
    indexes.append(np.where(content[:,1] == 6)[0])
    indexes.append(np.where(content[:,1] == 7)[0])
    indexes = np.concatenate(indexes).ravel()
    return content[indexes,:]

def performAnalysisB1(content,whichData):
    wcs = []
    scs = []

    for i in range(0, len(Ks)):
        k = Ks[i]
        print 'k = ' + str(k)

        clusters, means = doKMeans(k, content)

        WCSSD = findWCSSD(content, clusters, means)
        print 'WC-SSD ' + str(WCSSD)
        wcs.append(WCSSD)
        labels = assignLabels(clusters,len(content))
        SC = findSCFast(content[:,2:], labels)
        print 'SC ' + str(SC)
        scs.append(SC)

    print wcs
    print scs

    plt.figure()
    plt.plot(Ks, wcs)
    plt.ylabel('WC-SSD')
    plt.xlabel('k values')
    plt.title('WC-SSD for different k values ')
    plt.savefig('analysisB/B1WCSSD'+whichData+'.png')
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(Ks, scs)
    plt.ylabel('SC')
    plt.xlabel('k values')
    plt.title('SC for different k values ')
    plt.savefig('analysisB/B1SC'+whichData+'.png')
    plt.show()
    plt.close()

def performAnalysisB3(content,whichData):
    avgwcs = []
    avgscs = []
    varwcs = []
    varscs = []

    for i in range(0, len(Ks)):
        k = Ks[i]
        print 'k = ' + str(k)
        wcs = []
        scs = []
        for seeds in range(0,10):

            clusters, means = doKMeans(k, content)

            WCSSD = findWCSSD(content, clusters, means)
            print 'WC-SSD ' + str(WCSSD)
            wcs.append(WCSSD)
            labels = assignLabels(clusters,len(content))
            SC = findSCFast(content[:,2:],labels)
            print 'SC ' + str(SC)
            scs.append(SC)

        print wcs
        print scs

        avgwc = average(wcs)
        varwc = np.var(wcs)
        avgsc = average(scs)
        varsc = np.var(scs)

        avgwcs.append(avgwc)
        varwcs.append(varwc)
        avgscs.append(avgsc)
        varscs.append(varsc)

    print 'average and variance WC-SSDs'
    print avgwcs
    print varwcs
    print 'average and variance SCs'
    print avgscs
    print varscs

    plt.figure()
    plt.errorbar(Ks, avgwcs, varwcs)
    plt.ylabel('WC-SSD')
    plt.xlabel('k value')
    plt.title('WC-SSD for different k values ')
    plt.savefig('analysisB/B3WCSSD'+whichData+'.png')
    plt.close()

    plt.figure()
    plt.errorbar(Ks, avgscs, varscs)
    plt.ylabel('SC')
    plt.xlabel('k value')
    plt.title('SC for different k values ')
    plt.savefig('analysisB/B3SC'+whichData+'.png')
    plt.close()

def visualizeClustersWithLabels(content,clusters,whichData):
    for i in range(0,len(clusters)):
        plt.scatter(content[clusters[i],2],content[clusters[i],3],s=10,label=str(i))

    plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.)
    plt.title('Visualize clustering random 1000 points k means')
    plt.savefig('analysisB/B4Cluster'+whichData+'.png')
    plt.close()

def performAnalysisB4(content,k,whichData):
    print 'data ' + whichData
    clusters, means = doKMeans(k, content)
    NMI = findNMI(content, clusters)
    print 'NMI ' + str(NMI)
    visualizeClustersWithLabels(content,clusters,whichData)

def performAnalysisB(fulldata):
    data2467 = formdata2467(fulldata)
    data67 = formdata67(fulldata)

    print 'full data'
    performAnalysisB1(fulldata,'full')
    print 'data with 2 4 6 7'
    performAnalysisB1(data2467,'2467')
    print 'data with 6 7'
    performAnalysisB1(data67,'67')

    print 'full data'
    performAnalysisB3(fulldata,'full')
    print 'data with 2 4 6 7'
    performAnalysisB3(data2467,'2467')
    print 'data with 6 7'
    performAnalysisB3(data67,'67')

    performAnalysisB4(fulldata,8,'full')
    performAnalysisB4(data2467,4,'2467')
    performAnalysisB4(data67,2,'67')

############################Analysis C###########################
def performAnalysisC12(content):
    data = np.zeros((100,4))
    for i in range(0,len(np.unique(content[:,1]))):
        indexes = np.where(content[:,1] == i)[0]
        indexes10 = np.random.choice(indexes,10,replace=False)
        data[i*10:i*10+10,:] = content[indexes10]

    distMat = scipy.spatial.distance.pdist(data[:,2:])
    Zsingle = scipy.cluster.hierarchy.single(distMat)
    Zcomplete = scipy.cluster.hierarchy.complete(distMat)
    Zaverage = scipy.cluster.hierarchy.average(distMat)
    plotDendogram(Zsingle,'single')
    plotDendogram(Zcomplete,'complete')
    plotDendogram(Zaverage,'average')
    return Zsingle, Zcomplete, Zaverage, data

def labels2clusters(labels):
    uniqLabels = np.unique(labels)
    clusters = []
    for i in range(0,len(uniqLabels)):
        label = uniqLabels[i]
        indexes = np.where(labels == label)[0]
        clusters.append(indexes)
    return clusters

def findMeansOfClusters(content,clusters):
    means = []
    for i in range(0,len(clusters)):
        m = np.mean(content[clusters[i],2:],axis=0)
        means.append(np.array([0,0,m[0],m[1]]))
    return means

def performAnalysisC3(Zs, Zc, Za, content):
    wcsSingle = []
    scsSingle = []
    wcsComplete = []
    scsComplete = []
    wcsAverage= []
    scsAverage = []
    for i in range(0,len(Ks)):
        k = Ks[i]
        print 'k = ' + str(k)

        labelsSingle = scipy.cluster.hierarchy.fcluster(Zs, k, criterion='maxclust')
        clustersSingle = labels2clusters(labelsSingle)
        labelsComplete = scipy.cluster.hierarchy.fcluster(Zc, k, criterion='maxclust')
        clustersComplete = labels2clusters(labelsComplete)
        labelsAverage = scipy.cluster.hierarchy.fcluster(Za, k, criterion='maxclust')
        clustersAverage = labels2clusters(labelsAverage)

        print 'Single linkage'
        WCSSD = findWCSSD(content, clustersSingle, findMeansOfClusters(content,clustersSingle))
        print 'WC-SSD ' + str(WCSSD)
        wcsSingle.append(WCSSD)
        labels = assignLabels(clustersSingle,len(content))
        SC = findSCFast(content[:,2:],labels)
        print 'SC ' + str(SC)
        scsSingle.append(SC)

        print 'Complete linkage'
        WCSSD = findWCSSD(content, clustersComplete, findMeansOfClusters(content,clustersComplete))
        print 'WC-SSD ' + str(WCSSD)
        wcsComplete.append(WCSSD)
        labels = assignLabels(clustersComplete,len(content))
        SC = findSCFast(content[:,2:],labels)
        print 'SC ' + str(SC)
        scsComplete.append(SC)

        print 'Average linkage'
        WCSSD = findWCSSD(content, clustersAverage, findMeansOfClusters(content,clustersAverage))
        print 'WC-SSD ' + str(WCSSD)
        wcsAverage.append(WCSSD)
        labels = assignLabels(clustersAverage,len(content))
        SC = findSCFast(content[:,2:],labels)
        print 'SC ' + str(SC)
        scsAverage.append(SC)

    plt.figure()
    plt.plot(Ks, wcsSingle, label='Single')
    plt.plot(Ks, wcsComplete, label='Complete')
    plt.plot(Ks, wcsAverage, label='Average')
    plt.ylabel('WC-SSD')
    plt.xlabel('k values')
    plt.title('WC-SSD for different k values ')
    plt.legend()
    plt.savefig('analysisC/C3WCSSD.png')
    plt.close()

    plt.figure()
    plt.plot(Ks, scsSingle, label='Single')
    plt.plot(Ks, scsComplete, label='Complete')
    plt.plot(Ks, scsAverage, label='Average')
    plt.ylabel('SC')
    plt.xlabel('k values')
    plt.title('SC for different k values ')
    plt.legend()
    plt.savefig('analysisC/C3SC.png')
    plt.close()

def performAnalysisC5(Zs,Zc,Za,content,k):
    labelsSingle = scipy.cluster.hierarchy.fcluster(Zs, k, criterion='maxclust')
    clustersSingle = labels2clusters(labelsSingle)
    labelsComplete = scipy.cluster.hierarchy.fcluster(Zc, k, criterion='maxclust')
    clustersComplete = labels2clusters(labelsComplete)
    labelsAverage = scipy.cluster.hierarchy.fcluster(Za, k, criterion='maxclust')
    clustersAverage = labels2clusters(labelsAverage)

    NMISingle = findNMI(content,clustersSingle)
    NMIComplete = findNMI(content,clustersComplete)
    NMIAverage = findNMI(content,clustersAverage)

    print 'NMI SINGLE : ' + str(NMISingle)
    print 'NMI COMPLETE : ' + str(NMIComplete)
    print 'NMI AVERAGE : ' + str(NMIAverage)


def performAnalysisC(content):
    Zs, Zc, Za, data = performAnalysisC12(content)
    performAnalysisC3(Zs, Zc, Za, data)
    performAnalysisC5(Zs, Zc, Za, data,10)

###########################Bonus Analysis#################
def performPCA(D):
    meanD = np.mean(D,axis=0)
    X = D - meanD
    S = np.cov(X)
    np.save('covarianceMatrix',S)
    A,v = np.linalg.eig(S)
    return A,v,X

def visualizeEigVecs(eigVecsTop):
    for i in range(0,len(eigVecsTop[0])):
        e = eigVecsTop[:,i]
        e = e.reshape((28,28))
        plt.imshow(e,cmap='gray')
        plt.show
        plt.savefig('bonus/eigVec' + str(i) + '.png')
        plt.close()

def reduceDims(content,eigVecs,dims):

    contentRed = np.zeros((len(content),dims),dtype=float)
    contentRed = np.dot(content,eigVecs[:,-dims:])
    return contentRed

def bonusAnalysis(content,dims,k,whichData):

    eigVals, eigVecs, X = performPCA(content[:,2:].T)
    indexesSort = np.argsort(eigVals)
    eigVecsTop = eigVecs[:,indexesSort]
    eigVecsTop = np.real(eigVecsTop[:,-dims:])

    visualizeEigVecs(eigVecsTop[:,-dims:])

    reducedContent = reduceDims(X.T,eigVecsTop,2)
    reducedContent = np.concatenate((content[:,[0,1]],reducedContent),axis=1)
    visualize(reducedContent,10)

    reducedContent = reduceDims(X.T,eigVecsTop,10)
    reducedContent = np.concatenate((content[:,[0,1]],reducedContent),axis=1)
    performAnalysisB1(reducedContent,'BONUS'+whichData)
    performAnalysisB4(reducedContent,k,'BONUS'+whichData)

def performBonus(fulldata,dims):
    data2467 = formdata2467(fulldata)
    data67 = formdata67(fulldata)

    print 'full data'
    bonusAnalysis(fulldata,dims,8,'fulldata')
    print 'data 2 4 6 7'
    bonusAnalysis(data2467,dims,4,'data2467')
    print 'data 6 7'
    bonusAnalysis(data67,dims,2,'data67')

#########################Main function###################
def main(arg):

    if isVisualize:
        digitsContent = readCSVFile(arg[1])
        k = int(arg[2])
        visualize(digitsContent,k)

    if UsualSetting:
        digitsContent = readCSVFile(arg[1])
        k = int(arg[2])

        clusters, means = doKMeans(k, digitsContent)

        WCSSD = findWCSSD(digitsContent, clusters, means)
        print 'WC-SSD ' + str(WCSSD)
        labels = assignLabels(clusters,len(digitsContent))
        SC = findSCFast(digitsContent[:,2:],labels)
        print 'SC ' + str(SC)
        NMI = findNMI(digitsContent, clusters)
        print 'NMI ' + str(NMI)

    if AnalysisB:
        digitsContent = readCSVFile(arg[1])
        performAnalysisB(digitsContent)

    if AnalysisC:
        digitsContent = readCSVFile(arg[1])
        performAnalysisC(digitsContent)

    if Bonus:
        digitsRaw = readCSVFile(arg[1])
        dims = int(arg[2])
        performBonus(digitsRaw, dims)

if __name__ == "__main__":
    main(sys.argv[0:])
