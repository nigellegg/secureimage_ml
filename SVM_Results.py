# /*******************************************************
#   Copyright (C) 2015-2017 Yifan Xie <yxyxyxyxyx@gmail.com>
#
#   This file is part of the proejct "imagesecurity", 
#   and is written to be exploit within the scope of the aforementioned proejct
#
#   This code can not be copied and/or distributed without the express
#   permission of Yifan Xie
#  *******************************************************/


__author__ = 'xie'
## This file load the picked data, and perform random split,
# to form training, test, and validation set


import cPickle
import numpy as np
from random import randrange
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn import svm
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.learning_curve import learning_curve
from sklearn.neural_network import BernoulliRBM





########################################################################################################################
def get_train_test(data):
    features=data[0]
    labels=data[1]

    #flatten feature to 2d matrix

    flat_features=features.reshape(384, 60*40*3)

    seed =randrange(100)
    train_x, test_x, train_y, test_y=train_test_split(flat_features, labels,
                                                         test_size=0.2,
                                                         random_state=seed)
    return train_x, test_x, train_y, test_y


########################################################################################################################
def simpleSVM(trainfeatures, testfeatures, trainlabels, testlabels):
    ## ******************* Feature Scaling *******************
    #print "performing feature scaling"
    min_max_scaler=preprocessing.MinMaxScaler()
    trainfeatures_fs=min_max_scaler.fit_transform(trainfeatures)
    testfeatures_fs=min_max_scaler.transform(testfeatures)

    # Training
    #print "training SVM model"


    clf = svm.SVC(C=5.0, kernel='sigmoid', degree=3, gamma=0.5, coef0=10.0, shrinking=True,
    probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False,
    max_iter=-1, random_state=None)

    # clf = svm.SVC(C=5.0, kernel='rbf', degree=3, gamma=2, coef0=10, shrinking=True,
    # probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False,
    # max_iter=-1, random_state=None)

    clf.fit(trainfeatures_fs, trainlabels)
    results=clf.predict(testfeatures_fs)


    results=results.ravel()
    testerror=float(len(testlabels)-np.sum(testlabels==results))/float(len(testlabels))
    # print"error rate with SVM 2 is %.4f" %testerror
    return testerror


def RBM_SVM(trainfeatures, testfeatures, trainlabels, testlabels):
    # ******************* Scikit-learning RBM + SVM *******************
    print "train RBM+SVM model"

    ##    trainfeatures = (trainfeatures - np.min(trainfeatures, 0)) / (np.max(trainfeatures, 0) + 0.0001)  # 0-1 scaling
    min_max_scaler=preprocessing.MinMaxScaler()
    trainfeatures_fs=min_max_scaler.fit_transform(trainfeatures)
    testfeatures_fs=min_max_scaler.transform(testfeatures)

    # SVM parameters
    clf = svm.SVC(C=5.0, kernel='sigmoid', degree=3, gamma=0.5, coef0=10.0, shrinking=True,
    probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False,
    max_iter=-1, random_state=None)

    # RBM parameters
    rbm = BernoulliRBM(random_state=0, verbose=True)
    rbm.learning_rate = 0.06
    rbm.n_iter = 20

    # Machine learning pipeline
    classifier = Pipeline(steps=[('rbm', rbm), ('svm', clf)])


    # More components tend to give better prediction performance, but larger
    # fitting time
    rbm.n_components = 400
    classifier.fit(trainfeatures_fs, trainlabels)
    results=classifier.predict(testfeatures_fs)

    results=results.ravel()
    testerror=float(len(testlabels)-np.sum(testlabels==results))/float(len(testlabels))
    # print"error rate with SVM  is %.4f" %testerror

    return testerror



########################################################################################################################
if __name__ == '__main__':
    # folder = os.path.dirname(__file__)
    folder="c:/users/xie/playground/cctv classification"

    pickle_file=folder+"/pickle_data/data.pkl"


    load_file=open(pickle_file,'rb')
    data=cPickle.load(load_file)
    labels=data[1]
    features=data[0]


    rates=[]
    for i in xrange(1,20):
        trainfeatures, testfeatures, trainlabels, testlabels=get_train_test(data)
        # testerror=simpleSVM(trainfeatures, testfeatures, trainlabels, testlabels)
        testerror=RBM_SVM(trainfeatures, testfeatures, trainlabels, testlabels)
        print "test run %d, error rate is %1.4f " %(i, testerror)
        rates.append(testerror)

    mean_error=np.array(rates).mean()
    print "average error rate is %1.4f" %mean_error

    # trainfeatures, testfeatures, trainlabels, testlabels=get_train_test(data)




# test run 1: average error rate: 0.4252
    # clf = svm.SVC(C=5.0, kernel='poly', degree=3, gamma=0.5, coef0=10.0, shrinking=True,
    # probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False,
    # max_iter=-1, random_state=None)


# test run 2: average error rate: 0.4087
    # clf = svm.SVC(C=5.0, kernel='rbf', degree=3, gamma=0.5, coef0=10.0, shrinking=True,
    # probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False,
    # max_iter=-1, random_state=None)


# test run 3: average error rate: 0.3841, 0.3855 0.4224 0.3978 0.4040 0.4163 0.3835 ----- best sigmoid so far
    # clf = svm.SVC(C=5.0, kernel='sigmoid', degree=3, gamma=0.5, coef0=10.0, shrinking=True,
    # probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False,
    # max_iter=-1, random_state=None)



# test run 4: average error rate: 0.4087
#     clf = svm.SVC(C=5.0, kernel='sigmoid', degree=3, gamma=0.5, coef0=15.0, shrinking=True,
#     probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False,
#     max_iter=-1, random_state=None)


# test run 5: average error rate: 0.4033
    # clf = svm.SVC(C=5.0, kernel='sigmoid', degree=3, gamma=0.5, coef0=7.5, shrinking=True,
    # probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False,
    # max_iter=-1, random_state=None)


# test run 6: average error rate: 0.4108
    # clf = svm.SVC(C=3.0, kernel='sigmoid', degree=3, gamma=0.5, coef0=10, shrinking=True,
    # probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False,
    # max_iter=-1, random_state=None)


# test run 7: average error rate: 0.4156
    # clf = svm.SVC(C=7.0, kernel='sigmoid', degree=3, gamma=0.5, coef0=10, shrinking=True,
    # probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False,
    # max_iter=-1, random_state=None)

# test run 8: average error rate: 0.3923, 0.4060
    # clf = svm.SVC(C=5.0, kernel='rbf', degree=3, gamma=2, coef0=10, shrinking=True,
    # probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False,
    # max_iter=-1, random_state=None)


# RBM_SVM test_run 1: 0.4053, 0.4060
#     # SVM parameters
#     clf = svm.SVC(C=5.0, kernel='sigmoid', degree=3, gamma=0.5, coef0=10.0, shrinking=True,
#     probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False,
#     max_iter=-1, random_state=None)
#
#     # RBM parameters
#     rbm = BernoulliRBM(random_state=0, verbose=True)
#     rbm.learning_rate = 0.06
#     rbm.n_iter = 20


