# -*- coding: utf-8 -*-
"""
Predicitve_Analytics.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import sklearn
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from numpy import linalg as la
from sklearn.preprocessing import MinMaxScaler
from random import sample
import random
from scipy import stats
import time
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d
sns.set(color_codes=True)
sns.set_style('whitegrid')
sns.set_context('paper',font_scale=2)

df=pd.read_csv('data.csv')
xd=np.array(df[df.columns[0:48]])
yd=np.array(df['48'])
X_train=xd[0:300]
Y_train=yd[0:300].reshape(-1,1)
X_test=xd[300:400]
Y_test=yd[300:400]
#----------------------------------------------------------------PART 1 --------------------------------------------
#Print(Accuracy(Y_test,Y_test))
def Accuracy(y_true,y_pred):
    y=y_pred-y_true
    acc=((np.sum(np.where(y==0,1,0))/y_pred.shape[0])*100)
    return acc

def Recall(y_true,y_pred):
    y_final = confusion_matrix(y_true, y_pred)
    rec = 0
    for i in range(0, y_final.shape[1]):
        rec = rec + y_final[i][i] / np.sum(y_final[:, i])
    return rec / y_final.shape[1]

def Precision(y_true,y_pred):
    y_final = confusion_matrix(y_true, y_pred)
    pre = 0
    for i in range(0, y_final.shape[0]):
        pre = pre + y_final[i][i] / np.sum(y_final[i, :])
    return pre / y_final.shape[0]

def WCSS(Clusters):
    ClusteredPoints=Clusters
    k = len(ClusteredPoints)
    print(k)
    TotalSum =0
    for i in range(k):
        X_Clusteri = ClusteredPoints[i]
        rows_of_clusteri = X_Clusteri.shape[0]
        cols = X_Clusteri.shape[1]
        X_rows_matrix = np.ones((rows_of_clusteri, 1), dtype = int) # r x 1
        centroid_i = ((X_Clusteri.transpose()).dot(X_rows_matrix)).transpose()
        Centroid_i_matrix = np.tile(centroid_i,(rows_of_clusteri,1))
        Difference_to_centroid = X_Clusteri - Centroid_i_matrix
        d_ones = np.ones((cols, 1), dtype = int) # d x 1
        Square_Distance =  (np.square(Difference_to_centroid)).dot(d_ones)
        ones_needed = np.ones((rows_of_clusteri, 1), dtype = int)
        Sum_of_Squares = (ones_needed.transpose()).dot(Square_Distance)
        TotalSum = TotalSum + Sum_of_Squares
    return np.asscalar(TotalSum)

def ConfusionMatrix(y_true,y_pred):
    K = len(np.unique(y_true))
    Conf_Matrix = (y_true * 10) + y_pred
    a, b = np.histogram(Conf_Matrix, bins=11 * 11)
    return a.reshape(11, 11)
    

def KNN(X_train, X_test, Y_train, k):
    scalar = MinMaxScaler()
    scalar.fit(X_train)
    X_train = scalar.transform(X_train)
    scalar.fit(X_test)
    X_test = scalar.transform(X_test)

    diff = np.ones((1, X_train.shape[0]))
    Finalmode = []
    for i in range(0, X_test.shape[0]):
        ct = X_test[i]
        ct = ct.reshape(-1, 1)
        ED = np.square((X_train - np.dot(ct, diff).transpose()))

        Identity = np.ones((ED.shape[1], 1))
        ED = np.sqrt(np.dot(ED, Identity))
        FinalX_train = np.append(X_train, ED, axis=1)
        FinalX_train = np.append(FinalX_train, Y_train, axis=1)
        FinalX_train = FinalX_train[FinalX_train[:, 48].argsort()]
        counts = stats.mode(FinalX_train[:k, 49], axis=None)
        Finalmode.append(int(counts.mode))
    Y_pred = np.asarray(Finalmode)
    return Y_pred

def split_cond(index,xtrain_col,xtrain):
    i=index
    dic_left={}
    dic_right={}
    val=np.mean(xtrain_col)
    xtrain_left=xtrain[xtrain[:,i]<=val]
    xtrain_right=xtrain[xtrain[:,i]>val]
    summation_left=0
    summation_right=0
    for k in range(1,int(max(xtrain[:,-1])+1)):
        dic_left[k]=np.sum(np.where(xtrain_left[:,-1]==k,1,0))
        summation_left+=(dic_left[k]/xtrain_left.shape[0])**2
        dic_right[k]=np.sum(np.where(xtrain_right[:,-1]==k,1,0))
        summation_right+=(dic_right[k]/xtrain_right.shape[0])**2
    left_imp=(1-summation_left)*(xtrain_left.shape[0]/xtrain.shape[0])
    right_imp=(1-summation_right)*(xtrain_right.shape[0]/xtrain.shape[0])
    total=left_imp+right_imp
    return total,val

def best_split(xtrain,depth,feature_list):
    least=1
    node={}
    li=[]
    node['index']=feature_list[0]
    node['value']=np.mean(xtrain[:,node['index']].reshape(-1,1))
    for k in range(1,12):
            li.append(np.sum(np.where(xtrain[:,-1]==k,1,0)))
    for i in feature_list:
        imp,val=split_cond(i,xtrain[:,i].reshape(-1,1),xtrain)
        if(least>=imp):
            least=imp
            node['value']=val
            node['index']=i
    index=node['index']
    value=np.mean(xtrain[:,index].reshape(-1,1))
    depth=depth+1
    li=[]
    for k in range(1,12):
            li.append(np.sum(np.where(xtrain[:,-1]==k,1,0)))
    term=np.array(li)
    if(np.sum(np.where(term[:]==0,0,1))>1 and sum(li)>50):
        node['left']=best_split(xtrain[xtrain[:,node['index']]<=value],depth,feature_list)
        node['right']=best_split(xtrain[xtrain[:,node['index']]>value],depth,feature_list)
    else:
        node['class']=int(np.where(term==np.amax(term))[0][0])+1
    return node

def treePredict(xtest,node):
    ypred=np.zeros((xtest.shape[0],1))
    main=node
    for i in range(0,xtest.shape[0]):
        node=main
        while(1):
            ind=node['index']
            if(xtest[i][ind]<=node['value']):
                if('left' in node.keys()):
                    node=node['left']
                else:
                    ypred[i][0]=node['class']
                    break
            else:
                if('right' in node.keys()):
                    node=node['right']
                else:
                    ypred[i][0]=node['class']
                    break
    return ypred

def RandomForest(X_train,Y_train,X_test):
    t=time.time()
    xtrain=X_train
    ytrain=Y_train
    xtest=X_test
    xtrain=np.append(xtrain,ytrain,axis=1)
    ypred_final=np.zeros((X_test.shape[0],1))
    for i in range(0,10):
        feature_list=[]
        feature_list=list(np.random.randint(0,xtrain.shape[1]-1,size=7))
        a=np.random.choice(xtrain.shape[0],40000)
        node=best_split(xtrain[np.random.randint(0,xtrain.shape[0],size=32000)],0,feature_list)
        depth=0
        ypred=treePredict(xtest,node)
        ypred_final=np.append(ypred_final,ypred,axis=1)
    ypred_final=ypred_final[:,1:]
    b,c=stats.mode(ypred_final,axis=1)
    return ypred.astype(int)
#Y_pred=RandomForest(X_train,Y_train.reshape(-1,1),X_test)
#print(Y_pred,Y_pred.shape)
  
def PCA(X_train,N):   
    covmatrix = np.cov(X_train.transpose())
    w, v = np.linalg.eig(covmatrix)
    idx = np.argsort(w)
    w = w[idx]
    v = v[:, idx]
    Mat = np.dot(X_train, v[:, -N:])
    return Mat
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: numpy.ndarray
    """
def Kmeans(X_train,N):
    # X_train = X_train[0:-2]# remove last column
    k = 11  # should be assigned at start
    cols = X_train.shape[1]
    rows = X_train.shape[0]
    # initialize centroids
    Random_indices = np.random.randint(rows, size=k)
    oldcentroids = X_train[Random_indices, :]  # k x d matrix
    iterations = 0
    iter_count = 1  # number of needed iterations
    Clusters = []
    Newcentroids = []
    ClusteredPoints = []
    while (iterations < iter_count):  # ( not np.array_equal(oldcentroids , Newcentroids)
        iterations = iterations + 1
        ClusteredPoints = []
        # print(iterations)
        if (np.array_equal(oldcentroids, Newcentroids)):
            print("yes")
        else:
            print("NO")
        if iterations != 1:
            oldcentroids = Newcentroids
        Cdistance = []
        distance = []
        for i in range(oldcentroids.shape[0]):
            individualCentroid = (oldcentroids[i])
            Ctile = np.tile(individualCentroid, (rows, 1))  # N x d
            difference = Ctile - X_train  # N x d
            d_ones = np.ones((cols, 1), dtype=int)  # d x 1
            distance = (np.square(difference)).dot(d_ones)  # (N x d) x (d x 1) = N x 1
            # distance of every point to that centroid
            if i == 0:
                Cdistance = distance
            else:
                Cdistance = np.concatenate((Cdistance, distance),
                                           axis=1)  # after k iterations it will be N x k matrix ;
        Clusters = np.argmin(Cdistance, axis=1).reshape(-1, 1)
        XandClusters = np.concatenate((X_train, Clusters), axis=1)
        # update centriods
        Newcentroids = []
        for i in range(k):
            # filter rows of centroid i
            indices = np.where(XandClusters[:, -1] == i)
            Ci_rows = XandClusters[indices]
            Ci_rows = np.delete(Ci_rows, -1, axis=1)
            Ci_count = Ci_rows.shape[0]
            # print(i, Ci_count)
            # print()
            Ci_ones = np.ones((Ci_count, 1), dtype=int)
            New_Ci = (1 / Ci_count) * ((Ci_rows.transpose()).dot(Ci_ones)).transpose()
            if i == 0:
                Newcentroids = New_Ci
                # Ci_rows = np.expand_dims(Ci_rows, axis=0)
                # print(Ci_rows.shape)
                ClusteredPoints.append(Ci_rows)
            else:
                Newcentroids = np.concatenate((Newcentroids, New_Ci), axis=0)
                # Ci_rows = np.expand_dims(Ci_rows, axis=0)
                # print(Ci_rows.shape)
                ClusteredPoints.append(Ci_rows)
                # print(ClusteredPoints)
        dis = WCSS(ClusteredPoints)        
    return ClusteredPoints
"""
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: List[numpy.ndarray]
    """
#---------------------------------------------------------------PART 2 --------------------------------------------
def sklearnLogisticRegression(X_train,Y_train,X_test):
    X=X_train
    Y=Y_train
    xtest=X_test
    clf = LogisticRegression(max_iter=200,solver='liblinear').fit(X, Y)
    ypred=clf.predict(xtest)
    return ypred.reshape(-1,1)

def sklearnSVM(X_train,Y_train,X_test):
    X=X_train
    Y=Y_train
    xtest=X_test
    clf=svm.SVC(kernel='linear')
    clf.fit(X,Y)
    ypred=clf.predict(xtest)
    return ypred.reshape(-1,1)

def sklearnDecisionTree(X_train,Y_train,X_test):
    X=X_train    
    Y=Y_train    
    xtest=X_test    
    clf = tree.DecisionTreeClassifier()    
    clf = clf.fit(X, Y)    
    ypred=clf.predict(xtest)    
    return ypred.reshape(-1,1)

def sklearnKNN(X_train,Y_train,X_test):
    X=X_train
    Y=Y_train
    xtest=X_test
    clf=KNeighborsClassifier()
    scaler=MinMaxScaler()
    scaler.fit(X)
    X=scaler.transform(X)
    clf.fit(X,Y)
    scaler.fit(xtest)
    xtest=scaler.transform(xtest)
    ypred=clf.predict(xtest)
    return ypred.reshape(-1,1)

def sklearnVotingClassifier(X_train,Y_train,X_test):
    xtrain =X_train
    ytrain =Y_train
    xtest=X_test
    ypred_svm=sklearnKNN(xtrain,ytrain,xtest)
    ypred_svm=sklearnSVM(xtrain,ytrain,xtest)
    ypred_logistic=sklearnLogisticRegression(xtrain,ytrain,xtest)
    ypred_tree=sklearnDecisionTree(xtrain,ytrain,xtest)
    ypred_final=np.concatenate((ypred_logistic.reshape(-1,1),ypred_tree.reshape(-1,1)),axis=1)
    ypred,count=stats.mode(ypred_final,axis=1)
    return ypred.reshape(-1,1)

def SklearnSupervisedLearning(X_train,Y_train,X_test)
    ypred =[]
    ypred_logistic = sklearnLogisticRegression(X_train,Y_train,X_test)
    ypred.append(ypred_logistic)
    ypred_svm = sklearnSVM(X_train,Y_train,X_test)
    ypred.append(ypred_svm)
    ypred_Dtree = sklearnDecisionTree(X_train,Y_train,X_test)
    ypred.append(ypred_Dtree)
    ypred_KNN = sklearnKNN(X_train,Y_train,X_test)
    ypred.append(ypred_KNN)
    ypred_VC = sklearnVotingClassifier(X_train,Y_train,X_test)
    ypred.append(ypred_VC)
    
    return ypred

SklearnSupervisedLearning(X_train,Y_train,X_test)
    
"""
Please execute the 'sklearnGridPlots(X_train,Y_train)' function to 
see the visualizations of grid search  for supervised learning algorithms"""
def ConfusionMatrixPlots(Y_test,X_train,Y_train,X_test):
    fig = plt.figure(figsize=(30,6))
    ax = plt.subplot(1,5,1)
    correlation_matrix = ConfusionMatrix(Y_test,sklearnKNN(X_train,Y_train,X_test).reshape(-1))
    ax = sns.heatmap(data=correlation_matrix, annot=True)
    ax.set_title('KNearestNeighborsClassifier')
    ax = plt.subplot(1,5,2)
    correlation_matrix = ConfusionMatrix(Y_test,sklearnSVM(X_train,Y_train,X_test).reshape(-1))
    ax = sns.heatmap(data=correlation_matrix, annot=True)
    ax.set_title('SVM')
    ax = plt.subplot(1,5,3)
    correlation_matrix = ConfusionMatrix(Y_test,sklearnDecisionTree(X_train,Y_train,X_test).reshape(-1))
    ax = sns.heatmap(data=correlation_matrix, annot=True)
    ax.set_title('Decision Tree')
    ax = plt.subplot(1,5,4)
    correlation_matrix = ConfusionMatrix(Y_test,sklearnLogisticRegression(X_train,Y_train,X_test).reshape(-1))
    ax = sns.heatmap(data=correlation_matrix, annot=True)
    ax.set_title('Logistic Regression')
    ax = plt.subplot(1,5,5)
    correlation_matrix = ConfusionMatrix(Y_test,sklearnVotingClassifier(X_train,Y_train,X_test).reshape(-1))
    ax = sns.heatmap(data=correlation_matrix, annot=True)
    ax.set_title('Voting Classifier')
    
"""-----------------------------------------------------------------------------CONFUSION MATRIX OF ALL MENTIONED MODELS ------------

Please execute the 'ConfusionMatrixPlots(Y_test,X_train,Y_train,X_test)' function to 
see the visualizations of confusion matrices  for supervised learning algorithms"""

def sklearnGridKNN(X_train,Y_train):
    X=X_train
    Y=Y_train
    parameters={'n_neighbors': [10,11,12,13,14,15]}
    clf = KNeighborsClassifier()
    grid=GridSearchCV(clf,parameters)
    grid.fit(X, Y)
    plt.xticks(parameters['n_neighbors'])
    plt.yticks(grid.cv_results_['std_test_score'], " ")
    plt.xlabel('Num of Neighbors')
    plt.ylabel('Accuracy')
    plt.plot(parameters['n_neighbors'],grid.cv_results_['std_test_score'])

def sklearnGridSVM(X_train,Y_train):
    X=X_train
    Y=Y_train
    parameters={'C': [1,2,3,4,5,6,7,8,9,10]}
    clf = svm.SVC(kernel='linear')
    grid=GridSearchCV(clf,parameters)
    grid.fit(X, Y)
    plt.xticks(parameters['C'])
    plt.yticks(grid.cv_results_['std_test_score'], " ")
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.plot(parameters['C'],grid.cv_results_['std_test_score'])


def sklearnGridTree(X_train,Y_train):
    X=X_train
    Y=Y_train
    parameters={'max_depth': [1,2,3,4,5,6,7,8,9,10,11,12]}
    clf = tree.DecisionTreeClassifier()
    grid=GridSearchCV(clf,parameters)
    grid.fit(X, Y)
    plt.xticks(parameters['max_depth'])
    plt.yticks(grid.cv_results_['std_test_score'], " ")
    plt.xlabel('max_depth')
    plt.ylabel('Accuracy')
    plt.plot(parameters['max_depth'],grid.cv_results_['std_test_score'])
    
""" #-------------------------------------------------------------GRID PLOTS----------------------------------------------

Create your own custom functions for Matplotlib visualization of hyperparameter search. 
Make sure that plots are labeled and proper legends are used
"""
def sklearnGridPlots(X_train,Y_train):
    fig = plt.figure(figsize=(20,6))
    ax = plt.subplot(1,3,1)
    sklearnGridKNN(X_train,Y_train)
    ax.set_title('KNearestNeighborsClassifier')
    ax = plt.subplot(1,3,2)
    sklearnGridSVM(X_train,Y_train)
    ax.set_title('SVM')
    ax = plt.subplot(1,3,3)
    sklearnGridTree(X_train,Y_train)
    ax.set_title('Decision Tree')

    
    
    