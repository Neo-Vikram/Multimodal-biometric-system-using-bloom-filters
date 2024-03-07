# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 14:42:35 2021

@author: OKOK PROJECTS
"""
import pandas as pd
import pickle
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random as rd

from sklearn.cluster import KMeans
from matplotlib import pyplot
import seaborn

# Reading the file and creating two lists 1.sequence list 2.dna list
def read_file(file_name):
    sequence_list = []
    dna_list = []
    dna_string_list  = []
    with open(file_name, 'r') as f:
        count = 0
        for i in f:
            if count % 2 == 0:
                sequence_list.append(i[1:-1])
            if count % 2 == 1:
                dna_list.append(list(i[:-1]))
                dna_string_list.append(i[:-1])
            count += 1             

    return sequence_list, dna_list, dna_string_list

# creating data frame and convert the strings to numbers 
def create_df(dna_list):
    dna_df = pd.DataFrame(dna_list)   
    dna_df.to_csv('FNMR_string.csv')
    DNA = {'A': 1, 'C': 2, 'G' : 3, 'T':4 }
    dna_df.replace(to_replace=DNA, inplace=True)
    return dna_df


#caluclating hamming distance
def hamming_df(dna_string_list):
    
    n = len(dna_string_list)
    
    main_dist_list = []
    for i in range(n):
        main_dist_list.append([0]*n)
    
    for i in range(n):
        for j in range(i+1,n):
           ham_dist = hamdist(dna_string_list[i],dna_string_list[j])
           main_dist_list[i][j] = ham_dist
           main_dist_list[j][i] = ham_dist
    hamming_df = pd.DataFrame(main_dist_list)
    return hamming_df
    

def hamdist(str1, str2):
    diffs = 0
    for ch1, ch2 in zip(str1, str2):
        if ch1 != ch2:
            diffs += 1
    return diffs

# saving the data frame and the sequence_list using pickle
def save_df(df, ham_df, sequence_list):
    df.to_csv('FNMR.csv')
    ham_df.to_csv('FNMR_ham.csv')
    with open('seq_list.pkl','wb') as f:
        pickle.dump(sequence_list, f)
        
def main():
    file_name = "FNMR.fas"
    seq_list, dna_list, dna_string_list = read_file(file_name)
    df = create_df(dna_list)
    ham_df = hamming_df(dna_string_list)
    save_df(df, ham_df, seq_list)

if __name__ == "__main__":
    main()
    
#Framework for the parameter estimation in bloom filter-based template protection schemes

def main(): 
    ham_df = pd.read_csv("FNMR_ham.csv", index_col = 0)
    model = MDS(n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=None, random_state=1,dissimilarity='precomputed')
    mds_output = model.fit_transform(ham_df)
    principalDf = pd.DataFrame(data = mds_output, columns = ['X', 'Y'])
    principalDf.to_csv("FNMR_mds.csv")
    plt.scatter(mds_output[:, 0], mds_output[:, 1],label='MDS')
    plt.title('nWords estimation')
if __name__ == "__main__":
    main()
    
    
#Visualizing plots for MDS with different clusters highlighted in different colors Computation of nBits.
def kmeans(file_name, num_clusters, mode):
    
    X = pd.read_csv(file_name,index_col=0)
    m = X.shape[0]
    n = X.shape[1]
    
    n_iter=100
    K = num_clusters # number of clusters
    
    Centroids=np.array([]).reshape(n,0)
    for i in range(K):
        rand=rd.randint(0,m-1)
        Centroids=np.c_[Centroids,X.iloc[rand]]

    Output={}
    EuclidianDistance=np.array([]).reshape(m,0)
    for k in range(K):
        tempDist=np.sum((X-Centroids[:,k])**2,axis=1)
        EuclidianDistance=np.c_[EuclidianDistance,tempDist]
    C = np.argmin(EuclidianDistance,axis=1) + 1

    Y={}
    for k in range(K):
        Y[k+1]=np.array([]).reshape(2,0)
    for i in range(m):
        Y[C[i]]=np.c_[Y[C[i]],X.iloc[i]]        
    for k in range(K):
        Y[k+1]=Y[k+1].T        
    for k in range(K):
        Centroids[:,k]=np.mean(Y[k+1],axis=0)
    
    for j in range(n_iter):
        #step 2.a
        EuclideanDistance=np.array([]).reshape(m,0)
        for k in range(K):
            tempDist=np.sum((X-Centroids[:,k])**2,axis=1)
            EuclideanDistance=np.c_[EuclideanDistance,tempDist]
        C=np.argmin(EuclideanDistance,axis=1)+1
        #step 2.b
        Y={}
        for k in range(K):
            Y[k+1]=np.array([]).reshape(2,0)
        for i in range(m):
            Y[C[i]]=np.c_[Y[C[i]],X.iloc[i]]
        
        for k in range(K):
            Y[k+1]=Y[k+1].T
        
        for k in range(K):
            Centroids[:,k]=np.mean(Y[k+1],axis=0)
        if j > 0 and all(np.array_equal(Y[key], Output[key]) for key in Y):
            break
        Output = Y

    color=['yellow','blue','green','cyan','black','magenta','brown']
    labels=['nWords=16','nWords=20','nWords=32','nWords=53', 'cluster5', 'cluster6']

    for k in range(K):
        plt.scatter(Output[k+1][:,0],Output[k+1][:,1],c=color[k],label=labels[k])
    
    plt.scatter(Centroids[0,:],Centroids[1,:],s=30,c='red',label='Centroids')
    plt.xlabel('Face')
    plt.ylabel('Iris')
    plt.legend()
    plt.show('clustered_{}_{}.png'.format(K, mode))
    plt.clf()

    return Output, Centroids
# Bloom filter without losing its discriminative power preventing accuracy degradation

def sum_of_squares(Output,Centroids,num_clusters):
    sum_of_squares = 0
    Centroids = Centroids.T
    for k in range(num_clusters):
        sum_of_squares += np.sum((Output[k+1] - Centroids[k,:]**2))
    return sum_of_squares

def plot_elbow(sum_of_squares,mode):
    k_array = np.arange(3,7)
    plt.plot(k_array, sum_of_squares)
    plt.xlabel('False Match Rate')
    plt.ylabel('False Non Match Rate')
    plt.title('Accuracy Analysis Large-Scale Face')
    plt.show("elbow_{}.png".format(mode))

def main():
    sum_of_square_mds = np.array([])
    sum_of_square_pca = np.array([])
    for i in range(3,7):
        Output, Centroids = kmeans("FNMR_MDS.csv",i, "mds")
        sum_of_square_mds = np.append(sum_of_square_mds,sum_of_squares(Output, Centroids, i))

    plot_elbow(sum_of_square_mds,"mds")

if __name__ == "__main__":
    main()


data  = pd.read_csv('Iris.csv')
data


data = data.iloc[:,1:]
data.info()
data.isnull().sum()
data
print("",data.describe().T)
x = data.iloc[:,:-1].values
x[:,0]

y = data.iloc[:,-1].values
y

# each comparison between two bits from two different binary templates is essentially a Bernoulli trial, where correlations between successive “coin tosses” are a consequence of the existing correlations in the biometric samples.
from sklearn.preprocessing import LabelEncoder
y_labal = LabelEncoder()
y = y_labal.fit_transform(y)
y


sum_squer_error = []
for i in range(1,11):
    km = KMeans(n_clusters=i)
    km.fit(x)
    sum_squer_error.append(km.inertia_)
pyplot.xlabel('nWords')
pyplot.ylabel('Re-Map Probability ')
pyplot.plot(range(1,11),sum_squer_error)
pyplot.title('nWords')
pyplot.grid(True)
pyplot.show()
#  from elbow method K= 3.
kmeans = KMeans(n_clusters=3)

kmeans.fit(x)

y_means = kmeans.predict(x)

y_means

pd.DataFrame({'precdiction':y_means,'actual':y})

pyplot.scatter(x[:,0],x[:,2],c = y_means)
pyplot.xlabel('value1')
pyplot.ylabel('value2')
pyplot.title('Accuracy evaluation:')
pyplot.show()

pyplot.scatter(data['template'],data['value1'])
pyplot.show()
import seaborn
seaborn.swarmplot(x = 'template',y= 'value1',data=data)