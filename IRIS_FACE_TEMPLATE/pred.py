# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 10:40:12 2021

@author: OKOK PROJECTS
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot
import seaborn

data  = pd.read_csv('Iris.csv')
data

# remove unnecessary columns
data = data.iloc[:,1:]
data.info()

data

x = data.iloc[:,:-1].values
x[:,0]

y = data.iloc[:,-1].values
y

# convert string to int mean machine-readable form.
from sklearn.preprocessing import LabelEncoder
y_labal = LabelEncoder()
y = y_labal.fit_transform(y)
y

#  Elbow method to find suitable k for clustering.
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