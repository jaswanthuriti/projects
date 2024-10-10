import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import pickle
import bz2
train_set = pd.read_csv("C:\\Users\\kensa\\train_set.csv");
train_target = train_set['Diabetic']
train_set.drop(columns={'Diabetic'},inplace=True)
model = RandomForestClassifier()
model.fit(train_set, train_target)
ofile = bz2.BZ2File('m.pkl','wb')
pickle.dump(model, ofile)
ofile.close()