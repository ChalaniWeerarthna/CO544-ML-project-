import numpy as np
import pandas as pd
import csv

#read csv to
df_dataset = pd.read_csv( 'data.csv')
test_dataset = pd.read_csv( 'testdata.csv')  # test dataset
#test_dataset = pd.read_csv( 'testdata.csv',names =['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15'])# set names

df_dataset_copy = df_dataset  #make copy of original training dataset
test_dataset_copy = test_dataset   #make a copy of original testing dataset

#print dataset
df_dataset.head()

df_dataset.shape

col_list = df_dataset.columns
"""for col_name in col_list:
  print(df_dataset[col_name].unique())"""

#######handle missing values
#find null value
df_dataset.isnull().sum()

print("\n\ndata types of the original training dataset: \n")
print(df_dataset.dtypes)

#remove ? from df_dataset

col_list_obj = ['A1', 'A2', 'A3', 'A4', 'A6', 'A9']
for col_name in col_list:
  df_dataset = df_dataset[df_dataset[col_name] != '?']
  #print(df_dataset[col_name].mode())
  #df_dataset[col_name].replace(['?'], [df_dataset[col_name].mode()[0]], inplace=True)
  #df_dataset = df_dataset[df_dataset[col_name] != '?']

df_dataset['A2'] = df_dataset['A2'].astype(float)
df_dataset['A14'] = df_dataset['A14'].astype(int)

df_dataset.dtypes

#boolean convet to 1, 0
df_dataset.A8 = df_dataset.A8.astype(int)
df_dataset.A11 = df_dataset.A11.astype(int)
df_dataset.A13 = df_dataset.A13.astype(int)
df_dataset.head()

#convert values to numaric
df_dataset.A1.replace(['a', 'b'], [0, 1], inplace=True)
df_dataset.A3.replace(['u', 'y', 'l'], [0, 1, 2], inplace=True)
df_dataset.A4.replace(['g', 'p', 'gg'], [0, 1, 2], inplace=True)
df_dataset.A6.replace(['c', 'd', 'cc', 'i', 'j', 'k', 'm', 'r', 'q', 'w', 'x', 'e', 'aa', 'ff'], [0, 1, 2, 3, 4, 5, 6, 7, 8 ,9 ,10 ,11 ,12 ,13], inplace=True)
df_dataset.A9.replace(['v', 'h', 'bb', 'j', 'n', 'z', 'dd', 'ff', 'o'], [0, 1, 2, 3, 4, 5, 6, 7, 8], inplace=True)
df_dataset.A15.replace(['g', 'p', 's'], [0, 1, 2], inplace=True)
df_dataset.A16.replace(['Failure','Success'], [0, 1], inplace=True)

#dataframe after converting numaric
df_dataset.head()

#Change datatypes

df_dataset.describe()

#Plot Histrogram

import matplotlib.pyplot as plt

success = df_dataset.loc[df_dataset['A16'] == 1]
failure = df_dataset.loc[df_dataset['A16'] == 0]

bins = np.linspace(0, 28, 30)

plt.figure(figsize=(10,5))

plt.hist(success['A5'], bins, alpha=0.5, label='Success')
plt.hist(failure['A5'], bins, alpha=0.5, label='Failure')
plt.legend(loc='upper right')
plt.title('Histogram | A5')
plt.show()

bins = np.linspace(0, 100000, 30)

plt.figure(figsize=(10,5))

plt.hist(success['A7'], bins, alpha=0.5, label='Success')
plt.hist(failure['A7'], bins, alpha=0.5, label='Failure')
plt.legend(loc='upper right')
plt.title('Histogram | A7')
plt.show()

bins = np.linspace(0, 67, 30)

plt.figure(figsize=(10,5))

plt.hist(success['A12'], bins, alpha=0.5, label='Success')
plt.hist(failure['A12'], bins, alpha=0.5, label='Failure')
plt.legend(loc='upper right')
plt.title('Histogram | A12')
plt.show()

#Univariant Selection
#Selected the best features base on univarient statistics

X = df_dataset.drop("A16",1)   #Feature Matrix
y = df_dataset["A16"]          #Target Variable

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=f_classif, k=15)

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Feature','Score']  #naming the dataframe columns

print("\n\nScores given to the attributes : \n")
print(featureScores.nlargest(16,'Score'))  #score of features

selectedFeatures = featureScores.nlargest(16,'Score')
df_15F = X[selectedFeatures['Feature'].values]
df_15F.head()

#Accuracy

def randomForest(dataFrame, target):

  #Create a RF Classifier
  clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=42)
  scores = cross_val_score(clf, dataFrame, target, cv=3)

  return scores.mean()

def returnScoreDataFrameModels(dataFrame):
  lists3 = []

  for i in [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
    lists3.append(randomForest(dataFrame.iloc[:,0:(i)], y))

  rows = ["randomForest"]

  data = np.array([lists3])
  randomForestScore = pd.DataFrame(data=data, index=rows).transpose()

  return randomForestScore

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
import matplotlib.pyplot as plt

randomForestScore = returnScoreDataFrameModels(df_15F)

pcaScore = ["15", "14", "13", "12", "11", "10", "9", "8", "7", "6", "5", "4", "3", "2", "1"]

plt.plot(pcaScore, randomForestScore["randomForest"], label='linear')
plt.title(' Random Forest Score of features')
plt.show()

#Train Model

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics


def printClassificationResults(X_train, X_test, y_train, y_test):

  dt_clf = DecisionTreeClassifier(random_state=41,max_leaf_nodes=3)
  score_dt = cross_val_score(dt_clf, X_train, y_train, cv=3)
  dt_clf.fit(X_train, y_train)

  RF_clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=42)
  score_RF = cross_val_score(RF_clf, X_train, y_train, cv=3)
  RF_clf.fit(X_train, y_train)
 

  y_pred_dt = dt_clf.predict(X_test)
  y_pred_RF = RF_clf.predict(X_test)


  # comparing actual response values (y_test) with predicted response values (y_pred)
  print("\t\t\t\t\t\t\t Testing\t Training")
  print("Decision Tree model accuracy(in %) \t\t\t:", round(metrics.accuracy_score(np.int64(y_test.values), y_pred_dt)*100,4) ,"\t", round(score_dt.mean()*100,2))
  print("Random forest model accuracy(in %) \t \t\t:" , round(metrics.accuracy_score(np.int64(y_test.values), y_pred_RF)*100,4) ,"\t", round(score_RF.mean()*100,2))

  
selected_col_names = ["A1",	"A2",	"A3",	"A4", "A5",	"A6",	"A7",	"A8", "A9", "A10",	"A11",	"A12", "A13", "A14","A15"]
#selected_col_names = ["A2",  "A3",  "A4",  "A5",  "A7",  "A8",  "A9",  "A10",   "A11",	"A12"]
#selected_col_names = ["A11"]

X_selected = X[selected_col_names]

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=1)


printClassificationResults(X_train, X_test, y_train, y_test)


#let's choose Random forest model-----------------------------------------------------------------------------------------
test_col_list = test_dataset.columns

#remove ? from df_dataset

##############test_col_list_obj = ['A1', 'A2', 'A3', 'A4', 'A6', 'A9']
test_col_list_obj = ["A1",	"A2",	"A3",	"A4", "A5",	"A6",	"A7",	"A8", "A9", "A10",	"A11",	"A12", "A13", "A14","A15"]
for col_name in test_col_list:
  test_dataset = test_dataset[test_dataset[col_name] != '?'] # rows with missing values have removed.

test_dataset['A2'] = test_dataset['A2'].astype(float)
test_dataset['A14'] = test_dataset['A14'].astype(int)

test_dataset.dtypes

#find frq of values in col for categorical data

most_frq_val = []
test_col_list_obj = ['A1', 'A3', 'A4', 'A6', 'A9', 'A15']

for col_name in test_col_list_obj:
  most_frq_val.append(test_dataset[col_name].mode()[0])
  #df_dataset[col_name].replace(['?'], [df_dataset[col_name].mode()[0]], inplace=True)


#print(most_frq_val)


#categorical data replace with most frq val

i=0
for col_name in test_col_list_obj:
  test_dataset_copy[col_name].replace(['?'], [most_frq_val[i]], inplace=True)
  i=i+1


#numaric val replace with mean of col

#################test_col_list_obj_num = ['A2', 'A5', 'A7', 'A10', 'A12', 'A14']
test_col_list_obj_num = ['A2', 'A5', 'A7','A8', 'A10','A11', 'A12','A13', 'A14']
for col_name in test_col_list_obj_num:
  test_dataset_copy[col_name] = test_dataset_copy[col_name].astype(object)
  test_dataset_copy[col_name].replace(['?'], [test_dataset[col_name].mean()], inplace=True)
  test_dataset_copy[col_name] = test_dataset_copy[col_name].astype(float)

test_dataset = test_dataset_copy

#print(test_dataset.head())

test_dataset.A8 = test_dataset.A8.astype(int)
test_dataset.A11 = test_dataset.A11.astype(int)
test_dataset.A13 = test_dataset.A13.astype(int)

#print(df_dataset.dtypes)
test_dataset.A1.replace(['a', 'b'], [0, 1], inplace=True)
test_dataset.A3.replace(['u', 'y', 'l'], [0, 1, 2], inplace=True)
test_dataset.A4.replace(['g', 'p', 'gg'], [0, 1, 2], inplace=True)
test_dataset.A6.replace(['c', 'd', 'cc', 'i', 'j', 'k', 'm', 'r', 'q', 'w', 'x', 'e', 'aa', 'ff'], [0, 1, 2, 3, 4, 5, 6, 7, 8 ,9 ,10 ,11 ,12 ,13], inplace=True)
test_dataset.A9.replace(['v', 'h', 'bb', 'j', 'n', 'z', 'dd', 'ff', 'o'], [0, 1, 2, 3, 4, 5, 6, 7, 8], inplace=True)
test_dataset.A15.replace(['g', 'p', 's'], [0, 1, 2], inplace=True)

print("\n\nTesting dataset after processing\n")
print(test_dataset.head())

# select input and output dataset for training and testing
X_train = X[selected_col_names]
X_test = test_dataset[selected_col_names]
Y_train = y


model = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=42)
model.fit(X_train, Y_train)
predictions = model.predict(X_test)

print("\n\nPredictions: \n")
print(predictions)

# print predictions into a csv file #predictions.csv#
my_dict= {0:'Failure',1:'Success'}
predictions = [my_dict[zi] for zi in predictions]

print("\n\nPrediction results for testdata : \n")

print(predictions)


#predictions=pd.DataFrame(predictions, columns=['predictions']).to_csv('testdata_10%.csv')

#### create new csv file with the predictions
with open('testdata.csv','r') as csvinput:
    with open('predictions.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        reader = csv.reader(csvinput)

        all = []
        row = next(reader)
        row.append('A16')
        all.append(row)

        x = 0
        for row in reader:
          row.append(predictions[x])
          x = x + 1
          all.append(row)

        writer.writerows(all)
