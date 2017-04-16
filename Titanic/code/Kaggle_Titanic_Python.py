import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC, SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

from xgboost import XGBClassifier

import string

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from plot_learning_curve import plot_learning_curve

from time import time


import plotting as pt


import os

os.system('clear')


def input():
	raw_input('press enter...')

def print_hist(df,survival = None,bins = 25,xlabel = 'x data',ylabel = 'y data',title = 'Title',normed = False):
	temp_df = df.copy()
	temp_df = temp_df[-np.isnan(temp_df)]
	temp_df = sorted(temp_df)
	sns.distplot(temp_df,hist=True,bins=bins)

def print_seaborn(df,x,hue):
	sns.countplot(x = x,hue = hue, data = df)
	plt.show(block=False)
	# raw_input('Press [enter] to continue')
	# plt.close()

def seaborn_hist(df,subset,title='Title'):
	df.dropna(subset=[subset],how='all')
	g = sns.FacetGrid(df, col = 'Survived',col_wrap = 2, size = 5)
	g.map(sns.distplot,subset)
	plt.subplots_adjust(top = 0.9)
	g.fig.suptitle(title)
	plt.show(block=False)

def get_title(name):
	if '.' in name:
		return name.split(',')[1].split('.')[0].strip()
	else:
		return 'Unknown'

def replace_titles(x):
	title = x['Title']
	if title in ['Capt','Col','Don','Jonkheer','Major','Rev','Sir']:
		return 'Mr'
	elif title in ['the Countess','Mme','Lady']:
		return 'Mrs'
	elif title in ['Mlle','Ms']:
		return 'Ms'
	elif title == 'Dr':
		if x['Sex'] == 'male':
			return 'Mr'
		else:
			return 'Mrs'
	else:
		return title

def mean_analysis(df,column_vals):
	for val in column_vals:
		if val == 'Age':
			print 'Mean survival rate by age group'
			temp_df = df.groupby(pd.cut(df['Age'],np.arange(0,df['Age'].max()+10,10)))['Survived'].mean()
			print temp_df
			plt.figure()
			ax = temp_df.plot(kind = 'bar')
			ax.xlabel = val
			plt.show(block = False)
		elif val == 'Fare':
			print 'Mean survival rate by fare'
			temp_df = df.groupby(pd.cut(df['Fare'],np.arange(0,df['Fare'].max()+50,50)))['Survived'].mean()
			print temp_df
			plt.figure()
			ax = temp_df.plot(kind = 'bar')
			ax.xlabel = val
			plt.show(block = False)
		else:
			print 'Mean survival rate by ' + str(val)
			temp_df = df.groupby(val).Survived.mean()
			print temp_df
			plt.figure()
			sns.barplot(x = val, y = 'Survived', data = df, ci = None)
			# temp_df.plot(kind = 'bar')
			plt.show(block = False)

		print


def evaluation(classifiers,train_data,train_target,test_data,test_target):
	scores = []
	f_scores = []
	t0 = time()
	for clf in classifiers:
		clf.fit(train_data,train_target.values.ravel())
		y_pred = clf.predict(test_data)
		scores.append(clf.score(test_data,test_target.values.ravel()))
		f_score = 0.5 * (precision_recall_fscore_support(test_target,y_pred)[2][0] + precision_recall_fscore_support(test_target,y_pred)[2][1])
		f_scores.append(f_score)
	print 'done in %0.3fs' % (time() - t0)
	return scores,f_scores


def main():

	# Import Data
	train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
	test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

	# Check to see if there are any null training set values
	print '********Checking NaNs for Test Data********'
	print 'Fare NaNs:', len(train[pd.isnull(train['Fare'])])
	print 'Class NaNs:', len(train[pd.isnull(train['Pclass'])])
	print 'Age NaNs:', len(train[pd.isnull(train['Age'])])
	print 'Sibling NaNs:', len(train[pd.isnull(train['SibSp'])])
	print 'Parent/Child NaNs:', len(train[pd.isnull(train['Parch'])])
	print 'Sex NaNs:', len(train[pd.isnull(train['Sex'])])
	print 'Embarked NaNs:', len(train[pd.isnull(train['Embarked'])])
	print '*******************************************'
	
	pt.plot_distribution( train , var = 'Age' , target = 'Survived' , row = 'Sex' )
	pt.plot_distribution( train , var = 'Fare' , target = 'Survived' , row = 'Sex' )
	input()

	# Replacing on fine grouping
	print 'Replacing Age NaNs with categorical means for Class, Sex, Siblings, Parent/Child'
	train['Age'] = train.groupby(['Pclass','Sex','SibSp','Parch'])['Age'].transform(lambda x: x.fillna(x.mean()))

	# Replacing on less granular gropuing
	print 'Checking Age NaNs after first replacement:', len(train[pd.isnull(train['Age'])])
	train['Age'] = train.groupby(['Pclass','Sex'])['Age'].transform(lambda x: x.fillna(x.mean()))

	print 'Checking Age NaNs after replacement:', len(train[pd.isnull(train['Age'])])
	train['Age'] = train['Age'].astype(int)

	train['Embarked'].fillna('S',inplace=True)
	print 'Checking Embarked NaNs after replacement:', len(train[pd.isnull(train['Embarked'])])

	seaborn_hist(train,'Age','Age Distributions Before Replacement')
	seaborn_hist(train,'Age', 'Age Distributions After Replacement')

	# Create a Family Size column
	train['Family_Size'] = train['SibSp'] + train['Parch']

	# Creating Titles column in DataFrame
	titles = sorted(set([x for x in train.Name.map(lambda x:get_title(x))]))
	print 'List of titles in data'
	print len(titles),':',titles
	train['Title'] = train['Name'].map(lambda x:get_title(x))
	train['Title'] = train.apply(replace_titles,axis = 1)

	print '*******************************************'

	# Determine the number of cabins reserved per person
	# print train['Deck'].unique()
	# raw_input('press enter...')
	# train['Cabin_Length'] = train['Cabin'].str.split(' ').str.len()
	# train['Cabin_Length'].fillna(0,inplace=True)

	column_vals = ['Sex','Fare','Age','Pclass','Family_Size','Title','Embarked']
	mean_analysis(train,column_vals)
	print '*******************************************'


	""" Creating deck from cabin, age label bands, fare label bands, titles from names, and applying LabelEncoder to categorical variables """	
	# Convert Categorical Variables to Numerical
	le_age = LabelEncoder()
	le_fare = LabelEncoder()
	le_title = LabelEncoder()
	le_embarked = LabelEncoder()
	le_deck = LabelEncoder()
	le_sex = LabelEncoder()

	train['Deck'] = train['Cabin'].str[0]
	train['Deck'].fillna('Z',inplace=True)

	age_labels = ['Band_1','Band_2','Band_3','Band_4','Band_5','Band_6','Band_7','Band_8','Band_9','Band_10']
	train['AgeBand'] = pd.cut(train['Age'],bins=10,labels=age_labels)
	train['AgeBand'] = le_age.fit_transform(train['AgeBand'])

	fare_labels = ['Band_1','Band_2','Band_3','Band_4','Band_5','Band_6','Band_7','Band_8','Band_9','Band_10']
	train['FareBand'] = pd.cut(train['Fare'],bins=10,labels=fare_labels)
	train['FareBand'] = le_fare.fit_transform(train['FareBand'])

	train['Title'] = le_title.fit_transform(train['Title'])

	train['Embarked'] = le_embarked.fit_transform(train['Embarked'])

	train['Deck'] = le_deck.fit_transform(train['Deck'])

	train['Sex'] = le_sex.fit_transform(train['Sex'])

	train.drop(['PassengerId','Name','Age','SibSp','Parch','Ticket','Fare','Cabin'],inplace=True,axis=1)
	print train.head()
	input()

	colormap = plt.cm.viridis
	plt.figure(figsize=(12,12))
	plt.title('Pearson Correlation of Features', y=1.05, size=15)
	sns.heatmap(train.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
	# pt.plot_correlation_map(train)
	plt.show(block=False)

	print train.corr()['Survived']
	input()


	""" Model Training """
	# Create DF with only numeric values for data
	X_train = train[['Pclass','Sex','Age','Family_Size','Cabin_Length','Fare','Embarked']]
	y_train = train['Survived']

	# Scale data
	scaler = preprocessing.StandardScaler().fit(X_train)
	X_train = scaler.transform(X_train)

	# Cross Validation
	train_data, test_data, train_target, test_target = train_test_split(X_train,y_train,test_size=0.5,random_state=0)

	# Trying a bunch of different classifiers
	classifiers = [LinearSVC(), KNeighborsClassifier(), DecisionTreeClassifier(max_depth=5), RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), MLPClassifier(alpha=1), GaussianNB(), SVC(gamma=2, C=1), AdaBoostClassifier(), XGBClassifier()]

	# print 'Classifier score:', clf.score(test_data,test_target.values.ravel())





	""" Model Analysis """
	# cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
	# title = "Learning Curves (XGBoost)"
	# plot_learning_curve(clf, title, X_train, y_train.values, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
	# y_pred = clf.predict(X_train)
	# print 'done in %0.3fs' % (time() - t0)
	# print confusion_matrix(y_train,y_pred)


	# print '*******************************************'
	# print classification_report(y_train,y_pred)
	# print 0.5 * (precision_recall_fscore_support(y_train,y_pred)[2][0] + precision_recall_fscore_support(y_train,y_pred)[2][1])
	# print '*******************************************'

	# print max(evaluation(classifiers,train_data,train_target,test_data,test_target)[:][1])
	_ , f_scores = evaluation(classifiers,train_data,train_target,test_data,test_target)
	print f_scores.index(max(f_scores))








	""" Cleaning Test Set """
	# Check to see if there are any null testing set values
	print '********Checking NaNs for Test Data********'
	print 'Fare NaNs:', len(test[pd.isnull(test['Fare'])])
	print 'Class NaNs:', len(test[pd.isnull(test['Pclass'])])
	print 'Age NaNs:', len(test[pd.isnull(test['Age'])])
	print 'Sibling NaNs:', len(test[pd.isnull(test['SibSp'])])
	print 'Parent/Child NaNs:', len(test[pd.isnull(test['Parch'])])
	print 'Sex NaNs:', len(test[pd.isnull(test['Sex'])])
	print 'Embarked NaNs:', len(test[pd.isnull(test['Embarked'])])
	print '*******************************************'


	# Replacing on fine grouping
	print 'Replacing Age NaNs with categorical means for Class, Sex, Siblings, Parent/Child'
	test['Age'] = train.groupby(['Pclass','Sex','SibSp','Parch'])['Age'].transform(lambda x: x.fillna(x.mean()))
	print 'Checking Age NaNs after first replacement:', len(test[pd.isnull(test['Age'])])
	test['Age'] = test['Age'].astype(int)

	# Replacing on fine grouping
	print 'Replacing Fare NaNs with categorical means for Class, Sex, Siblings, Parent/Child'
	test['Fare'] = train.groupby(['Pclass','Sex','SibSp','Parch'])['Fare'].transform(lambda x: x.fillna(x.mean()))
	print 'Checking Fare NaNs after first replacement:', len(test[pd.isnull(test['Fare'])])


	test['Cabin_Length'] = test['Cabin'].str.split(' ').str.len()
	test['Cabin_Length'].fillna(0,inplace=True)
	test['Family_Size'] = test['SibSp'] + test['Parch']



	# """ Model Predicting """
	clf = XGBClassifier()
	clf.fit(X_train,y_train)
	results = pd.DataFrame(columns=['PassengerId','Survived'])
	results['PassengerId'] = test['PassengerId']

	test.replace({'Sex':mapping_sex},inplace=True)
	test.replace({'Embarked':mapping_embarked},inplace=True)
	X_test = test[['Pclass','Sex','Age','Family_Size','Cabin_Length','Fare','Embarked']]
	X_test = scaler.transform(X_test)
	results['Survived'] = clf.predict(X_test)

	results.to_csv('Titanic_Results.csv',sep=',',index=False)

	plt.show(block=False)
	raw_input('Press [enter] to close.')

if __name__ == '__main__':
	main()