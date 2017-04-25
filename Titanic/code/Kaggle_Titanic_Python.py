import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC, SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier

from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit, StratifiedKFold, KFold
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

from tpot import TPOTClassifier

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
	elif title in ['the Countess','Mme','Lady','Dona']:
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

def ensemble_stacking(train, test):
	""" Ensemble help from https://www.kaggle.com/arthurtok/titanic/introduction-to-ensembling-stacking-in-python """
	# Some useful parameters which will come in handy later on
	ntrain = train.shape[0]
	ntest = test.shape[0]
	# print ntrain
	SEED = 23 # for reproducibility
	NFOLDS = 5 # set folds for out-of-fold prediction
	kf = KFold(n_splits= NFOLDS, random_state=SEED)
	# kf = pd.DataFrame(kf.split(ntrain))
	# print kf.get_n_splits(ntrain)
	# print(kf)
	# for train_index, test_index in kf.split(train):
	#	 print("TRAIN:", train_index, "TEST:", test_index)
 #		# X_train, X_test = train[train_index], train[test_index]
 #		# y_train, y_test = y[train_index], y[test_index]
	# input()

	# Class to extend the Sklearn classifier
	class SklearnHelper(object):
		def __init__(self, clf, seed=0, params=None):
			try:
				params['random_state'] = seed
				self.clf = clf(**params)
			except Exception as e:
				self.clf = clf(**params)

		def train(self, x_train, y_train):
			self.clf.fit(x_train, y_train)

		def predict(self, x):
			return self.clf.predict(x)
		
		def fit(self,x,y):
			return self.clf.fit(x,y)
		
		def feature_importances(self,x,y):
			return self.clf.fit(x,y).feature_importances_

	def get_oof(clf, x_train, y_train, x_test):
		oof_train = np.zeros((ntrain,))
		oof_test = np.zeros((ntest,))

		oof_test_skf = np.empty((NFOLDS, ntest))
 
		for i, (train_index, test_index) in enumerate(kf.split(train)):

			x_tr = x_train[train_index]
			y_tr = y_train[train_index]
			x_te = x_train[test_index]

			clf.train(x_tr, y_tr)

			oof_train[test_index] = clf.predict(x_te)
			oof_test_skf[i, :] = clf.predict(x_test)

		oof_test[:] = oof_test_skf.mean(axis=0)
		return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

	# Put in our parameters for said classifiers

	# lsvc_params = {}

	# knn_params = {}

	# dt_params = {
	# 	'max_depth': 5
	# }

	# rf_params = {
	# 	'max_depth': 5,
	# 	'n_estimators': 10,
	# 	'max_features': 1
	# }

	# mlpc_params = {
	# 	'alpha': 1
	# }

	# gnb_params = {}

	# svc_params = {
	# 	'gamma': 2,
	# 	'C': 1
	# }

	# ada_params = {}

	# Random Forest parameters
	rf_params = {
		'n_jobs': -1,
		'n_estimators': 500,
		 'warm_start': True, 
		 #'max_features': 0.2,
		'max_depth': 6,
		'min_samples_leaf': 2,
		'max_features' : 'sqrt',
		'verbose': 0
	}

	# Extra Trees Parameters
	et_params = {
		'n_jobs': -1,
		'n_estimators':500,
		#'max_features': 0.5,
		'max_depth': 8,
		'min_samples_leaf': 2,
		'verbose': 0
	}

	# AdaBoost parameters
	ada_params = {
		'n_estimators': 500,
		'learning_rate' : 0.75
	}

	# Gradient Boosting parameters
	gb_params = {
		'n_estimators': 500,
		 #'max_features': 0.2,
		'max_depth': 5,
		'min_samples_leaf': 2,
		'verbose': 0
	}

	# Support Vector Classifier parameters 
	svc_params = {
		'kernel' : 'linear',
		'C' : 0.025
		}

	# Create 5 objects that represent our 4 models
	# lsvc = SklearnHelper(clf=LinearSVC, seed=SEED, params=lsvc_params)
	# knn = SklearnHelper(clf=KNeighborsClassifier, seed=SEED, params=knn_params)
	# dt = SklearnHelper(clf=DecisionTreeClassifier, seed=SEED, params=dt_params)
	# rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
	# mlpc = SklearnHelper(clf=MLPClassifier, seed=SEED, params=mlpc_params)
	# gnb = SklearnHelper(clf=GaussianNB, seed=SEED, params=gnb_params)
	# ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
	# svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
	rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
	et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
	ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
	gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
	svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

	# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
	y_train = train['Survived'].ravel()
	train = train.drop(['Survived'], axis=1)
	x_train = train.values # Creates an array of the train data
	x_test = test.values # Creats an array of the test data

	# Create our OOF train and test predictions. These base results will be used as new features
	# lsvc_oof_train, lsvc_oof_test = get_oof(lsvc, x_train, y_train, x_test)
	# knn_oof_train, knn_oof_test = get_oof(knn,x_train, y_train, x_test)
	# dt_oof_train, dt_oof_test = get_oof(dt, x_train, y_train, x_test)
	# rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test)
	# mlpc_oof_train, mlpc_oof_test = get_oof(mlpc,x_train, y_train, x_test)
	# gnb_oof_train, gnb_oof_test = get_oof(gnb,x_train, y_train, x_test)
	# ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test)
	# svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test)
	et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
	rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
	ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
	gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
	svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier

	print "Training is complete"

	# lsvc_features = lsvc.feature_importances(x_train,y_train)
	# knn_features = knn.feature_importances(x_train, y_train)
	# dt_features = dt.feature_importances(x_train, y_train)
	# rf_features = rf.feature_importances(x_train,y_train)
	# mlpc_features = mlpc.feature_importances(x_train,y_train)
	# gnb_features = gnb.feature_importances(x_train, y_train)
	# ada_features = ada.feature_importances(x_train, y_train)
	# svc_features = svc.feature_importances(x_train,y_train)
	rf_features = rf.feature_importances(x_train,y_train)
	et_features = et.feature_importances(x_train, y_train)
	ada_features = ada.feature_importances(x_train, y_train)
	gb_features = gb.feature_importances(x_train,y_train)

	cols = train.columns.values
	# Create a dataframe with features
	feature_dataframe = pd.DataFrame( {'features': cols,
		# 'Linear SVC feature importances': lsvc_features,
		# 'K Nearest Neighbors feature importances': knn_features,
		# 'Decision Trees feature importances': dt_features,
		# 'Random Forest feature importances': rf_features,
		# 'MLPC feature importances': mlpc_features,
		# 'Gaussian NB feature importances': gnb_features,
		# 'AdaBoost feature importances': ada_features,
		# 'SVC feature importances': svc_features,
		'Random Forest feature importances': rf_features,
		'Extra Trees feature importances': et_features,
		'AdaBoost feature importances': ada_features,
		'Gradient Boost feature importances': gb_features
	})

	# Create the new column containing the average of values

	feature_dataframe['mean'] = feature_dataframe.mean(axis= 1) # axis = 1 computes the mean row-wise
	print feature_dataframe.head(12)

	base_predictions_train = pd.DataFrame( {
		# 'LinearSVC': lsvc_oof_train.ravel(),
		# 'KNearest': knn_oof_train.ravel(),
		# 'DecisionTrees': dt_oof_train.ravel(),
		# 'RandomForest': rf_oof_train.ravel(),
		# 'MLPClass': mlpc_oof_train.ravel(),
		# 'GaussianNB': gnb_oof_train.ravel(),
		# 'AdaBoost': ada_oof_train.ravel(),
		# 'SVC': svc_oof_train.ravel(),
		'RandomForest': rf_oof_train.ravel(),
		'ExtraTrees': et_oof_train.ravel(),
		'AdaBoost': ada_oof_train.ravel(),
		'GradientBoost': gb_oof_train.ravel()
		})
	base_predictions_train.head()

	colormap = plt.cm.viridis
	plt.figure(figsize=(12,12))
	plt.title('Pearson Correlation of First Order Models', y=1.05, size=15)
	sns.heatmap(base_predictions_train.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
	# pt.plot_correlation_map(train)
	plt.show(block=False)

	# x_train = np.concatenate(( lsvc_oof_train, dt_oof_train, rf_oof_train, mlpc_oof_train, ada_oof_train, svc_oof_train), axis=1)#, gnb_oof_train, knn_oof_train
	# x_test = np.concatenate(( lsvc_oof_test, dt_oof_test, rf_oof_test, mlpc_oof_test, ada_oof_test, svc_oof_test), axis=1)#,  gnb_oof_test, knn_oof_test
	x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
	x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

	gbm = XGBClassifier(
		#learning_rate = 0.02,
	 n_estimators= 2000,
	 max_depth= 4,
	 min_child_weight= 2,
	 #gamma=1,
	 gamma=0.9,
	 subsample=0.8,
	 colsample_bytree=0.8,
	 objective= 'binary:logistic',
	 nthread= -1,
	 scale_pos_weight=1).fit(x_train, y_train)
	predictions = gbm.predict(x_test)
	# results['Survived'] = predictions
	return predictions


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
	
	""" Use train average information to replace NaNs in test set """
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
	le_fam = LabelEncoder()


	train['Deck'] = train['Cabin'].str[0]
	train['Deck'].fillna('Z',inplace=True)

	age_labels = ['Band_1','Band_2','Band_3','Band_4','Band_5','Band_6','Band_7','Band_8','Band_9','Band_10']
	train['AgeBand'] = pd.cut(train['Age'],bins=10,labels=age_labels)
	train['AgeBand'] = le_age.fit_transform(train['AgeBand'])

	fare_labels = ['Band_1','Band_2','Band_3','Band_4','Band_5','Band_6','Band_7','Band_8','Band_9','Band_10']
	train['FareBand'] = pd.cut(train['Fare'],bins=10,labels=fare_labels)
	train['FareBand'] = le_fare.fit_transform(train['FareBand'])

	fam_size_labels = ['Band_1','Band_2','Band_3']
	train['FamilySizeBand'] = pd.cut(train['Family_Size'],bins=3,labels=fam_size_labels)
	train['FamilySizeBand'] = le_fam.fit_transform(train['FamilySizeBand'])

	train['Title'] = le_title.fit_transform(train['Title'])

	train['Embarked'] = le_embarked.fit_transform(train['Embarked'])

	train['Deck'] = le_deck.fit_transform(train['Deck'])

	train['Sex'] = le_sex.fit_transform(train['Sex'])

	train.drop(['PassengerId','Name','Age','SibSp','Parch','Ticket','FareBand','Cabin','Family_Size'],inplace=True,axis=1)
	train.drop(['AgeBand','Deck','Title'],inplace=True,axis=1)
	# survived = train['Survived']
	# train.drop(['Survived'],inplace=True,axis=1)
	# scaler = preprocessing.StandardScaler().fit(train)
	# train = pd.DataFrame(scaler.transform(train))
	# train['Survived'] = survived
	# del survived
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


	# """ Cleaning Test Set """
	test['Deck'] = test['Cabin'].str[0]
	test['Deck'].fillna('Z',inplace=True)
	test['Family_Size'] = test['SibSp'] + test['Parch']

	test['AgeBand'] = pd.cut(test['Age'],bins=10,labels=age_labels)
	test['AgeBand'] = le_age.transform(test['AgeBand'])

	test['FareBand'] = pd.cut(test['Fare'],bins=10,labels=fare_labels)
	test['FareBand'] = le_fare.transform(test['FareBand'])

	test['FamilySizeBand'] = pd.cut(test['Family_Size'],bins=3,labels=fam_size_labels)
	test['FamilySizeBand'] = le_fam.transform(test['FamilySizeBand'])

	test['Title'] = test['Name'].map(lambda x:get_title(x))
	test['Title'] = test.apply(replace_titles,axis = 1)
	test['Title'] = le_title.transform(test['Title'])

	test['Embarked'] = le_embarked.transform(test['Embarked'])

	test['Deck'] = le_deck.transform(test['Deck'])

	test['Sex'] = le_sex.transform(test['Sex'])

	results = pd.DataFrame(columns=['PassengerId','Survived'])
	results['PassengerId'] = test['PassengerId']
	test.drop(['PassengerId','Name','Age','SibSp','Parch','Ticket','FareBand','Cabin','Family_Size'],inplace=True,axis=1)
	test.drop(['AgeBand','Deck','Title'],inplace=True,axis=1)
	# test = pd.DataFrame(scaler.transform(test))


	""" Model Training """
	train.rename(columns = {'Survived':'class'},inplace = True)
	X_train = train.drop(['class'],axis = 1)
	y_train = train['class']
	print y_train.head()
	input()

	# Cross Validation
	train_data, test_data, train_target, test_target = train_test_split(X_train,y_train,test_size=0.25,random_state=0)


	pipeline_optimizer = TPOTClassifier(generations = 10, population_size = 25, random_state = 42, cv = 5, verbosity = 2, n_jobs = 3, scoring = 'f1')
	pipeline_optimizer.fit(train_data,train_target)
	print pipeline_optimizer.score(test_data,test_target)
	pipeline_optimizer.export('Titanic_TPOT_Classifier.py')
	results['Survived'] = pipeline_optimizer.predict(test)

	# Trying a bunch of different classifiers
	# classifiers = [LinearSVC(), KNeighborsClassifier(), DecisionTreeClassifier(max_depth=5), RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), MLPClassifier(alpha=1), GaussianNB(), SVC(gamma=2, C=1), AdaBoostClassifier(), XGBClassifier()]

	# print 'Classifier score:', clf.score(test_data,test_target.values.ravel())





	# # """ Model Analysis """
	# # # cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
	# # # title = "Learning Curves (XGBoost)"
	# # # plot_learning_curve(clf, title, X_train, y_train.values, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
	# # # y_pred = clf.predict(X_train)
	# # # print 'done in %0.3fs' % (time() - t0)
	# # # print confusion_matrix(y_train,y_pred)


	# # # print '*******************************************'
	# # # print classification_report(y_train,y_pred)
	# # # print 0.5 * (precision_recall_fscore_support(y_train,y_pred)[2][0] + precision_recall_fscore_support(y_train,y_pred)[2][1])
	# # # print '*******************************************'

	# # # print max(evaluation(classifiers,train_data,train_target,test_data,test_target)[:][1])
	# # _ , f_scores = evaluation(classifiers,train_data,train_target,test_data,test_target)
	# # print f_scores.index(max(f_scores))
	# # model = classifiers[-1]#f_scores.index(max(f_scores))]
	# # print model
	# # rfecv = RFECV( estimator = model , step = 1 , cv = StratifiedKFold( 2 ) , scoring = 'accuracy' )
	# # rfecv.fit( train_data , train_target )

	# # print rfecv.score( train_data , train_target ) , rfecv.score( test_data , test_target )
	# # print "Optimal number of features : %d" % rfecv.n_features_ 
	# # input()

	# # # Plot number of features VS. cross-validation scores
	# # plt.figure()
	# # plt.xlabel( "Number of features selected" )
	# # plt.ylabel( "Cross validation score (nb of correct classifications)" )
	# # plt.plot( range( 1 , len( rfecv.grid_scores_ ) + 1 ) , rfecv.grid_scores_ )
	# # plt.show(block=False)

	# # print X_train.head()
	# # X_train = pd.DataFrame(rfecv.transform(X_train))
	# # print rfecv.ranking_
	# # print X_train.head()
	# # input()








	# # # """ Model Predicting """
	# # print 'Fitting model to full training set...'
	# # model.fit(X_train,y_train)

	# # X_test = test[['Pclass','Sex','Embarked','Title','Deck','AgeBand','FareBand','FamilySizeBand']]
	# # X_test = pd.DataFrame(rfecv.transform(pd.DataFrame(scaler.transform(X_test))))
	# # results['Survived'] = model.predict(X_test)










	""" Ensemble Stacking """
	# results['Survived'] = ensemble_stacking(train,test)




	results.to_csv('Titanic_Results.csv',sep=',',index=False)

	plt.show(block=False)
	raw_input('Press [enter] to close.')

if __name__ == '__main__':
	main()