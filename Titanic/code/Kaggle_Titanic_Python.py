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
from sklearn.preprocessing import Imputer

from xgboost import XGBClassifier

import string

import matplotlib.pyplot as plt
from ggplot import *
import seaborn as sns

from plot_learning_curve import plot_learning_curve



def print_hist(df,survival = None,bins = 25,xlabel = 'x data',ylabel = 'y data',title = 'Title',normed = False):
	temp_df = df.copy()
	temp_df = temp_df[-np.isnan(temp_df)]
	temp_df = sorted(temp_df)
	sns.distplot(temp_df,hist=True,bins=bins)

def print_seaborn(df,x,hue):
	sns.countplot(x = x,hue = hue, data = df)
	plt.show()
	raw_input('Press [enter] to continue')
	plt.close()

def seaborn_hist(df,subset):
	df.dropna(subset=[subset],how='all')
	g = sns.FacetGrid(df, col = 'Survived',col_wrap = 2, size = 5)
	g.map(sns.distplot,subset)

def main():

	# Import Data
	train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
	test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

	# Check to see if there are any null Fare, Class, Embarked values
	''' This will need to be done for other features, as well.  You'll need to replace NaNs with valid values '''
	print 'Fare NaNs:', train[pd.isnull(train['Fare'])]
	print train[pd.isnull(train['Pclass'])]
	age_nans = train[pd.isnull(train['Age'])]
	print age_nans.groupby(['Pclass','Sex','Embarked']).count()

	""" # Print survival counts grouped by different parameters
	print_seaborn(train,'Survived','Sex')
	print_seaborn(train,'Survived','Embarked')
	print_seaborn(train,'Survived','Pclass')
	"""
	# Map sex, port of embarkation to numeric values
	mapping_sex = {'female':0,'male':1}
	mapping_embarked = {'Q':0,'S':1,'C':2}
	
	train.replace({'Sex':mapping_sex},inplace=True)
	train.replace({'Embarked':mapping_embarked},inplace=True)

	# Determine the number of cabins reserved per person
	train['Cabin_Length'] = train['Cabin'].str.split(' ').str.len()
	train['Cabin_Length'].fillna(0,inplace=True)

	# Print number of people who reserved each quantity of cabins
	print train.groupby('Cabin_Length').Cabin_Length.count()

	# Print the mean survival rate by sex
	print train.groupby('Sex').Survived.mean()

	# Print the mean survival rate by age group
	print train.groupby(pd.cut(train['Age'],np.arange(0,train['Age'].max()+10,10)))['Survived'].mean()

	# Print the mean survival rate by class
	print train.groupby('Pclass').Survived.mean()


	"""

	You'll need to do more statistical analysis for the other features as well as perform an analysis on the titles of passengers


	"""



	# seaborn_hist(train,'Sex')
	# print_seaborn(train,'Survived', 'Sex')
	# print_hist(train['Age'],xlabel='Age',ylabel='Number of Deaths',title='Age Histogram',normed=True)
	# print_hist(train['Fare'],xlabel='Fare',ylabel='Number of Tickets',title='Fare Grouping',bins=100)




	""" Model Training """
	# Create DF with only numeric values for data, use imputer to replace NaNs with mean
	X_train = train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
	# X_train = train[['Age','Family_Size','Parch','Sex','Age*Class','Fare_Per_Person']]
	imp = Imputer(missing_values = 'NaN',strategy='mean',axis=0)
	X_train = imp.fit_transform(X_train)
	y_train = train['Survived']

	# Scale data
	scaler = preprocessing.StandardScaler().fit(X_train)
	X_train = scaler.transform(X_train)

	# Cross Validation
	train_data, test_data, train_target, test_target = train_test_split(X_train,y_train,test_size=0.5,random_state=0)

	# Trying a bunch of different classifiers
	# clf = LinearSVC().fit(train_data,train_target.values.ravel())
	# clf = tree.DecisionTreeClassifier().fit(train_data,train_target.values.ravel())
	# clf = KNeighborsClassifier().fit(train_data,train_target.values.ravel())
	# clf = DecisionTreeClassifier(max_depth=5).fit(train_data,train_target.values.ravel())
	# clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1).fit(train_data,train_target.values.ravel())
	# clf = MLPClassifier(alpha=1).fit(train_data,train_target.values.ravel())
	# clf = GaussianNB().fit(train_data,train_target.values.ravel())
	# clf = SVC(gamma=2, C=1).fit(train_data,train_target.values.ravel())
	# clf = AdaBoostClassifier().fit(train_data,train_target.values.ravel())
	clf = XGBClassifier().fit(train_data,train_target.values.ravel())


	print clf.score(test_data,test_target.values.ravel())


	""" Model Analysis """
	cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
	title = "Learning Curves (XGBoost)"
	plot_learning_curve(clf, title, X_train, y_train.values, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
	# train_sizes, train_scores, valid_scores = learning_curve(XGBClassifier, X_train, y_train, train_sizes=[50,80,110],cv=5)
	# print train_sizes
	# print train_scores
	# print valid_scores


	""" Model Predicting """
	# results = pd.DataFrame(columns=['PassengerId','Survived'])
	# results['PassengerId'] = test['PassengerId']

	# test.replace({'Sex':mapping_sex},inplace=True)
	# test.replace({'Embarked':mapping_embarked},inplace=True)
	# X_test = test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
	# X_test = imp.fit_transform(X_test)
	# X_test = scaler.transform(X_test)
	# results['Survived'] = clf.predict(X_test)

	# results.to_csv('Titanic_Results.csv',sep=',',index=False)

	plt.show()

if __name__ == '__main__':
	main()