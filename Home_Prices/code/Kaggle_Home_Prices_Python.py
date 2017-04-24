import numpy as np
import pandas as pd

import scipy.stats as stats



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

def print_hist(df,bins = 25,xlabel = 'x data',ylabel = 'y data',title = 'Title',normed = False):
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

def mean_analysis(df,column_vals):
	for val in column_vals:
		if val == 'Age':
			print 'Mean survival rate by age group'
			temp_df = df.groupby(pd.cut(df['Age'],np.arange(0,df['Age'].max()+10,10)))['LogPrice'].mean()
			print temp_df
			plt.figure()
			ax = temp_df.plot(kind = 'bar')
			ax.xlabel = val
			plt.show(block = False)
		elif val == 'Fare':
			print 'Mean survival rate by fare'
			temp_df = df.groupby(pd.cut(df['Fare'],np.arange(0,df['Fare'].max()+50,50)))['LogPrice'].mean()
			print temp_df
			plt.figure()
			ax = temp_df.plot(kind = 'bar')
			ax.xlabel = val
			plt.show(block = False)
		else:
			print 'Mean survival rate by ' + str(val)
			temp_df = df.groupby(val).LogPrice.mean()
			print temp_df
			plt.figure()
			sns.barplot(x = val, y = 'LogPrice', data = df, ci = None)
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


def data_analysis():

	# Import Data
	train = pd.read_csv("../data/train.csv")#, dtype={"Age": np.float64}, )
	test = pd.read_csv("../data/test.csv")#, dtype={"Age": np.float64}, )

	# Print the list of categorical and numerical columns
	categorical = []
	numerical = []
	for i in range(len(train.columns)):
		# j = 0
		# type_val = type(train.ix[j,i])
		# while not math.isnan(type_val):
		# 	j += 1
		# 	type_val = type(train.ix[j,i])
		if type(train.ix[0,i]) == np.str:
			categorical.append(train.columns.values[i])
		else:
			numerical.append(train.columns.values[i])
	print 'Categorical:', categorical
	print
	print 'Numerical:',numerical

	# Plot the distribution and qq plot of the sale price
	fig1 = plt.figure()
	sns.distplot(train['SalePrice'],bins=25)
	sns.plt.suptitle('Sale Price Original Distribution')
	fig2 = plt.figure()
	ax = fig2.add_subplot(111)
	stats.probplot(train['SalePrice'],plot=plt)
	ax.set_title("Probplot Sale Prices")
	plt.show(block=False)
	input()
	plt.close(fig1)
	plt.close(fig2)
	
	# Take the Log of the sale price
	train['LogPrice'] = np.log(train['SalePrice'])
	# print train['SalePrice'].head(10)
	# print train['LogPrice'].head(10)
	numerical.append('LogPrice')
	categorical.append('LogPrice')

	# Plot the distribution and qq plot of the logged price
	plt.figure()
	sns.distplot(train['LogPrice'],bins=25)
	sns.plt.suptitle('Sale Price Log Distribution')
	fig = plt.figure()
	ax = fig.add_subplot(111)
	stats.probplot(train['LogPrice'],plot=plt)
	ax.set_title("Probplot logged Sale Prices")
	plt.show(block=False)

	# Create separate numerical and categorical databases
	train_numerical = train[numerical].copy()
	train_numerical.drop(['Id','SalePrice'],axis=1,inplace=True)
	train_categorical = train[categorical].copy()
	# train_categorical.drop(['SalePrice'],axis=1,inplace=True)

	# Check correlation of numerical features with Log price
	# colormap = plt.cm.viridis
	# plt.figure(figsize=(20,20))
	# plt.title('Pearson Correlation of Numerical Features', y=1.05, size=15)
	# sns.heatmap(train_numerical.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
	# plt.show(block=False)
	
	# Check correlation of numerical features with Log price
	# colormap = plt.cm.viridis
	# plt.figure(figsize=(20,20))
	# plt.title('Pearson Correlation of Categorical Features', y=1.05, size=15)
	# sns.heatmap(train_numerical.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
	# plt.show(block=False)

	# Check correlation of lot characteristics
	le = LabelEncoder()

	lot = train[['LotFrontage','LotArea']].copy()
	lot['LotShape'] = le.fit_transform(train['LotShape'])
	# We wouldn't expect both Contour and Slope to play differing roles, but include them in these analyses at first
	lot['LandContour'] = le.fit_transform(train['LandContour'])
	lot['LandSlope'] = le.fit_transform(train['LandSlope'])
	lot['LotConfig'] = le.fit_transform(train['LotConfig'])
	lot['LogPrice'] = train['LogPrice']
	colormap = plt.cm.viridis
	plt.figure(figsize=(20,20))
	plt.title('Pearson Correlation of Lot Features', y=1.05, size=15)
	sns.heatmap(lot.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
	plt.show(block=False)






	# Next examine year built and remodeled.  You'll likely only want to use remodeled because that will be the more telling value but this will be checked







	# print train_numerical.corr()['LogPrice']


def main():
	data_analysis()
	plt.show(block=False)
	raw_input('Press [enter] to close.')

if __name__ == '__main__':
	main()