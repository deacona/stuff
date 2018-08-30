#!/usr/bin/python -tt
"""
Created on Mon Feb 06 16:49:37 2017

@author: adeacon
"""

import pandas as pd
#import re
#import matplotlib.pyplot as plt
from scipy.spatial import distance
import math
from numpy.random import permutation
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import ftb_clubs_wrangle as ftb

pd.set_option('display.max_columns', 500)
pd.set_option('display.expand_frame_repr', False)

def results_analysis(dframe):

    dframe = ftb.get_master("results")

    #dframe.info()
    #print dframe.describe(include="all")
    #print dframe.groupby(['Div']).count()
    #print dframe.sort_values(by='FTHG', ascending=False)[:20]
    #print len(dframe['Country'].unique())
    #print dframe.groupby('Season')['HS','AS','HST','AST','FTHG','FTAG'].agg(['mean'])
    #print dframe[dframe.Country.isin(['Germany'])].groupby(['Div','Season'])['HS','AS','HST','AST','FTHG','FTAG'].agg(['mean'])

    #gbase = dframe[dframe.Div.isin(['E0'])].groupby(['Season'])['HS','AS','HST','AST','FTHG','FTAG'].agg(['mean'])
    #gbase = dframe[dframe.Div.isin(['D1','E0','F1','I1','SP1'])].groupby(['Season'])['HS','AS','HST','AST','FTHG','FTAG'].agg(['sum'])
    #gbase.plot(kind='line')

    #values = dframe['FTHG'][:20]
    #values.plot(kind='kde', style='k--')

    #dframe[dframe.Div.isin(['E0'])].plot(kind='scatter',x='FTHG',y='HC')

    #print dframe[['FTHG','FTAG']]
    buckets = ['Div','Season']
    stats = 'HS','AS','HST','AST','FTHG','FTAG'
    filteron = 'HomeTeam'
    values = ['Barcelona']
    aggfunc = 'mean'
    print "SELECT "+aggfunc+" OF "+str(stats)+" WHERE "+filteron+" IS "+str(values)+" GROUPED BY "+str(buckets)
    #selected = dframe[['FTHG','FTAG','HST','AST']]
    selected = dframe[dframe[filteron].isin(values)].groupby(buckets)[stats].agg([aggfunc])
    pd.scatter_matrix(selected, diagonal='kde')
    #plt.show()

def fulldata_analysis():

    fulldata = ftb.get_master("fulldata")

    #print pd.show_versions()
    fulldata.info()
    #print fulldata.describe(include="all")
    #print fulldata["Div"].value_counts()
    #print fulldata["Tier"].value_counts()
    #print fulldata["Country"].value_counts()
    #print fulldata["Season"].value_counts()
    #print fulldata["Goals"].value_counts()
    #print fulldata["Stadium"].value_counts()
    #print fulldata["City"].value_counts()
    #print fulldata["Manager"].value_counts()
    #print fulldata[(fulldata['Team']=="Middlesbrough")&(fulldata['Season']=="2017-2018")].describe(include="all")
    #print fulldata[(fulldata['EuclideanDistance']==0)] #.head()

    buckets = ['Div','Season','Team']
    stats = 'Shots','ShotsOnTarget','Goals','Corners','Points','Win'
    filteron = 'Div'
    values = ['E0']
    aggfunc = 'mean'
    print "SELECT "+aggfunc+" OF "+str(stats)+" WHERE "+filteron+" IS "+str(values)+" GROUPED BY "+str(buckets)
    selected = fulldata[fulldata[filteron].isin(values)].groupby(buckets)[stats].agg([aggfunc])
    pd.scatter_matrix(selected, diagonal='kde')


def find_similar():

    group_key = "Manager"
    base_filters = {
            "Tier": [1]
            , "Country": ["England", "Spain", "Italy", "Germany"]
            }
    metric_mins = {
            "NumberOfMatches": 100
            }

    items = ftb.get_summary(group_key, base_filters=base_filters, metric_mins=metric_mins)
    items.info()

    #print players
    print "\nNumber of items included: "+str(len(items))

    # Normalize all of the numeric columns
    items_normalized = (items - items.mean()) / items.std()
    items_normalized.fillna(0, inplace=True)
    #items_normalized.info()
    #print items_normalized.describe(include="all")

    #print players_normalized.index.values
    for item in items_normalized.index.values:
        #print "\n###############################"
        print "\n"+item,

        #selected_player = players.loc[name]
        #print selected_player.name
        #print selected_player.to_frame().T #.name

        # Normalize all of the numeric columns
        selected_normalized = items_normalized.loc[item]
        #print selected_normalized

        # Find the distance between select player and everyone else.
        euclidean_distances = items_normalized.apply(lambda row: distance.euclidean(row, selected_normalized), axis=1)

        # Create a new dataframe with distances.
        distance_frame = pd.DataFrame(data={"dist": euclidean_distances, "idx": euclidean_distances.index})
        distance_frame.sort_values("dist", inplace=True)

        most_similar = distance_frame.iloc[1:4]["idx"]
        #most_similar = items.loc[nearest_neighbours]
        #print most_similar
        print "... is similar to... ",
        print list(most_similar.index.values)

def make_prediction():

    group_key = "Team"
    base_filters = {
            "Div":["E0"]
            }
    pred_col = 'Points'

    items = ftb.get_summary(group_key, base_filters=base_filters)

    x_columns = list(items.columns.values)
    x_columns.remove(pred_col)
    y_column = [pred_col]

    ###Generating training and testing sets

    # Randomly shuffle the index of nba.
    random_indices = permutation(items.index)
    # Set a cutoff for how many items we want in the test set (in this case 1/3 of the items)
    test_cutoff = math.floor(len(items)/3)
    # Generate the test set by taking the first 1/3 of the randomly shuffled indices.
    test = items.loc[random_indices[1:test_cutoff]]
    test.fillna(0, inplace=True)
    #test.info()
    #print test.describe(include="all")
    # Generate the train set with the rest of the data.
    train = items.loc[random_indices[test_cutoff:]]
    train.fillna(0, inplace=True)
    #train.info()
    #print train.describe(include="all")

    ###Using sklearn for k nearest neighbors
    #print "Using sklearn for k nearest neighbors..."

    from sklearn.neighbors import KNeighborsRegressor
    # Create the knn model.
    # Look at the five closest neighbors.
    knn = KNeighborsRegressor(n_neighbors=5)
    #print knn
    # Fit the model on the training data.
    knn.fit(train[x_columns], train[y_column])
    #print knn
    # Make point predictions on the test set using the fit model.
    predictions = knn.predict(test[x_columns])
    #print "\nPredicted PointsPGm:"
    #print predictions.shape

    ###Computing error

    # Get the actual values for the test set.
    actual = test[y_column].copy()

    # Compute the mean squared error of our predictions.
    mse = (((predictions - actual) ** 2).sum()) / len(predictions)
    print "\nMean Squared Error:"
    print mse

    actual["Predicted"+pred_col] = predictions
    actual["Diff"] = actual[pred_col] - actual["Predicted"+pred_col]
    print "\nActual and Predicted "+pred_col+":"
    print actual.sort_values(["Diff"], ascending=False)

def do_classification():
    # Load dataset
    #url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    dataset = ftb.get_master("fulldata")

    #filter unwanted records and columns
    dataset = dataset[(dataset['Div']=="E0")]
    #dataset.info()
    dataset = dataset[["ShotsOnTarget", "ShotsOnTargetOpp", "Saves", "SavesOpp", "Result"]]
    dataset.dropna(subset=["ShotsOnTarget", "ShotsOnTargetOpp", "Saves", "SavesOpp", "Result"], inplace=True)
    #print list(dataset.columns.values)

    # shape
    print(dataset.shape)

    # head
    print(dataset.head(20))

    # descriptions
    print(dataset.describe())

    # class distribution
    print(dataset.groupby('Result').size())

    # box and whisker plots
    dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    plt.show()

    # histograms
    dataset.hist()
    plt.show()

    # scatter plot matrix
    scatter_matrix(dataset)
    plt.show()

    # Split-out validation dataset
    array = dataset.values
    X = array[:,0:4]
    Y = array[:,4]
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

    print X_train[:5]
    print X_validation[:5]
    print Y_train[:5]
    print Y_validation[:5]

    # Test options and evaluation metric
    seed = 7
    scoring = 'accuracy'

    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
    	kfold = model_selection.KFold(n_splits=10, random_state=seed)
    	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    	results.append(cv_results)
    	names.append(name)
    	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    	print(msg)

    # Compare Algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

    # Make predictions on validation dataset
    print "\nPrediction using K Nearest Neighbours algorithm..."
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))

    print "\nPrediction using Support Vector Clustering algorithm..."
    svc = SVC()
    svc.fit(X_train, Y_train)
    predictions = svc.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))

def test_soccernomics():
    pass
#        ## Soccernomics (* need financial/economic data)
#            *Avg lge pos Vs Relative wage spend
#                EPL+Champ 98-07 R^2 = 0.8872
#                40 English clubs 78-97 R^2 = 0.92
#            *Change in lge pos Vs Pre-tax profits
#            Change in lge pos Vs Change in attendance
#            Win% Vs Experience, Population, Wealth
#                International teams 1872-2001
#                    Home adv = 2/3 GoalsPG
#                    Int experience (matches) = 1/2GoalsPG
#                    *Population = 1/10 GoalsPG
#                    *GDP = 1/10 GoalsPG

def test_numbers_game():

#        ## Numbers Game (* need betting data)
#            Corners Vs Shots, Goals
#                EPL 2001/02-2010/11
    group_key = "Corners"
    base_filters = {
            "Div": ["E0"]
            , "Season": ["2001-2002", "2002-2003", "2003-2004", "2004-2005", "2005-2006", "2006-2007", "2007-2008", "2008-2009", "2009-2010", "2010-2011"]
            }
    output_metrics = ["NumberOfMatches", "Shots","ShotsOnTarget", "Goals", "Win", "WinDraw", "Points", "GraysonScore"]
    corners = ftb.get_summary(group_key, base_filters=base_filters, output_metrics=output_metrics)
    corners.info()
    print corners
#            Actual goal freq Vs Predicted (Poisson dist)
#                England, Germany, Spain, Italy, France 1993-2011
#            Home goals Vs Away goals (%)
#                EPL 2011/02-2010/11
#                Serie A
#                La Liga
#                Bundesliga
#            *% favourite wins Vs % gap in odds

def test_footballintheclouds():
    ### Various Shot Metrics - Premier League 2016/17
    #http://footballintheclouds.blogspot.co.uk/p/various-shot-metrics-premier-league.html
    #https://jameswgrayson.wordpress.com/2014/08/11/methodology-and-validation-of-the-team-ratings/
    group_key = "Team"
    base_filters = {
            "Div": ["E0"]
            , "Season": ["2016-2017"] #["2009-2010"]
            }
    output_metrics = ["NumberOfMatches", 'Shots', 'ShotsOpp', 'TSR', 'ShotPercent'	, 'SavePercent', 'PDO', '%TSoTt', 'GraysonRating', 'GraysonScore']
    teams = ftb.get_summary(group_key, base_filters=base_filters, output_metrics=output_metrics)
    teams.info()
    print teams

def boro_analysis():
    #group_key = "Season"
    group_key = "Manager"
    base_filters = {
            "Team": ["Middlesbrough"]
            #, "Div": ["E1"]
            #, "Manager": ["Aitor Karanka", "Garry Monk"]
            }
    output_metrics = [
            "NumberOfMatches",
            "Goals", #"GoalsOpp",
            #'Shots', 'ShotsOpp',
            #'ShotsOnTarget', 'ShotsOnTargetOpp',
            'TSR', #'TSROpp',
            #'ShotPercent',
            #'SavePercent',
            'PDO', #'PDOOpp',
            'GraysonScore', 'GraysonScoreOpp',
            ]
    boro = ftb.get_summary(group_key, base_filters=base_filters, output_metrics=output_metrics)
    #boro.info()
    print boro
    pd.scatter_matrix(boro, diagonal='kde')

def best_rivalries():
    pass
#    Best rivalries (frequency, closeness, equality, action)

def main():

#    results_analysis()
#    fulldata_analysis()

#    find_similar()
#    make_prediction()
#    do_classification()

#    test_soccernomics()
#    test_numbers_game()
#    test_footballintheclouds()
    boro_analysis()
#    best_rivalries()

if __name__ == '__main__':
    main()