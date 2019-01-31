#!/usr/bin/python -tt
"""
Created on Fri Aug 25 14:29:17 2017

@author: adeacon
"""

import pandas as pd
import math
from scipy.spatial import distance
import random
from numpy.random import permutation
import config

pd.set_option('display.max_columns', 500)
pd.set_option('display.expand_frame_repr', False)

master_file = config.MASTER_FILES["ftb_players"]
distance_columns = ["Age", "ChancesInvolved", "DefensiveActions", "FoulsCommited", "FoulsSuffered", "Height", "Minutes", "NPG+A", "Points", "Weight", "SuccessfulPasses"]

def df_info(dframe):
    dframe.info()
    #print dframe.describe(include="all")
    print(list(dframe.columns.values))
    #print dframe #.head(5)
    #print dframe.tail(5)
    #print dframe["PositionGroup"].value_counts()
    #print dframe["Season"].value_counts()
    #print dframe["Competition"].value_counts()
    #print dframe["Name"].value_counts()[:10]

def get_players():
    #fetch from master csv
    df = pd.read_csv(master_file, sep='|', encoding="ISO-8859-1")
    #filter unwanted records
    df = df[(df["Season"] >= "s1314") & (df["Competition"].isin(["chm","cpo","prm"]))]
    df.dropna(subset=["Name"], inplace=True)
    #select columns
    group_key = "Name"
    max_cols = ["Age", "Height", "Weight"]
    #p90_cols = ["AerialsWon", "ChancesInvolved", "DefensiveActions", "Dispossesed", "Dribbles", "FoulsCommited", "FoulsSuffered", "NPG+A", "SuccessfulPasses"]
    p90_cols = ['AerialsWon'	, 'Assists'	, 'BadControl'	, 'Blocks'	, 'CalledOffside'	, 'Clearances'	, 'Crosses'	, 'Dispossesed'	, 'Dribbles'	, 'DribblesAgainst'	, 'FirstYellowCards'	, 'FoulsCommited'	, 'FoulsSuffered'	, 'GoalsConceded'	, 'Interceptions'	, 'KeyPasses'	, 'LongBalls'	, 'NonPenaltyGoals'	, 'OffsidesWon'	, 'OwnGoals'	, 'Passes'	, 'PenaltyGoals'	, 'RedCards'	, 'Saves'	, 'Shots'	, 'ShotsFaced'	, 'ShotsOnTarget'	, 'Tackles'	, 'ThroughBalls'	, 'YellowCards']
    pGm_cols = ["Appearances", "Minutes", "Points"]
    sum_cols = p90_cols + pGm_cols
    selected_columns = [group_key]+max_cols+sum_cols
    df = df[selected_columns]
    #aggregate to player level
    df_max = df[[group_key]+max_cols].groupby(group_key).max()
    df_sum = df[[group_key]+sum_cols].groupby(group_key).sum()
    df = pd.concat([df_max, df_sum], axis=1)
    df = df[(df["Minutes"] >= 900)]
    #convert action totals to per90
    for col in p90_cols:
        df[col+"P90"] = 90 * df[col] / df["Minutes"]
    for col in pGm_cols:
        df[col+"PGm"] = df[col] / df["Appearances"]
    for col in sum_cols:
        del df[col]
    del df["AppearancesPGm"]

    #df_info(df)
    return df

def find_similar():
    players = get_players()
    #print players
    print("\nNumber of players included: "+str(len(players)))

    # Normalize all of the numeric columns
    players_normalized = (players - players.mean()) / players.std()
    players_normalized.fillna(0, inplace=True)
    #players_normalized.info()
    #print players_normalized.describe(include="all")

    #print players_normalized.index.values
    for name in players_normalized.index.values: #["Adam Clayton", "Ben Gibson", "Daniel Ayala", "Tomas Mejias"]:
        #print "\n###############################"
        print("\n"+name, end=' ')

        #selected_player = players.loc[name]
        #print selected_player.name
        #print selected_player.to_frame().T #.name

        # Normalize all of the numeric columns
        selected_normalized = players_normalized.loc[name]
        #print selected_normalized

        # Find the distance between select player and everyone else.
        euclidean_distances = players_normalized.apply(lambda row: distance.euclidean(row, selected_normalized), axis=1)

        # Create a new dataframe with distances.
        distance_frame = pd.DataFrame(data={"dist": euclidean_distances, "idx": euclidean_distances.index})
        distance_frame.sort_values("dist", inplace=True)

        most_similar_players = distance_frame.iloc[1:4]["idx"]
        #most_similar_players = players.loc[nearest_neighbours] #["Name"]
        #print most_similar_players
        print("... is similar to... ", end=' ')
        print(list(most_similar_players.index.values))

def make_prediction():
    players = get_players()
    pred_col = 'AssistsP90'
    x_columns = list(players.columns.values)
    x_columns.remove(pred_col)
    y_column = [pred_col]

#    # The columns that we will be making predictions with.
#    x_columns = ['Age', 'Height', 'Weight', 'AerialsWonP90', 'AssistsP90', 'BadControlP90', 'BlocksP90', 'CalledOffsideP90', 'ClearancesP90', 'CrossesP90', 'DispossesedP90', 'DribblesP90', 'DribblesAgainstP90', 'FirstYellowCardsP90', 'FoulsCommitedP90', 'FoulsSufferedP90', 'GoalsConcededP90', 'InterceptionsP90', 'KeyPassesP90', 'LongBallsP90', 'NonPenaltyGoalsP90', 'OffsidesWonP90', 'OwnGoalsP90', 'PassesP90', 'PenaltyGoalsP90', 'RedCardsP90', 'SavesP90', 'ShotsP90', 'ShotsFacedP90', 'ShotsOnTargetP90', 'TacklesP90', 'ThroughBallsP90', 'YellowCardsP90', 'MinutesPGm']
#    print x_columns
#    # The column that we want to predict.
#    y_column = [pred_col]
#    print y_column

    ###Generating training and testing sets

    # Randomly shuffle the index of nba.
    random_indices = permutation(players.index)
    # Set a cutoff for how many items we want in the test set (in this case 1/3 of the items)
    test_cutoff = math.floor(len(players)/3)
    # Generate the test set by taking the first 1/3 of the randomly shuffled indices.
    test = players.loc[random_indices[1:test_cutoff]]
    test.fillna(0, inplace=True)
    #test.info()
    #print test.describe(include="all")
    # Generate the train set with the rest of the data.
    train = players.loc[random_indices[test_cutoff:]]
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
    print("\nMean Squared Error:")
    print(mse)

    actual["Predicted"+pred_col] = predictions
    actual["Diff"] = actual[pred_col] - actual["Predicted"+pred_col]
    print("\nActual and Predicted "+pred_col+":")
    print(actual.sort_values(["Diff"], ascending=False))

def test_opinions():
    players = get_players()
    players = players.reset_index()
    players = players[players["Name"].isin(["Alvaro Negredo","Patrick Bamford","Jordan Rhodes","Garcia Kike","Cristhian Stuani","David Nugent","Danny Graham","Jelle Vossen","Kei Kamara"])]
    #df_info(players)
    players["ShotAccuracy"] = players["ShotsOnTargetP90"] / players["ShotsP90"]
    players["ShotEfficiency"] = (players["NonPenaltyGoalsP90"]+players["PenaltyGoalsP90"].fillna(0)) / players["ShotsP90"]
    players["ShotPercentage"] = (players["NonPenaltyGoalsP90"]+players["PenaltyGoalsP90"].fillna(0)) / players["ShotsOnTargetP90"]
    players = players[["Name", "NonPenaltyGoalsP90", "PenaltyGoalsP90", "ShotsP90", "ShotsOnTargetP90", "ShotAccuracy", "ShotEfficiency", "ShotPercentage"]]
    #df_info(players)
    print(players.describe())
    print(players)

def main():
    # get_players()
    # find_similar()
    # make_prediction()
    test_opinions()

if __name__ == '__main__':
    main()
