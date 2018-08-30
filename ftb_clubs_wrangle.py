#!/usr/bin/python -tt
"""
Created on Mon Feb 06 16:49:37 2017

@author: adeacon
"""

import os
import zipfile
import pandas as pd
#import re
import datetime
#import matplotlib.pyplot as plt
import config

pd.set_option('display.max_columns', 500)
pd.set_option('display.expand_frame_repr', False)

results_url = "http://football-data.co.uk/downloadm.php"
results_dir = config.SOURCE_DIR+"/ftd"

stadiums_dict = config.STADIUMS_SCRAPE
managers_dict = config.MANAGERS_SCRAPE

def unzip_files():
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file == "data.zip":
                #print root
                #print dirs
                #print file
                print(os.path.join(root, file))
                with zipfile.ZipFile(os.path.join(root, file),"r") as zip_ref:
                    zip_ref.extractall(root)

def master_path(stub):
    return config.MASTER_DIR+"/"+"ftb_"+stub+".txt"

def get_master(stub):
    print "Fetching "+master_path(stub)
    return pd.read_csv(master_path(stub), encoding = "utf8", sep='|')
    
#    chunksize = 1000
#    chunks = []
#    for chunk in pd.read_csv(master_path(stub), chunksize=chunksize):
#        # Do stuff...
#        chunks.append(chunk)
#    
#    df = pd.concat(chunks, axis=0)
#    return df

def save_master(dframe, stub):
    dframe.to_csv(master_path(stub), encoding='utf-8', sep='|')
    print "... saved to "+master_path(stub)
    
def func_div(x):
    div_dict = {
        'B1': ['Belgium',1]
        , 'D1': ['Germany',1]
        , 'D2': ['Germany',2]
        , 'E0': ['England',1]
        , 'E1': ['England',2]
        , 'E2': ['England',3]
        , 'E3': ['England',4]
        , 'EC': ['England',5]
        , 'F1': ['France',1]
        , 'F2': ['France',2]
        , 'G1': ['Greece',1]
        , 'I1': ['Italy',1]
        , 'I2': ['Italy',2]
        , 'N1': ['Holland',1]
        , 'P1': ['Portugal',1]
        , 'SC0': ['Scotland',1]
        , 'SC1': ['Scotland',2]
        , 'SC2': ['Scotland',3]
        , 'SC3': ['Scotland',4]
        , 'SP1': ['Spain',1]
        , 'SP2': ['Spain',2]
        , 'T1': ['Turkey',1]
        }
    return div_dict[x]

def func_score(x):
    #Result,Points,PointsOpp,Win,WinDraw,Draw,DrawLoss,Loss,WinShare
    if x > 0:
        return 'Win', 3, 0, 1, 1, 0, 0, 0, 1.0
    elif x < 0:
        return 'Loss', 0, 3, 0, 0, 0, 1, 1, 0.5
    else:
        return 'Draw', 1, 1, 0, 1, 1, 1, 0, 0.0

def func_nogoal(x):
    if x == 0:
        return 1
    else:
        return 0


def format_results():
    pieces = []
    core_cols = ['Div','Date'] #,'HomeTeam','AwayTeam','FTHG','FTAG','FTR']
    use_cols = ['Season','Div','Country','Tier','Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','Attendance','Referee','HS','AS','HST','AST','HHW','AHW','HC','AC','HF','AF','HO','AO','HY','AY','HR','AR','HBP','ABP']

    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith(".csv"):
                #print root
                #print os.path.join(root, file)
                #print root[-9:]
                try:
                    df = pd.read_csv(os.path.join(root, file), encoding = "utf8", parse_dates=['Date'])
                    #df['File'] = file
                    df['Season'] = root[-9:]
                    #drop useless rows
                    df = df.dropna(subset=core_cols)
                    pieces.append(df)
                except:
                    print "read_csv FAILED: "+os.path.join(root, file)
                #print df.count()
    
    # Concatenate everything into a single DataFrame
    dframe = pd.concat(pieces, ignore_index=True)
    
    dframe['Country'], dframe['Tier'] = zip(*dframe['Div'].map(func_div))
    
    #print dframe.describe(include="all")
    
    save_master(dframe[use_cols], "results")
    #return dframe[use_cols]

def format_stadiums():
    dgl = pd.read_csv(stadiums_dict["dgl"][1], encoding = "utf8", sep=',')
    dgl.rename(columns={'Name':'Stadium'}, inplace=True)
    dgl.set_index('Team', inplace=True)
    dgl.info()
    #print dgl.describe(include="all")
    
    ops = pd.read_csv(stadiums_dict["ops"][1], encoding = "utf8", sep=',')
    ops.rename(columns={'Team':'TeamFull', 'FDCOUK':'Team'}, inplace=True)
    ops.set_index('Team', inplace=True)
    ops.info()
    #print ops.describe(include="all")
    
    #combo = pd.merge(dgl, ops, left_on='Team', right_on='FDCOUK', how='inner')
    combo = ops.combine_first(dgl)
    combo.reset_index(level=0, inplace=True)
    #combo.info()
    #print combo.describe(include="all")
    
    save_master(combo, "stadiums")
    #return combo

def format_managers():
    pieces = []
    #future_date = datetime.datetime(2099, 12, 31)
    for code, endpoints in managers_dict.items():
        dataframe = pd.read_csv(endpoints[1], encoding = "utf8", sep=',', parse_dates=['DateFrom', 'DateTo'])
        pieces.append(dataframe)
    
    # Concatenate everything into a single DataFrame
    managers = pd.concat(pieces, ignore_index=True)
    managers.drop(['ManagerCountry','Notes','Unnamed: 0'], axis=1, inplace=True)
    #managers['DateFrom'] = managers['DateFrom'].fillna(method='ffill')
    managers['DateTo'] = managers['DateTo'].fillna(datetime.date.today())
    #managers['DateTo'] = managers['DateTo'].fillna(future_date)
    
    # Fix some missing start dates
    managers.set_value(managers[(managers['Manager']=="Peter Taylor") & (managers['Team']=="Hull City")].index.values, 'DateFrom', datetime.datetime(2002, 10, 1))
    #print managers[(managers['Manager']=="Peter Taylor") & (managers['Team']=="Hull City")]
    managers.set_value(managers[(managers['Manager']=="Ray Lewington") & (managers['Team']=="Watford")].index.values, 'DateFrom', datetime.datetime(2002, 6, 1))
    #print managers[(managers['Manager']=="Ray Lewington") & (managers['Team']=="Watford")]
    
    #managers.info()
    #print managers.describe(include="all")
    #print managers[95:105]
    
    save_master(managers, "managers")
    #return managers

def build_fulldata():
    print "Building fulldata dataframe from results, stadiums, managers ..."
    
    results = get_master("results")
    stadiums = get_master("stadiums")
    managers = get_master("managers")
    managers.dropna(subset=["Manager"], inplace=True)
    
    home_renames = {
            'HomeTeam': 'Team'
            , 'AwayTeam': 'TeamOpp'
            , 'FTHG': 'Goals'
            , 'FTAG': 'GoalsOpp'
            , 'HTHG': 'Goals1stHalf'
            , 'HTAG': 'Goals1stHalfOpp'
            , 'HS': 'Shots'
            , 'AS': 'ShotsOpp'
            , 'HST': 'ShotsOnTarget'
            , 'AST': 'ShotsOnTargetOpp'
            , 'HHW': 'ShotsHitWoodwork'
            , 'AHW': 'ShotsHitWoodworkOpp'
            , 'HC': 'Corners'
            , 'AC': 'CornersOpp'
            , 'HF': 'Fouls'
            , 'AF': 'FoulsOpp'
            , 'HO': 'Offsides'
            , 'AO': 'OffsidesOpp'
            , 'HY': 'YellowCards'
            , 'AY': 'YellowCardsOpp'
            , 'HR': 'RedCards'
            , 'AR': 'RedCardsOpp'
            , 'HBP': 'BookingPoints'
            , 'ABP': 'BookingPointsOpp'
            }
    away_renames = {}
    for key,val in home_renames.items():
        if val.endswith('Opp'):
            away_renames[key] = val[:-3]
        else:
            away_renames[key] = val+'Opp'
    stat_to_diff = ['Goals','Goals1stHalf','Shots','ShotsOnTarget','ShotsHitWoodwork','Corners','Fouls','Offsides','YellowCards','RedCards','BookingPoints']
    #print list(away_renames)
    #print list(home_renames)
    
    homeresults = results.rename(columns=home_renames)
    homeresults['HomeAway'] = 'Home'
    #homeresults.info()
    #print homeresults.describe(include="all")

    awayresults = results.rename(columns=away_renames)
    awayresults['HomeAway'] = 'Away'
    #awayresults.info()
    #print homeresults.describe(include="all")
    
    allresults = pd.concat([homeresults,awayresults], ignore_index=True)
    allresults.drop(['FTR','HTR','Unnamed: 0'], axis=1, inplace=True)
    
    for stat in stat_to_diff:
        allresults[stat+'Diff'] = allresults[stat] - allresults[stat+'Opp']
        allresults['Total'+stat] = allresults[stat] + allresults[stat+'Opp']

    allresults['Saves'] = allresults['ShotsOnTargetOpp'] - allresults['GoalsOpp']
    allresults['SavesOpp'] = allresults['ShotsOnTarget'] - allresults['Goals']
    allresults['SavesDiff'] = allresults['Saves'] - allresults['SavesOpp']
    allresults['Goals2ndHalf'] = allresults['Goals'] - allresults['Goals1stHalf']
    allresults['Goals2ndHalfOpp'] = allresults['GoalsOpp'] - allresults['Goals1stHalfOpp']
    allresults['Goals2ndHalfDiff'] = allresults['GoalsDiff'] - allresults['Goals1stHalfDiff']
    #Result,Points,PointsOpp,Win,WinDraw,Draw,DrawLoss,Loss,CleanSheet,CleanSheetOpp
    allresults['Result'], allresults['Points'], allresults['PointsOpp'], allresults['Win'], allresults['WinDraw'], allresults['Draw'], allresults['DrawLoss'], allresults['Loss'], allresults['WinShare'] = zip(*allresults['GoalsDiff'].map(func_score))
    allresults['CleanSheet'] = allresults['Goals'].map(func_nogoal)
    allresults['CleanSheetOpp'] = allresults['GoalsOpp'].map(func_nogoal)
    
    #allresults['Date'] = pd.to_datetime(allresults['Date'], format="%d/%m/%y")
    
    allresults['GameWeek'] = allresults.sort_values('Date').groupby(['Season','Div','Team']).cumcount() + 1
    
    stadiums.drop(['Country','TeamFull'], axis=1, inplace=True)
    fulldata = pd.merge(allresults, stadiums, on='Team', how='left')
    #fulldata.drop(['Unnamed: 0'], axis=1, inplace=True)
    stadiums.rename(columns={'Team': 'TeamOpp'}, inplace=True)
    fulldata = pd.merge(fulldata, stadiums, on='TeamOpp', how='left', suffixes=('', 'Opp'))
    fulldata.drop(['Unnamed: 0','Unnamed: 0Opp'], axis=1, inplace=True)
    
    fulldata['EuclideanDistance'] = ( (fulldata.Latitude-fulldata.LatitudeOpp)**2 + (fulldata.Longitude-fulldata.LongitudeOpp)**2 )**0.5
    #print 100000+len(fulldata[(fulldata['Team']=="Middlesbrough")&(fulldata['TeamOpp']=="Chelsea")&(fulldata['Season']=="2016-2017")&(fulldata['HomeAway']=='Home')])

    fulldata = pd.merge(fulldata, managers, on='Team', how='left')
    #print 200000+len(fulldata[(fulldata['Team']=="Middlesbrough")&(fulldata['TeamOpp']=="Chelsea")&(fulldata['Season']=="2016-2017")&(fulldata['HomeAway']=='Home')])
    fulldata = fulldata[((fulldata['Date'] >= fulldata['DateFrom']) & (fulldata['Date'] <= fulldata['DateTo'])) | (fulldata['Manager'].isnull())]
    #print 300000+len(fulldata[(fulldata['Team']=="Middlesbrough")&(fulldata['TeamOpp']=="Chelsea")&(fulldata['Season']=="2016-2017")&(fulldata['HomeAway']=='Home')])
    fulldata.drop(['Unnamed: 0','DateFrom','DateTo','Duration','YearRange'], axis=1, inplace=True)
    fulldata = fulldata.drop_duplicates()
    #fulldata.info()
    #print 400000+len(fulldata[(fulldata['Team']=="Middlesbrough")&(fulldata['TeamOpp']=="Chelsea")&(fulldata['Season']=="2016-2017")&(fulldata['HomeAway']=='Home')])

    managers.rename(columns={'Team': 'TeamOpp'}, inplace=True)
    fulldata = pd.merge(fulldata, managers, on='TeamOpp', how='left', suffixes=('', 'Opp'))
    #print 500000+len(fulldata[(fulldata['Team']=="Middlesbrough")&(fulldata['TeamOpp']=="Chelsea")&(fulldata['Season']=="2016-2017")&(fulldata['HomeAway']=='Home')])
    fulldata = fulldata[((fulldata['Date'] >= fulldata['DateFrom']) & (fulldata['Date'] <= fulldata['DateTo'])) | (fulldata['ManagerOpp'].isnull())]
    #print 600000+len(fulldata[(fulldata['Team']=="Middlesbrough")&(fulldata['TeamOpp']=="Chelsea")&(fulldata['Season']=="2016-2017")&(fulldata['HomeAway']=='Home')])
    fulldata.drop(['Unnamed: 0','DateFrom','DateTo','Duration','YearRange'], axis=1, inplace=True)
    fulldata = fulldata.drop_duplicates()
    #fulldata.info()
    #print 700000+len(fulldata[(fulldata['Team']=="Middlesbrough")&(fulldata['TeamOpp']=="Chelsea")&(fulldata['Season']=="2016-2017")&(fulldata['HomeAway']=='Home')])
    
    #fulldata.info()
    #print fulldata.describe(include="all")
    #print fulldata[(fulldata['Team']=="Middlesbrough")&(fulldata['TeamOpp']=="Bournemouth")&(fulldata['Season']=="2016-2017")&(fulldata['HomeAway']=='Away')].describe(include="all")
    
    save_master(fulldata, "fulldata")
    #return fulldata

def get_summary(group_key, agg_method="mean", base_filters={}, metric_mins={}, output_metrics=[]):

    #fetch from master csv
    df = get_master("fulldata")
    #print list(df.columns.values)
    
    #filter unwanted records
    #df = df[(df['Team']=="Chelsea")]
    #df = df[(df['Country']=="England")]
    #df = df[(df['Tier']==1)]
    for field, vals in base_filters.items():
        df = df[(df[field].isin(vals))]
        
#    selected_columns = [group_key]+metrics
#    df = df[selected_columns]
    df.dropna(subset=[group_key], inplace=True)
    
    #aggregate data
#   df_avg = df[[group_key]+metrics].groupby(group_key).mean()
    df_avg = df.groupby(group_key).agg(agg_method)
    #df_avg.info()
    df_cnt = df[group_key].value_counts()
    #df_cnt.columns = ['NumberOfMatches']
    #print df_cnt
    df = pd.concat([df_cnt, df_avg], axis=1)
    df.rename(columns = {group_key:'NumberOfMatches'}, inplace = True)
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    
    #add derived metrics
    df["ShotAccuracy"] = df["ShotsOnTarget"] / df["Shots"]
    df["ShotAccuracyOpp"] = df["ShotsOnTargetOpp"] / df["ShotsOpp"]
    df["ShotPercent"] = df["Goals"] / df["ShotsOnTarget"]
    df["ShotPercentOpp"] = df["GoalsOpp"] / df["ShotsOnTargetOpp"]
    df["SavePercent"] = df["Saves"] / df["ShotsOnTargetOpp"]
    df["SavePercentOpp"] = df["SavesOpp"] / df["ShotsOnTarget"]
    df["ShotConversion"] = df["Goals"] / df["Shots"]
    df["ShotConversionOpp"] = df["GoalsOpp"] / df["ShotsOpp"]
    df["TSR"] = df["Shots"] / df["TotalShots"]
    df["TSROpp"] = df["ShotsOpp"] / df["TotalShots"]
    df["ShotOnTargetRatio"] = df["ShotsOnTarget"] / df["TotalShotsOnTarget"]
    df["ShotOnTargetRatioOpp"] = df["ShotsOnTargetOpp"] / df["TotalShotsOnTarget"]
    df["ShotDominance"] = df["Shots"] / df["ShotsOpp"]
    df["ShotPace"] = df["TotalShots"]
    df["PDO"] = 1000 * (df["ShotPercent"] + df["SavePercent"])
    df["PDOOpp"] = 1000 * (df["ShotPercentOpp"] + df["SavePercentOpp"])
    df["%TSoTt"] = df["ShotAccuracy"] + (1 - df["ShotAccuracyOpp"])
    df["%TSoTtOpp"] = df["ShotAccuracyOpp"] + (1 - df["ShotAccuracy"])
    df["GraysonRating"] = (0.5+(df["TSR"]-0.5)*0.732**0.5)*(1.0+(df["%TSoTt"]-1.0)*0.166**0.5)*(1000+(df["PDO"]-1000)*0.176**0.5)
    df["GraysonRatingOpp"] = (0.5+(df["TSROpp"]-0.5)*0.732**0.5)*(1.0+(df["%TSoTtOpp"]-1.0)*0.166**0.5)*(1000+(df["PDOOpp"]-1000)*0.176**0.5)
    df["GraysonScore"] = 10 * (df["GraysonRating"]-363) / (695-363)
    df["GraysonScoreOpp"] = 10 * (df["GraysonRatingOpp"]-363) / (695-363)
    
    #filter unwanted aggregate data
    #df = df[(df["NumberOfMatches"] >= 50)]
    for field, val in metric_mins.items():
        df = df[(df[field] >= val)]
    
    if output_metrics:
        df = df[output_metrics]
    #df.info()
    #print df
    return df

def main():

    unzip_files()
    format_results() ## SOME FILES DO NOT LOAD
    format_stadiums() ## name inconsistencies??
    format_managers() ## team names don't match results/stadiums data
    build_fulldata()

#   ## TO DO...
#    SOME FILES DO NOT LOAD
#    Some dates wrong
#    Other data quality checks
#    Rank (Points/Goals)
#    Date detail - Pandas or Kimball dim?
#    Place data?
   
if __name__ == '__main__':
    main()