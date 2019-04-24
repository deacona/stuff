#!/usr/bin/python -tt
"""
Created on Tue May 09 11:28:47 2017

@author: adeacon
"""

import requests
from bs4 import BeautifulSoup

# import re
import pandas as pd

# import time
import matplotlib.pyplot as plt

# import matplotlib.dates as mdates
# import datetime as dt
# import time
import numpy as np
# import sys
import config

pd.set_option("display.max_columns", 500)

min_year = 1977
max_year = 2017
use_local = 1


def make_soup(url):
    response = requests.get(url)
    html = response.content

    # soup = BeautifulSoup(html)
    return BeautifulSoup(html, "lxml")
    # print soup.prettify()


def _unpack(row, kind="td"):
    elts = row.findAll("%s" % kind)
    return [val.text for val in elts]


def get_elections():
    soup = make_soup(
        "https://en.wikipedia.org/wiki/List_of_United_Kingdom_general_elections"
    )

    table = soup.find("table", attrs={"class": "wikitable sortable"})
    rows = table.findAll("tr")

    data = []
    for r in rows:
        data_row = _unpack(r, kind=("th")) + _unpack(r, kind=("td"))
        # print data_row
        data.append(data_row)
    dataframe = pd.DataFrame(data[1:], columns=data[0])
    # dataframe['Year'] = pd.to_numeric(dataframe['Election'].str[:4], errors='coerce')
    dataframe["Year"] = pd.to_numeric(dataframe["Election"].str[:4], errors="coerce")
    # dataframe['Year'] = pd.to_datetime(dataframe['Election'].str[:4], errors='coerce', format='%Y')
    # dataframe['Year'] = dataframe['Election'].str[:4].str.extract('(\d+)').astype(int)
    dataframe.dropna(subset=["Year"], inplace=True)
    dataframe["Year"] = dataframe["Year"].astype(int)
    # dataframe.info()
    # print(dataframe.describe(include="all"))
    # print dataframe[:4]

    return dataframe[(dataframe["Year"] >= min_year) & (dataframe["Year"] <= max_year)]


def get_population():
    population = pd.read_excel(
        config.SOURCE_DIR + r"\ons\population\a33197de.xls",
        "UK Population",
        names=[
            "Blank",
            "Year",
            "TotalPopulation",
            "AnnualGrowthRatePerc",
            "AnnualGrowth",
        ],
    )
    population.drop(population.index[:4], inplace=True)
    del population["Blank"]

    # population['Year'] = population['Year'][:4]
    population["Year"] = pd.to_numeric(population["Year"], errors="coerce")
    population.dropna(subset=["Year"], inplace=True)

    # population = pd.concat([population1, population2], ignore_index=True)
    population["Year"] = population["Year"].astype(int)

    population["TotalPopulation"] = pd.to_numeric(
        population["TotalPopulation"], errors="coerce"
    )
    population.dropna(subset=["TotalPopulation"], inplace=True)
    population["TotalPopulation"] = population["TotalPopulation"].astype(int)

    # population.info()
    # print(population.describe(include="all"))
    # print population[['Year','TotalPopulation']].ix[40:65]
    # print population[(population['Year'] >= min_year) & (population['Year'] <= max_year)]
    return population[
        (population["Year"] >= min_year) & (population["Year"] <= max_year)
    ]


def get_results():
    if use_local:
        # print use_local
        return pd.read_csv(
            config.MASTER_FILES["pop_results"], encoding="utf-8", sep="|"
        )
    else:
        pieces = []
        for index, row in get_elections().iterrows():
            print("Fetching results for " + str(row["Year"]) + "...")
            soup = make_soup(
                "https://en.wikipedia.org/wiki/United_Kingdom_general_election,_"
                + str(row["Year"])
            )

            if row["Year"] <= 1979:
                table = soup.find(
                    "caption", text="UK General Election " + str(row["Year"])
                ).find_parent("table")
            elif row["Year"] <= 1983:
                table = soup.find(
                    "caption",
                    text="United Kingdom General Election " + str(row["Year"]),
                ).find_parent("table")
            elif row["Year"] <= 2001:
                table = soup.find("table", attrs={"class": "wikitable"})
            else:
                table = soup.find(
                    "table",
                    attrs={
                        "class": "wikitable sortable",
                        "style": "text-align:right; font-size: 95%;",
                    },
                )
            rows = table.findAll("tr")

            data = []
            for r in rows:
                data_row = _unpack(r, kind=("th")) + _unpack(r, kind=("td"))
                # print str(len(data_row))+" / "+data_row[1]
                if (len(data_row) > 10) and (
                    data_row[1] not in ("Political party", "Leader")
                ):
                    if row["Year"] <= 2001:
                        data_sel = list(data_row[i] for i in [1, 2, 3, 4, 10])
                    else:
                        data_sel = list(data_row[i] for i in [1, 2, 3, 4, 9])
                    data.append(data_sel)

            # print data[:5]
            # print data[2:5]

            dataframe = pd.DataFrame(
                data,
                columns=["Political party", "Leader", "Candidates", "Elected", "Votes"],
            )
            dataframe["Year"] = row["Year"]
            # print dataframe
            # for col in [u'MPsCandidates', u'MPsTotal', u'MPsGained', u'MPsLost', u'MPsPercOfTotal', u'VotesTotal', u'VotesPercOfTotal']:
            for col in ["Candidates", "Elected", "Votes"]:
                # print dataframe[col]
                dataframe[col] = pd.to_numeric(
                    dataframe[col].astype(str).str.replace(",", ""), errors="coerce"
                )
            # dataframe.info()
            # print dataframe.describe(include="all")
            # print dataframe[:4]
            pieces.append(dataframe)

        # Concatenate everything into a single DataFrame
        results = pd.concat(pieces, ignore_index=True)
        # results.info()
        # print results.describe(include="all")
        results["Political party"] = results["Political party"].str.replace(
            r"\[.*\]", ""
        )
        # print results[results["Political party"]=="Conservative"]
        results.to_csv(config.MASTER_FILES["pop_results"], encoding="utf-8", sep="|")
        return results


# def func_type(x):
#    return config.UK_PARTIES[x]


def show_dfplot(dftable, ylabel, ymax, yinterval, dfplotfile):
    dfplot = dftable.plot(kind="line", x="Year", y="Total")
    dfplot.set_xlabel("Year")
    dfplot.set_ylabel(ylabel)
    dfplot.set_xticks(list(range(min_year, max_year)), minor=True)
    dfplot.set_yticks(list(range(0, ymax, yinterval)), minor=True)
    dfplot.grid("on", which="minor", axis="x")
    dfplot.grid("on", which="minor", axis="y")

    plt.savefig(config.ANALYSIS_DIR + "/" + dfplotfile)


def do_analysis():
    print("do_analysis: population, results ...")
    population = get_population()
    results = get_results()
    # print population[['Year','TotalPopulation']]

    results["Type"] = results["Political party"]  # .map(func_type)
    results["Total"] = results["Votes"]
    # print results

    results_sub = results[["Year", "Votes"]].groupby("Year").sum()
    results_sub["Type"] = "Voters"
    results_sub["Total"] = results_sub["Votes"]
    results_sub.reset_index(inplace=True)
    # results_sub.info()
    # print results_sub

    population["Type"] = "Population"
    population["Total"] = population["TotalPopulation"]

    results["Type"] = results["Political party"].map(config.UK_PARTIES)
    results["Type"].fillna("Other", inplace=True)
    # print results

    fulldata = pd.concat(
        [results, results_sub, population], ignore_index=True, sort=True
    )
    # check for any other large parties to attribute)
    # print fulldata[['Year','Political party','Type','Leader','Votes']][(fulldata['Type']=="Other") & (fulldata['Votes']>=100000)].nlargest(10, 'Votes')
    # test = fulldata[['Year','Political party','Type','Leader','Votes']][(fulldata['Type']=="Other") & (fulldata['Votes']>=100000)].nlargest(2, 'Votes')
    # print test["Political party"].tolist()
    # sys.exit(0)

    if use_local:
        fulldata.drop(
            [
                "Political party",
                "Leader",
                "Candidates",
                "Elected",
                "AnnualGrowth",
                "AnnualGrowthRatePerc",
                "TotalPopulation",
                "Votes",
                "Unnamed: 0",
            ],
            axis=1,
            inplace=True,
        )
    else:
        fulldata.drop(
            [
                "Political party",
                "Leader",
                "Candidates",
                "Elected",
                "AnnualGrowth",
                "AnnualGrowthRatePerc",
                "TotalPopulation",
                "Votes",
            ],
            axis=1,
            inplace=True,
        )
    # fulldata['Type'] = fulldata['Political party'].map(func_type)

    # fulldata.info()
    # print fulldata.describe(include="all")
    # print fulldata
    # print fulldata[['Year','Political party','VotesTotal','TotalPopulation']]
    # print fulldata["Political party"].value_counts()

    print("\n\nTotal (millions)...")

    pop_group = fulldata.groupby(["Year", "Type"]).sum()
    # pop_group.info()
    # print pop_group.unstack()
    pop_table = pop_group.unstack().reset_index()  # .ffill()
    pop_table = pop_table.interpolate(method="linear", axis=0)  # .ffill().bfill()

    # manually add 2016 EU Referendum
    pop_table.loc[len(pop_table.index)] = np.array(
        [2016, 0, 65572409, 16141241, 17410742, 33551983]
    )

    # manually add 2017 election - FOR NOW
    pop_table.loc[len(pop_table.index)] = np.array(
        [
            2017,
            0,
            66029928,
            (12874985 + 2371772 + 977569 + 525371 + 164466),
            (13632914 + 593852 + 4642),
            (
                (12874985 + 2371772 + 977569 + 525371 + 164466)
                + (13632914 + 593852 + 4642)
            ),
        ]
    )

    pop_table["Total"] = pop_table["Total"] / 1000000
    # pop_table.info()
    print(pop_table)
    show_dfplot(pop_table, "Total (millions)", 70, 5, "pop_plot.png")

    print("\n\n% of Population...")
    xpop_table = pop_table.copy()
    xpop_table["Total"] = 100 * xpop_table["Total"].div(
        xpop_table["Total", "Population"], axis=0
    )
    print(xpop_table)
    show_dfplot(xpop_table, "% of Population", 100, 5, "pop_xpop_plot.png")

    # time.sleep(20)
    print("\n\n% of Voters...")
    xvot_table = pop_table.copy()
    xvot_table["Total"] = 100 * xvot_table["Total"].div(
        xvot_table["Total", "Voters"], axis=0
    )
    print(xvot_table)
    show_dfplot(xvot_table, "% of Voters", 250, 10, "pop_xvot_plot.png")

    # time.sleep(20)
    print("\n\n% of Progressive Voters...")
    xpro_table = pop_table.copy()
    xpro_table["Total"] = 100 * xpro_table["Total"].div(
        xpro_table["Total", "Progressive"], axis=0
    )
    print(xpro_table)
    show_dfplot(xpro_table, "% of Progressive Voters", 650, 20, "pop_xpro_plot.png")


def main():
    # get_elections()
    # get_population() # years mix int and str
    # get_results() # table formats change pre-2010
    do_analysis()


if __name__ == "__main__":
    main()
