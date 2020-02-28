#!/usr/bin/python -tt
import configparser
import os
import csv

conf = configparser.RawConfigParser()
conf.read("config.ini")

def csv_to_dict(filename):
    reader = csv.reader(open(os.path.join(MASTER_DIR, filename), 'r'))
    d = {}
    for row in reader:
       k, v = row
       d[k] = v
    return d

DOWNLOAD_DIR = conf["PROJECT"]["DOWNLOAD_DIR"]
SOURCE_DIR = conf["PROJECT"]["SOURCE_DIR"]
MASTER_DIR = conf["PROJECT"]["MASTER_DIR"]
ANALYSIS_DIR = conf["PROJECT"]["ANALYSIS_DIR"]

MASTER_FILES = {
    "pop_results": os.path.join(MASTER_DIR, "pop_results.txt"),
    "date_dim": os.path.join(MASTER_DIR, "Ch10-DateDim_AD.xls"),
}
UK_PARTIES = csv_to_dict("pop_parties.csv")
