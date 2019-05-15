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
    "fin_master": os.path.join(MASTER_DIR, "fin_master.txt"),
    "fin_legacy": os.path.join(MASTER_DIR, "fin_legacy.csv"),
    "date_dim": os.path.join(MASTER_DIR, "Ch10-DateDim_AD.xls"),
    "fin_datasheet": os.path.join(ANALYSIS_DIR, "fin_analysis_DATA.xlsx"),
    "fin_budget_template": os.path.join(ANALYSIS_DIR, "budget_planner_TEMPLATE.xlsx"),
    "fin_budget_output": os.path.join(ANALYSIS_DIR, "budget_planner_TEST.xlsx"),
}
UK_PARTIES = csv_to_dict("pop_parties.csv")
FIN_PASS = conf["THING"]["PASSWORD"]
FIN_ACCOUNTS = {
    conf["ACCOUNT01"]["CODE"]: [conf["ACCOUNT01"]["WITH"], conf["ACCOUNT01"]["NAME"]],
    conf["ACCOUNT02"]["CODE"]: [conf["ACCOUNT02"]["WITH"], conf["ACCOUNT02"]["NAME"]],
    conf["ACCOUNT03"]["CODE"]: [conf["ACCOUNT03"]["WITH"], conf["ACCOUNT03"]["NAME"]],
}
FIN_DOCS = {
    conf["ACCOUNT01"]["CODE"]: os.path.join(SOURCE_DIR, conf["ACCOUNT01"]["WITH"]),
    conf["ACCOUNT02"]["CODE"]: os.path.join(SOURCE_DIR, conf["ACCOUNT02"]["WITH"]),
    conf["ACCOUNT03"]["CODE"]: os.path.join(SOURCE_DIR, conf["ACCOUNT03"]["WITH"]),
    "fin": MASTER_DIR,
    # "Annual estimate": ANALYSIS_DIR,
    "budget_planner": ANALYSIS_DIR,
    "fin_analysis": ANALYSIS_DIR,
    "cash_calculator": ANALYSIS_DIR,
    "AD finances": ANALYSIS_DIR,
}
FIN_GROUPS = csv_to_dict("fn_groups.csv")
