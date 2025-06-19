#!/usr/bin/env python
# coding: utf-8

# demo script for running CIDEr
from pydataformat.loadData import LoadData
import pdb
import json
from pyciderevalcap.eval import CIDErEvalCap as ciderEval
import pandas as pd
import os


def eval_cider(df, target_dir="../data/AbstractScenes_v1.1/parameter_tuning/", pathToData = "../data/AbstractScenes_v1.1/parameter_tuning/", 
               result_file = 'results.json', gen_column = "cap_RSA_t"):

    gold_caps = []
    gen_caps = []
    
    for c, row in df.iterrows():
        gold = [{"image_id": row["file"], "caption": row["cap" + str(i)]} for i in range(6) if type(row["cap" + str(i)])==str]
        gen = {"image_id": row["file"], "caption": row[gen_column]}
        
        gold_caps += gold
        gen_caps.append(gen)
        
    os.makedirs(target_dir, exist_ok=True)
        
    with open(target_dir + "gold_caps.json", "w") as f:
        json.dump(gold_caps, f)
    with open(target_dir + "gen_caps.json", "w") as f:
        json.dump(gen_caps, f)
    
    
    # load reference and candidate sentences
    loadDat = LoadData(pathToData)
    gts, res = loadDat.readJson("gold_caps.json", "gen_caps.json")
    
    
    scorer = ciderEval(gts, res, "corpus")
    # scores: dict of list with key = metric and value = score given to each candidate
    
    scores = scorer.evaluate()
    
    # scores['CIDEr'] contains CIDEr scores
    # scores['CIDErD'] contains CIDEr-D scores
    
    with open(target_dir + result_file, 'w') as outfile:
        json.dump(scores, outfile)
    
    return(scores)

