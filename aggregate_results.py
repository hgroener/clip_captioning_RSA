import pandas as pd
import os 
import json
import itertools
from test import RSA_OPTIONS, DECODING_OPTIONS

current_folder = os.path.dirname(os.path.abspath(__file__))

test_path = current_folder + "/data/AbstractScenes_v1.1/testing/"
score_dics = []
for rsa, decoding in itertools.product(RSA_OPTIONS.keys(), DECODING_OPTIONS):
    fpath = "{}{}/{}/".format(test_path, decoding, rsa)
    file = "test_scores_{}_{}.txt".format(decoding, rsa)
    with open(fpath + file) as f:
        scores = json.load(f)

    t, pos = RSA_OPTIONS[rsa].values()
    rsa_bool = rsa!="no_RSA"
    score_dics.append({"decoding": decoding, "RSA": rsa_bool, "t": t, "POS": pos, "CIDEr": scores["cider"], "informativity": scores["informative"]})

## aggregating scores


score_df = pd.DataFrame(score_dics)
score_df.to_csv(test_path + "summary.csv")