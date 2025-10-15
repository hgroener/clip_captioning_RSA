## AGGREGATE SCORES FROM TEST TABLES TO CREATE SUMMARY TABLE ##

import pandas as pd
import os 
import json
import itertools
from test import RSA_OPTIONS, DECODING_OPTIONS

current_folder = os.path.dirname(os.path.abspath(__file__))

test_path = current_folder + "/output/test_results/"
output_csv_path = current_folder + "/output/test_results/"
score_dics = []
for rsa, decoding in itertools.product(RSA_OPTIONS.keys(), DECODING_OPTIONS):
    # read scores for every configuration of RSA decoding and "traditional" decoding (greedy & beam search)
    fpath = "{}{}/{}/".format(test_path, decoding, rsa)
    file = "test_scores_{}_{}.txt".format(decoding, rsa)
    with open(fpath + file) as f:
        scores = json.load(f)

    # check boolean values for RSA decoding modifications for summary table 
    t, pos = RSA_OPTIONS[rsa].values()
    rsa_bool = rsa!="no_RSA"
    rsa_int = int(rsa_bool)
    decoding = "Beam Search" if decoding=="beam_search" else "Greedy"
    t_dic = {"0": "n.a.", "1": "0", "0.7": "1"}
    t = t_dic[str(t)]
    pos = int(pos)
    pos = "n.a." if not rsa_bool else pos

    # convert quality and informativity scores to %
    cider = "%.2f" % round(scores["cider"] * 100, 2)
    informative = "%.2f" % round(scores["informative"]*100, 2)
    score_dics.append({"Dekodierung": decoding, "RSA": rsa_int, "t": t, "POS": pos, "CIDEr [%]": cider, "Informativität [%]": informative})

## aggregating scores


score_df = pd.DataFrame(score_dics)
os.makedirs(output_csv_path, exist_ok=True)
summary_file = output_csv_path + "summary.csv"
score_df.to_csv(summary_file, index=False)
print("Aggregated test scores saved to {}".format(summary_file))