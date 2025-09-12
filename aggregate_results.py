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
    rsa_int = int(rsa_bool)
    decoding = "Beam Search" if decoding=="beam_search" else "Greedy"
    t_dic = {"0": "n.a.", "1": "0", "0.7": "1"}
    t = t_dic[str(t)]
    pos = int(pos)
    cider = "%.2f" % round(scores["cider"] * 100, 2)
    informative = "%.2f" % round(scores["informative"], 2)
    score_dics.append({"Dekodierung": decoding, "RSA": rsa_int, "t": t, "POS": pos, "CIDEr": cider, "Informativität": informative})

## aggregating scores


score_df = pd.DataFrame(score_dics)
summary_file = test_path + "summary.csv"
score_df.to_csv(summary_file)
print("Aggregated test scores saved to {}".format(summary_file))