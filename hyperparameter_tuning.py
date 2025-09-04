import itertools
import os 
from tqdm import tqdm 
import pandas as pd
import json 
import re 
import matplotlib.pyplot as plt 

from generate import generate_RSA_t
from eval_informativity import eval_informativity
from cider.cidereval import eval_cider

### hyperparameter tuning
# hyperparameters
temp_range = [i/100 for i in list(range(60,110,10))]
#t_range = [0.5, 0.6, 0.7, 0.8]
a_range = [3,4,5]
#p_range = [i/100 for i in list(range(60,105,5))]

# grid search

def grid_search(train_df, combs, output_dir="./data/AbstractScenes_v1.1/parameter_tuning/", result_file= "results_finetuning_nn.json", generate=True, k=3):
    os.makedirs(output_dir, exist_ok = True)
    #combs = itertools.product(temp_range, p_range)
    scores = {}
    for temp, a in tqdm(combs, total=len(list(combs))):

        key = "temp{}a{}".format(str(int(temp*100)), str(int(a)))
        if generate:
            print("generating captions with temperature={}, a={}".format(temp, a))
            caps, _ = generate_RSA_t(model, tokenizer, train_df, temperature = temp, t=1, a=a, top_k=100)
            train_df["train_cap"] = caps 
        else: 
            print("loading captions generated with temperature={}, a={}".format(temp, a))
            train_df = pd.read_csv("./data/AbstractScenes_v1.1/parameter_tuning/{}.csv".format(key))
        #train_df["informative"] = ["nan"]*len(train_df)
        
        print("evaluating scenes...")        
        cider = eval_cider(train_df, target_dir=output_dir, pathToData = output_dir, 
                           result_file = "results_" + key +  ".json" , gen_column="train_cap")["CIDEr"]
        train_df["CIDEr"] = cider
        mean_cider = sum(cider)/len(cider)
    
        for scene in tqdm(list(set(train_df["scene_idx"]))):
            scene_df = train_df[train_df["scene_idx"]==scene]
            caps = list(scene_df["train_cap"])
            files = list(scene_df["file"])
            informative = [int(eval_informativity(cap, files[i], files, k=k)[2]) for i, cap in enumerate(caps)]
            train_df.loc[train_df.scene_idx==scene, 'informative'] = informative
    
        train_df.to_csv(output_dir + "{}.csv".format(key))
        mean_informative = sum(list(train_df['informative']))/len(train_df)
        scores[key] = (mean_cider, mean_informative)
    
    with open(output_dir + result_file, "w") as f:
        json.dump(scores, f)
    return(scores)




def aggregate_scores(combs, scores, scores_path = "./data/AbstractScenes_v1.1/parameter_tuning/"):

    for temp, a in combs: 
        key = "temp{}a{}".format(str(int(temp*100)), str(int(a)))
        file =  scores_path + "{}.csv".format(key)
        df = pd.read_csv(file)
        informative = sum([int(i) for i in list(df["informative"])])/len(list(df["informative"]))
        scores[key] = (scores[key], informative)

    cider_scores = []
    inf_scores = []

    for k in scores.keys():
        temp= int(re.search("(?<=temp)\d+", k).group(0))
        a= int(re.search("(?<=a)\d", k).group(0))
        cider, inf = scores[k] 
        cider_scores.append((temp, a,cider))
        inf_scores.append((temp, a, inf))


    total_cider = sum([p[2][0] for p in cider_scores])
    total_inf = sum([p[2] for p in inf_scores])
    mean_scores = [(c[0], c[1], (c[2][0]/total_cider + i[2]/total_inf)/2) for c, i in zip(cider_scores, inf_scores)]

    return(cider_scores, inf_scores, mean_scores)


def create_figs(scores, score_type="CIDEr", show=True, output_path=""):
    plt.figure()

    for a in list(set([a for temp, a, mean in scores])):
        a_scores = [p for p in scores if p[1]==a]
        plt.plot([p[0] for p in a_scores], [p[2][0] for p in a_scores], label=str(a))

    plt.title("decoding parameter testing")
    plt.xlabel("temperature")
    plt.ylabel(score_type)
    plt.legend(loc="upper right", title="a")
    if show: 
        plt.show()

    if output_path:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        fname = output_path + score_type + '_scores.png'
        plt.savefig(fname)
        print("{} figure saved to {}".format(score_type, fname))



def main():
    combs = list(itertools.product(temp_range, a_range))
    print("number of hyperparameter combinations: {}".format(len(combs)))

    df = pd.read_csv("./processed_data/train_df_r3.csv")
    scores = grid_search(df, combs, result_file="results_finetuning_nn_r3.json", k=3)
    cider_scores, inf_scores, mean_scores = aggregate_scores(combs, scores)
    for f in [(cider_scores, "CIDEr score"), (inf_scores, "informativity"), (mean_scores, "mean")]
        create_figs(f[0], score_type=f[1], output_path="./data/AbstractScenes_v1.1/parameter_tuning/figs", show=False)



if __name__=="__main__":
    main()