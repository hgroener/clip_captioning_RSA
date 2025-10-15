## HYPERPARAMETER TUNING ## 

import itertools
import os 
from tqdm import tqdm 
import pandas as pd
import json 
import re 
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Adobe Garamond Pro"
})
from os.path import dirname, abspath
from transformers import GPT2Tokenizer
#import statistics
from scipy.stats import gmean
from generate import generate_RSA, load_clip_model
from eval_informativity import eval_informativity
from cider.cidereval import eval_cider
import numpy as np

current_folder = dirname(abspath(__file__))
parent_folder = dirname(current_folder)
image_path=current_folder + "/data/AbstractScenes_v1.1/RenderedScenes/"
save_path = os.path.join(parent_folder, "pretrained_models")
model_path = os.path.join(save_path, 'conceptual_weights.pt')

# load clipcap and gpt-2 model
generation_model = load_clip_model(model_path)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# hyperparameters
temp_range = [i/100 for i in list(range(60,110,10))]
a_range = [3,4,5]


def grid_search(train_df, combs, # combinations of hyperparameter values 
                output_dir=current_folder + "/data/AbstractScenes_v1.1/parameter_tuning/", result_file= "results_finetuning_nn.json", 
                generate=True, # set to False to load captions from dataframe instead of generating new captions 
                k=3):
    
    # create output directory if it doesn't exist already 
    os.makedirs(output_dir, exist_ok = True)

    scores = {}
    for temp, a in tqdm(combs, total=len(list(combs))):

        key = "temp{}a{}".format(str(int(temp*100)), str(int(a)))
        if generate:
            print("generating captions with temperature={}, a={}".format(temp, a))
            train_df, _, _ = generate_RSA(generation_model, tokenizer, train_df, temperature = temp, t=1, a=a, top_k=100, cname="train_cap", 
                                    image_path=image_path)
        else: 
            print("loading captions generated with temperature={}, a={}".format(temp, a))
            train_df = pd.read_csv("./data/AbstractScenes_v1.1/parameter_tuning/{}.csv".format(key))
        
        print("evaluating scenes...")        
        cider = eval_cider(train_df, target_dir=output_dir, pathToData = output_dir, 
                           result_file = "results_" + key +  ".json" , gen_column="train_cap")["CIDEr"]
        train_df["CIDEr"] = cider
        mean_cider = sum(cider)/len(cider)

        # calculate informitivity of captions for each scene 
        for scene in tqdm(list(set(train_df["scene_idx"]))):
            # get subset of dataframe with rows (images) belonging to current scene 
            scene_df = train_df[train_df["scene_idx"]==scene]
            caps = list(scene_df["train_cap"])
            files = list(scene_df["file"])
            informative = [int(eval_informativity(cap, files[i], files, k=k)[2]) for i, cap in enumerate(caps)]
            # write informativity value to dataframe 
            train_df.loc[train_df.scene_idx==scene, 'informative'] = informative
    
        train_df.to_csv(output_dir + "{}.csv".format(key))
        # calculate mean informativity value for data 
        mean_informative = sum(list(train_df['informative']))/len(train_df)
        scores[key] = (mean_cider, mean_informative)
    
    # save scores to json file 
    with open(output_dir + result_file, "w") as f:
        json.dump(scores, f)
    return(scores)




def aggregate_scores(combs, scores, scores_path = current_folder + "/output/hp_tuning/"):

    for temp, a in combs: 
        key = "temp{}a{}".format(str(int(temp*100)), str(int(a)))
        file =  scores_path + "{}.csv".format(key)
        df = pd.read_csv(file)
        # calculate mean informativity value 
        informative = sum([int(i) for i in list(df["informative"])])/len(list(df["informative"]))
        scores[key] = (scores[key], informative)

    cider_scores = []
    inf_scores = []

    for k in scores.keys():
        # get hyperparameter values from dictionary names 
        temp= int(re.search("(?<=temp)\d+", k).group(0))
        a= int(re.search("(?<=a)\d", k).group(0))
        cider, inf = scores[k] 
        # add evaluation results to list 
        cider_scores.append((temp, a, cider[0]))
        inf_scores.append((temp, a, inf))

    # calculate hmean scores of informativity and cider scores for each hyperparameter combination
    hmean_scores = [(c[0], c[1], gmean([c[2], i[2]])) for c, i in zip(cider_scores, inf_scores)]

    return(cider_scores, inf_scores, hmean_scores)


# create figures from scores 
def create_figs(scores, score_type="CIDEr", show=True, output_path=""):
    plt.figure()
    fig, ax = plt.subplots()
    N = 10 # <- you can adjust the step here

    for a in list(set([a for temp, a, mean in scores])):
        a_scores = [p for p in scores if p[1]==a]
        ax.plot([p[0] for p in a_scores], [p[2] for p in a_scores], label=str(a))

    ax.set_xticks(np.arange(min([p[0] for p in a_scores]), max([p[0] for p in a_scores]) + 1, N))
    ax.set_xlabel("Temperatur")
    ax.set_ylabel(score_type)
    ax.legend(loc="upper right", title="a")

    if show: 
        plt.show()

    if output_path:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        #fname = output_path + score_type + '_scores.png'
        fname = output_path + score_type + '_scores.pdf'
        plt.savefig(fname)
        print("{} figure saved to {}".format(score_type, fname))



def main(generate=True, score_file=None, output_path=current_folder + "/output/hp_tuning/"):
    os.makedirs(output_path, exist_ok=True)
    combs = list(itertools.product(temp_range, a_range)) # get all combinations of hyperparameter values 
    print("number of hyperparameter combinations: {}".format(len(combs)))

    df = pd.read_csv(current_folder + "/data/AbstractScenes_v1.1/processed_data/train_df_r3.csv")
    if generate: 
        # generate captions with given hyperparameter combinations 
        scores = grid_search(df, combs, output_dir=output_path, result_file="hp_tuning_scores.json", k=3)
    elif score_file: 
        # load scores from files 
        with open(score_file) as f:
            scores = json.load(f)
        print("scores loaded from {}.".format(score_file))
    else: 
        print("need to generate captions or provide score file.")
        return
    #aggregate results for each hyperparameter combination
    cider_scores, inf_scores, hmean_scores = aggregate_scores(combs, scores)
    # save to json files 
    for scores, fname in [(cider_scores, "cider_scores"), (inf_scores, "informativity_scores"), (hmean_scores, "hmean_scores")]:
        with open(output_path + fname, "w+") as f: 
            json.dump(scores,f)
    # create figures 
    for f in [(cider_scores, "CIDEr"), (inf_scores, "Informativität"), (hmean_scores, "hmean")]:
        create_figs(f[0], score_type=f[1], output_path=output_path + "/figures/", show=False)



if __name__=="__main__":
    generate = False # set to true to generate captions (this will take a long time!)
    score_file = "/srv/storage/hgroener/other/clip_captioning_RSA/output/hp_tuning/hp_tuning_scores.json"
    main(generate=generate, score_file=score_file)