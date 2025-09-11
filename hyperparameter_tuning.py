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

generation_model = load_clip_model(model_path)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
### hyperparameter tuning
# hyperparameters
temp_range = [i/100 for i in list(range(60,110,10))]
#t_range = [0.5, 0.6, 0.7, 0.8]
a_range = [3,4,5]
#p_range = [i/100 for i in list(range(60,105,5))]



# grid search

def grid_search(train_df, combs, output_dir=current_folder + "/data/AbstractScenes_v1.1/parameter_tuning/", result_file= "results_finetuning_nn.json", generate=True, k=3):
    os.makedirs(output_dir, exist_ok = True)
    #combs = itertools.product(temp_range, p_range)
    scores = {}
    for temp, a in tqdm(combs, total=len(list(combs))):

        key = "temp{}a{}".format(str(int(temp*100)), str(int(a)))
        if generate:
            print("generating captions with temperature={}, a={}".format(temp, a))
            train_df, _, _ = generate_RSA(generation_model, tokenizer, train_df, temperature = temp, t=1, a=a, top_k=100, cname="train_cap", 
                                    image_path=image_path)
            #train_df["train_cap"] = caps 
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




def aggregate_scores(combs, scores, scores_path = current_folder + "/data/AbstractScenes_v1.1/parameter_tuning/"):

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
        cider_scores.append((temp, a, cider[0]))
        inf_scores.append((temp, a, inf))


    #total_cider = sum([p[2] for p in cider_scores])
    #total_inf = sum([p[2] for p in inf_scores])
    #mean_scores = [(c[0], c[1], (c[2]/total_cider + i[2]/total_inf)/2) for c, i in zip(cider_scores, inf_scores)]
    hmean_scores = [(c[0], c[1], gmean([c[2], i[2]])) for c, i in zip(cider_scores, inf_scores)]

    return(cider_scores, inf_scores, hmean_scores)


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



def main(generate=True, score_file=None):
    combs = list(itertools.product(temp_range, a_range))
    print("number of hyperparameter combinations: {}".format(len(combs)))

    df = pd.read_csv(current_folder + "/data/AbstractScenes_v1.1/processed_data/train_df_r3.csv")
    if generate: 
        scores = grid_search(df, combs, result_file="hp_tuning_scores.json", k=3)
    elif score_file: 
        with open(score_file) as f:
            scores = json.load(f)
        print("scores loaded from {}.".format(score_file))
    else: 
        print("need to generate captions or provide score file.")
        return
    cider_scores, inf_scores, hmean_scores = aggregate_scores(combs, scores)
    for scores, fname in [(cider_scores, "cider_scores"), (inf_scores, "informativity_scores"), (hmean_scores, "hmean_scores")]:
        with open("{}/data/AbstractScenes_v1.1/parameter_tuning/{}.json".format(current_folder, fname), "w+") as f: 
            json.dump(scores,f)
    for f in [(cider_scores, "CIDEr"), (inf_scores, "Informativität"), (hmean_scores, "hmean")]:
        create_figs(f[0], score_type=f[1], output_path=current_folder + "/figures/", show=False)



if __name__=="__main__":
    generate = False
    score_file = "/srv/storage/hgroener/other/clip_captioning_RSA/data/AbstractScenes_v1.1/parameter_tuning/hp_tuning_scores.json"
    main(generate=generate, score_file=score_file)