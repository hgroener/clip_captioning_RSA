## TESTING ## 

import pandas as pd
import os 
from os.path import dirname, abspath
from tqdm import tqdm
import itertools
from transformers import GPT2Tokenizer
import json

import generate as gen
from eval_informativity import eval_informativity
from cider.cidereval import eval_cider

# settings for "classic" decoding, RSA decoding with respective settings for threshold based/POS based RSA-decoding
DECODING_OPTIONS = ["greedy", "beam_search"]
RSA_OPTIONS = {"no_RSA": {"t": 0, "pos": False}, 
               "RSA": {"t": 1,"pos": False}, 
               "RSA_t": {"t": 0.7, "pos": False},
               "RSA_POS": {"t": 1, "pos": True}}


def test(df, model= None, tokenizer=None, t=1, # set threshold value to 0 to disable RSA decoding, set to 1 to disable threshold based RSA-decoding
         pos = False, # flag for POS based RSA decoding
         beam_search = False, # set to True for beam search, else greedy decoding 
         a=4, # rationality hyperparameter alpha
         temp=0.9, # temperature hyperparameter 
         target_dir="./output/", 
         image_path = "./data/AbstractScenes_v1.1/RenderedScenes/", 
         cname="RSA_caps", # output column name 
         results_file="results.json",
         results_csv="RSA_caps.csv", score_file = "scores.json", k=3, # number of images per scene 
         presave=True, # save generated captions before evalution 
         group_by = "scene_idx", generate=True # set to False to load captions from file instead of generating
         ):
    
    # make output directory if it doesn't exist already 
    os.makedirs(target_dir, exist_ok = True)
    print("generating captions with temperature={}, t={}, a={}, pos_decoding={}, beam_search={}".format(temp, t, a, pos, beam_search))
    if generate: # generate captions
        if not beam_search: # greedy decoding
            df,_, _ = gen.generate_RSA(model, tokenizer, df, temperature = temp, t=t, a=a, group_by=group_by, pos_decoding=pos, 
                                       image_path=image_path, cname=cname)
        else: # beam search
            df, _, _ = gen.generate_beam_RSA(df, generation_model=model, tokenizer=tokenizer, temperature = temp, t=t, a=a, pos=pos, image_path=image_path, cname=cname)
        print("captions generated.")
        if presave: 
            df.to_csv(target_dir + results_csv)
    else: # load captions from file 
        df = pd.read_csv(target_dir + results_csv)
    
    print("evaluating scenes...")
    print("calculating CIDEr scores...")
    cider = eval_cider(df, target_dir=target_dir, pathToData = target_dir, 
                       result_file = results_file , gen_column=cname)["CIDEr"]
    df["CIDEr"] = cider
    mean_cider = sum(cider)/len(cider)

    print("calculating informativity scores...")
    for scene in tqdm(list(set(df[group_by]))):
        scene_df = df[df[group_by]==scene]
        caps = list(scene_df[cname])
        files = list(scene_df["file"])
        informative = [int(eval_informativity(cap, files[i], files, k=k, image_path=image_path)[2]) for i, cap in enumerate(caps)]
        df.loc[df[group_by]==scene, 'informative'] = informative

    df.to_csv(results_csv)
    mean_informative = sum(list(df['informative']))/len(df)
    
    with open(target_dir + score_file, "w+") as f:
        json.dump({'cider': mean_cider, "informative": mean_informative}, f)

    return(df, (mean_cider, mean_informative))

def main(test_file, target_dir, model=None, tokenizer=None, image_path="/data/AbstractScenes_v1.1/RenderedScenes/"):
    test_df = pd.read_csv(test_file)
    for rsa, decoding in itertools.product(list(RSA_OPTIONS.keys()), DECODING_OPTIONS): 
        # generate and evaluate captions for each combination of decoding settings 
        beam_search = True if decoding=="beam_search" else False
        # results are saved to output folder 
        test_df, scores = test(test_df, model=model, tokenizer= tokenizer, beam_search=beam_search, pos=RSA_OPTIONS[rsa]["pos"], t=RSA_OPTIONS[rsa]["t"], 
                               target_dir=target_dir + "{}/{}/".format(decoding, rsa), 
                               image_path=image_path,
                               cname="caps_{}_{}".format(decoding, rsa), presave=True, 
                               results_file="test_{}_{}.json".format(decoding, rsa), 
                               results_csv="test_{}_{}.csv".format(decoding, rsa), 
                               score_file = "test_scores_{}_{}.txt".format(decoding, rsa), k=3, group_by="scene_idx")


        
if __name__=="__main__":
    current_folder = dirname(abspath(__file__))
    parent_folder = dirname(current_folder)
    image_path=current_folder + "/data/AbstractScenes_v1.1/RenderedScenes/"
    save_path = os.path.join(parent_folder, "pretrained_models")
    model_path = os.path.join(save_path, 'conceptual_weights.pt')

    generation_model = gen.load_clip_model(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    test_file = current_folder + "/data/AbstractScenes_v1.1/processed_data/test_df_r3.csv"
    target_dir = current_folder + "/output/"
    main(test_file, target_dir, model=generation_model, tokenizer=tokenizer, image_path=image_path)


    

