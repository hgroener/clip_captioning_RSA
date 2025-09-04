
import pandas as pd
import os 
from tqdm import tqdm
import itertools
from transformers import GPT2Tokenizer

import generate as gen
from eval_informativity import eval_informativity
from cider.cidereval import eval_ciderGPT2Tokenizer

save_path = os.path.join(os.path.dirname(os.getcwd()), "pretrained_models")
model_path = os.path.join(save_path, 'conceptual_weights.pt')

generation_model = gen.load_clip_model(model_path)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

RSA_options = ["greedy", "RSA", "RSA_t", "RSA_POS"]
decoding_options = ["greedy", "beam_search"]


RSA_dic = {"greedy": {"t": 0, "cname": "greedy_caps", "func": "RSA_t"}, 
           "RSA": {"t": 1, "cname": "RSA_caps", "func": "RSA_t"}, 
           "RSA_t": {"t": 0.5, "cname": "RSA_t_caps", "func": "RSA_t"},
           "RSA_POS": {"t": 1, "cname": "RSA_POS_caps", "func": "RSA_POS"}}


## TESTING
def test(df, t=1, pos = False, beam_search = False, a=3, top_k=100, temp=0.8, target_dir="./data/AbstractScenes_v1.1/testing/RSA_r3/", cname="RSA_caps", results_file="results.json",
        results_csv="RSA_caps.csv", score_file = "scores.txt", k=10, presave=True, group_by = "scene_idx", generate=True):
    os.makedirs(target_dir, exist_ok = True)
    print("generating captions with temperature={}, t={}, a={}".format(temp, t, a))
    if generate:
        if not beam_search: 
            if not pos: 
                gen.generate_RSA_pos(generation_model, tokenizer, df)
        if func == "RSA_t": 
            caps, _ = gen.generate_RSA_t(generation_model, tokenizer, df, temperature = temp, t=t, a=a, top_k=top_k, group_by=group_by)
        elif func == "RSA_POS":

        elif func == "beam":
            caps = gen.generate_beam_RSA(df, temperature = temp, t=t, a=a, )

        print("captions generated, ")
        for cap in caps: 
            df.loc[df["file"]==cap[0], cname] = cap[1] 

        if presave: 
            df.to_csv(target_dir + results_csv)
    else:
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
        informative = [int(eval_informativity(cap, files[i], files, k=k)[2]) for i, cap in enumerate(caps)]
        df.loc[df[group_by]==scene, 'informative'] = informative

    df.to_csv(results_csv)
    mean_informative = sum(list(df['informative']))/len(df)
    
    with open(target_dir + score_file, "w") as f:
        f.write("cider: {}\ninformative: {}".format(mean_cider, mean_informative))

    return(df, (mean_cider, mean_informative))


if __name__=="__main__":
    test_df = pd.read_csv("other/clip_captioning_RSA/data/AbstractScenes_v1.1/processed_data/test_df_r3.csv")
    for rsa, decoding in itertools.product(list(RSA_dic.keys()), decoding_options):
        if decoding=="beam_search":
            test_df, scores = test(test_df, func="beam", t=1, target_dir="./data/AbstractScenes_v1.1/testing/", cname="caps_RSA", presave=True,
                                   results_file="results_test_RSA_random_triplets.json", results_csv="test_RSA_random_triplets.csv", 
                                   score_file = "test_scores_RSA_random_triplets.txt", k=3, group_by="triplet_idx")
        else: 


    test_df_r3 = pd.read_csv("./data/AbstractScenes_v1.1/processed_data/test_df_r3.csv")
     


    test_df, scores = test(df_test_triplets, t=1, target_dir="./data/AbstractScenes_v1.1/testing/", cname="actual_caps_RSA", presave=True,
                        results_file="results_test_actual_RSA_random_triplets.json", results_csv="test_actual_RSA_random_triplets.csv", 
                        score_file = "test_scores_actual_RSA_random_triplets.txt", k=3, group_by="triplet_idx")

    test_df, scores = test(reduced_df, t=1, target_dir="./data/AbstractScenes_v1.1/testing/", cname="RSA_caps", presave=True,
                        results_file="results_test_RSA_r3.json", results_csv="test_RSA_r3.csv", 
                        score_file = "test_scores_RSA_r3.txt", k=3)

    print(scores)

