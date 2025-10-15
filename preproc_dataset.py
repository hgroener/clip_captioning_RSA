## PREPROCESS ABSTRACT SCENES DATASET ## 

import re 
from tqdm import tqdm
import pandas as pd 
import random as rd 
import os 

sents_files = ["./data/AbstractScenes_v1.1/SimpleSentences/SimpleSentences1_10020.txt", 
               "./data/AbstractScenes_v1.1/SimpleSentences/SimpleSentences2_10020.txt"]

# get pandas dataframe from txt files 
def get_cap_df(sents_files, output_path="./data/AbstractScenes_v1.1/AS_caps.csv"):
    lines = []
    i = 0
    for file in sents_files:
        with open(file, "r") as f:
            for line in f.readlines():
                # select numbered lines
                if re.match("^[0-9]", line):
                    # split by tabulators 
                    line_idx, sent_idx, sent = line.split("\t")
                    # delete newlines inside sentences
                    sent = re.sub("( )\\n", "", sent)
                    # get scene IDs and picture IDs 
                    if len(line_idx)>1: 
                        scene_idx, pic_idx = line_idx[:-1], line_idx[-1]
                    else:
                        scene_idx, pic_idx = 0, line_idx

                    # add data to dictionary list with unique caption identifier
                    lines.append({"cap_idx": i, "scene_idx": scene_idx, "pic_idx": pic_idx, 
                                "file": "Scene{}_{}.png".format(scene_idx, pic_idx),"caption": sent})
                    i += 1
    # create dataframe
    df = pd.DataFrame.from_dict(lines)
    if output_path: # save to .csv
         df.to_csv(output_path)
    return(df)


def get_pic_df(cap_df, output_path="./data/AbstractScenes_v1.1/imagewise_df.csv"):
    scenes = list(set(cap_df["scene_idx"]))
    pic_dics = []
    for scene in tqdm(scenes): 
        # get subset of rows belonging to current scene 
        df_scene = cap_df[cap_df["scene_idx"]==scene]
        # iterate over pictures in scene 
        for pic_idx in list(set(df_scene["pic_idx"])):
            df_pic = df_scene[df_scene["pic_idx"]==pic_idx]
            # get dictionary of captions beloning to picture 
            caps = {"cap" + str(i): cap for i, cap in enumerate(list(df_pic["caption"])[:6])}
            file = list(df_pic["file"])[0]
            dic ={"scene_idx": scene, "pic_idx": pic_idx, "file": file}
            dic.update(caps)
            # create dictionary list with all captions of one picture in one dictionary
            pic_dics.append(dic)
    
    # create dataframe 
    pic_df = pd.DataFrame.from_dict(pic_dics)
    if output_path:
        pic_df.to_csv(output_path)

    return(pic_df)

# substitute names in dataset with generic descriptions 
def substitute_names(pic_df, output_path="./data/AbstractScenes_v1.1/imagewise_df_nn.csv"):
    for c in range(6):
        col = "cap"+str(c)
        caps = list(pic_df[col])
        new_caps=[]
        for cap in caps:
            if type(cap)==str:
                cap_new = re.sub("^(M|m)ike", "The boy", cap)
                cap_new = re.sub("^(J|j)enny", "The girl", cap_new)
                cap_new = re.sub("(M|m)ike", "the boy", cap_new)
                cap_new = re.sub("(J|j)enny", "the girl", cap_new)
                new_caps.append(cap_new)
            else:
                print("Error: caption: {} is of type {}".format(cap, type(cap)))
                new_caps.append("")
        pic_df[col] = new_caps
    if output_path:
        pic_df.to_csv(output_path)
    return(pic_df)


def reduce_pics(df, k=3, output_path="./data/AbstractScenes_v1.1/processed_data/test_df_3samples.csv"):
    # reduce scenes to k randomly selected pictures 
    files = []
    scenes = list(set(df["scene_idx"]))
    for scene in scenes: 
        scene_df = df[df["scene_idx"]==scene]
        sample_pics = rd.sample(list(scene_df["pic_idx"]), k=3)
        sample_files = list(scene_df[scene_df["pic_idx"].isin(sample_pics)]["file"])
        files += sample_files
    reduced_df = df[df["file"].isin(files)]
    if output_path:
        reduced_df.to_csv(output_path)
    return(reduced_df)


# instead of grouping pictures by scenes, group randomly (not used in finished project)
def get_random_triplets(df, output_path="./data/AbstractScenes_v1.1/processed_data/test_df_random_triplets.csv", k=1000, only_triplets = True):
    pics = list(zip(list(df["file"]), list(df["scene_idx"])))
    triplets = []
    i = 0
    while i < k: 
        triplet = rd.sample(pics, 3)
        pics = [pic for pic in pics if not pic[0] in [p[0] for p in triplet]]
        #print("pics num:", len(pics))
        if len(triplet) == len(set(triplet)):
            triplets.append(triplet)
            i += 1 
    
    for n, triplet in enumerate(triplets): 
        df.loc[df["file"].isin([t[0] for t in triplet]), "triplet_idx"] = n   
    if only_triplets:
        mask = pd.notna(df['triplet_idx'])
        df = df[mask]
    if output_path:
        df.to_csv(output_path) 
    return(df)

# split in train and test set 
def train_test_split(df, train_size=250, train_file=None, test_file=None):
    train = rd.sample(list(df["scene_idx"]), k=train_size)
    train_df = df[df["scene_idx"].isin(train)]
    test_df = df[~df["scene_idx"].isin(train)]
    tt_folder = "./data/AbstractScenes_v1.1/processed_data/"
    if not os.path.exists(tt_folder):
        os.makedirs(tt_folder)
    if train_file:
        train_path = tt_folder + "train_df.csv"
        train_df.to_csv(train_path)
        print("train df saved to {}.".format(train_path))
    if test_file: 
        test_path = tt_folder + "test_df.csv"
        test_df.to_csv(test_path)
        print("test df saved to {}.".format(test_path))

    return(train_file, test_file)

# preprocess data and save output path 
def main(no_names = True, r3=True, random_triplets = False, output_path = "./data/AbstractScenes_v1.1/processed_data/"):
    df = get_cap_df(sents_files)
    pic_df = get_pic_df(df)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if no_names:
        pic_df = substitute_names(pic_df, output_path=output_path+"as_nn.csv")
    if r3:
        pic_df = reduce_pics(pic_df, output_path=output_path + "as_r3.csv")
    if random_triplets:
        pic_df = get_random_triplets(pic_df, output_path=output_path+"as_triplets.csv")
    train_df, test_df = train_test_split(pic_df, train_file=output_path+"train.csv", test_file=output_path+"test.csv")


if __name__=="__main__":
    print("preprocessing...")
    main()


