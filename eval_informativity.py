## EVALUATE INFORMATIVITY USING Leval MODEL ##

import torch
import os
from generate import get_logits
from generate import preproc_img
from generate import load_clip_model
from generate import download_model
from generate import get_device
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
CUDA = get_device
device = CUDA(0) 

current_folder = os.path.dirname( os.path.abspath(__file__))
parent_folder =  os.path.dirname(current_folder)
save_path = os.path.join(parent_folder, "pretrained_models")
# use clipcap model trained on COCO
eval_model_path = os.path.join(save_path, 'coco_weights.pt')
download_eval_model=False

if download_eval_model:
    download_model("COCO", model_path=eval_model_path)

eval_model = load_clip_model(eval_model_path)

def eval_informativity(caption, target_file, all_files, temperature=1, image_path = "./data/AbstractScenes_v1.1/RenderedScenes/", k=10):

    cap_tok = tokenizer.encode(caption)
    token_ps = []
    p = torch.tensor([1]*k)
    # get embeds for all files using clip
    embeds = [preproc_img(image_path + img, proj_model=eval_model) for img in all_files]
    generated = embeds
    for tok in cap_tok:
        # get softmax distribution given each picture
        logits = [torch.softmax(get_logits(gen, temperature, model=eval_model), dim=-1) for gen in generated]
        # get tensor of probabilities of current token given each picture 
        p_tok = torch.tensor([logit[0].detach()[tok] for logit in logits])
        # multiply probability of current token for each picture with probability so far 
        p = p * p_tok
        token_ps.append(p_tok)
        # get embedding for current token 
        next_token_embed = eval_model.gpt.transformer.wte(torch.tensor(tok).to(device).unsqueeze(dim=0))
        # append current token embedding to input tensor for each picture
        generated = [torch.cat((gen.squeeze(dim=0), next_token_embed)).unsqueeze(dim=0) for gen in generated]

    # divide probabilities of sequence given each picture by sum of all probabilities     
    p_norm = p/torch.sum(p)
    # select picture with the highest sequence probability 
    top_pic = all_files[torch.argmax(p_norm)]
    # check if prediction is true 
    pred_true = int(top_pic==target_file)
    return(p_norm, top_pic, pred_true) 

    