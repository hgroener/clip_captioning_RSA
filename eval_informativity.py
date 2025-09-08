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
eval_model_path = os.path.join(save_path, 'coco_weights.pt')
download_eval_model=False

if download_eval_model:
    download_model("COCO", model_path=eval_model_path)

eval_model = load_clip_model(eval_model_path)

def eval_informativity(caption, target_file, all_files, temperature=1, image_path = "./data/AbstractScenes_v1.1/RenderedScenes/", k=10):

    cap_tok = tokenizer.encode(caption)
    #print("cap_tok:", cap_tok)
    token_ps = []
    p = torch.tensor([1]*k)
    #target_embed = preproc_img(image_path + file, d=True)
    embeds = [preproc_img(image_path + img, d=False) for img in all_files]
    generated = embeds
    for tok in cap_tok:
        logits = [torch.softmax(get_logits(gen, temperature, model=eval_model), dim=-1) for gen in generated]

        #print(tok)
        p_tok = torch.tensor([logit[0].detach()[tok] for logit in logits])
        #p = [prob * tok_prob for prob, tok_prob in zip(p,p_tok)]

        p = p * p_tok
        token_ps.append(p_tok)

        next_token_embed = eval_model.gpt.transformer.wte(torch.tensor(tok).to(device).unsqueeze(dim=0))
        #print("next token embed size:", next_token_embed.size())

        generated = [torch.cat((gen.squeeze(dim=0), next_token_embed)).unsqueeze(dim=0) for gen in generated]
        
    p_norm = p/torch.sum(p)
    #print("p_norm:", p_norm)
    #print(list(scene["file"]))
    top_pic = all_files[torch.argmax(p_norm)]
    pred_true = int(top_pic==target_file)
    return(p_norm, top_pic, pred_true) #probabilities for each picture, chosen picture

    