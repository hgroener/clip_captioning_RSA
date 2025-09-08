# CLIP Captioning mit RSA-basiertem pragmatischem Decoding 
## von Hannes Gröner 
### Code abgewandelt von https://github.com/rmokady/CLIP_prefix_caption/tree/main/notebooks
#@title Imports
import pandas as pd
import clip
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from tqdm.notebook import tqdm, trange
#from google.colab import files
import skimage.io as io
import PIL.Image
from IPython.display import Image 
import matplotlib.pyplot as plt

import itertools
#import functools
import random as rd
import gdown
import json
import nltk
import re 
#from wordfreq import word_frequency
#from sklearn.model_selection import train_test_split 
import random as rd
import itertools 

from cider.cidereval import eval_cider

#nltk.download()
content_tags = ["JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "RB" , "RBR", "RBS", "VB", 
                "VBD", "VBG", "VBN", "VBP", "VBZ"]

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]


D = torch.device
CPU = torch.device('cpu')


def get_device(device_id: int) -> D:
    if not torch.cuda.is_available():
        return CPU
    device_id = min(torch.cuda.device_count() - 1, device_id)
    return torch.device(f'cuda:{device_id}')


CUDA = get_device

current_directory = os.getcwd()
save_path = os.path.join(os.path.dirname(current_directory), "pretrained_models")
os.makedirs(save_path, exist_ok=True)
model_path = os.path.join(save_path, 'conceptual_weights.pt')


#@title Model

class MLP(nn.Module):

    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) -1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class ClipCaptionModel(nn.Module):

    #@functools.lru_cache #FIXME
    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        #print(embedding_text.size()) #torch.Size([5, 67, 768])
        #print(prefix_projections.size()) #torch.Size([5, 1, 768])
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)
        else:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length))


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self
    

#@title GPU/CPU
is_gpu = True #@param {type:"boolean"}  
#@title CLIP model + GPT2 tokenizer

device = CUDA(0) if is_gpu else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#@title Choose pretrained model - COCO or Coneptual captions

download_generation_model = False

def download_model(name, model_path = model_path): 
    if name == 'Conceptual captions':
        id = "14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT"
    elif name == "COCO":
        id = "14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT"
    else:
        print("unknown model type!")
        return
        
    gdown.download(id=id, output=model_path, quiet=False)
    
if download_generation_model:
    download_model("Conceptual captions", model_path=model_path)


def load_clip_model(model_path, prefix_length=10):
    model = ClipCaptionModel(prefix_length)
    model.load_state_dict(torch.load(model_path, map_location=CPU)) 
    model = model.eval() 
    device = CUDA(0) if is_gpu else "cpu"
    model = model.to(device)
    return(model)

generation_model = load_clip_model(model_path)

## generation
# get clip prefix embedding from image path 
def preproc_img(path, proj_model=generation_model, prefix_length = 10, d=False, resize_tuna=False):
    # read image using scikit-image
    img = io.imread(path)
    # convert to PIL format 
    pil_img = PIL.Image.fromarray(img)
    if resize_tuna:
        pil_img=resize(pil_img, max_w, max_h)
    # image is printed based on boolean value of d 
    if d:
        display(pil_img)
    # preprocess image using CLIP 
    pil_img = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        # extract CLIP prefix embedding
        prefix = clip_model.encode_image(pil_img).to(device, dtype=torch.float32)
        prefix_embed = proj_model.clip_project(prefix).reshape(1, prefix_length, -1)
    return prefix_embed


def get_logits(embeds, temperature, model=model):
    #print("embeds size: ", embeds.size())
    outputs = model.gpt(inputs_embeds=embeds)
    logits = outputs.logits
    #print("logits before temperature: ", logits.size())
    logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
    #print("logits after temperature: ", logits.size())
    return(logits)


def generate_RSA(
        model,
        tokenizer,
        df,
        entry_length=30,  # maximum number of words
        top_k=100,
        temperature=1.,
        stop_token: str = '.',
        a = 1,
        pos_decoding = False,
        print_logits = False,
        t = 0.5,
        image_path = "./data/AbstractScenes_v1.1/RenderedScenes/",
        d = False,
        group_by = "scene_idx"

):
    model.eval()
    generated_num = 0
    generated_list = []
    rsa_stop_list = []
    no_rsa_pos_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    #filter_value = 0
    device = next(model.parameters()).device
    
    
    with torch.no_grad():
        scenes = [scene for scene in list(df[group_by]) if not scene in ["None", None]]
        for scene in tqdm(list(set(scenes))):
            
            df_scene = df[df[group_by]==scene]
            for file in df_scene["file"]:
                
                tokens = None
                l0_prob_target = 0 
                RSA = True
                pos = True
                rsa_stop = None
                no_rsa_pos = []

                #dnames = ["d_" + str(num) for num in range(6)]
                target_embed = preproc_img(image_path + file, d=d, proj_model=model)
                distr_embeds = [preproc_img(image_path + img, d=d, proj_model=model) for img in list(df_scene["file"]) if not img==file]

                if not None in [target_embed] + distr_embeds:
                    # one embedding per image (1 target, 2 distractors)
                    generated_target = target_embed
                    generated_distr = distr_embeds
                else:
                    #print("Embedding missing, generation stopped!")
                    break
                image_priors = [1/(len(distr_embeds)+1) for embed in range(len(distr_embeds)+1)]
                for i in range(entry_length):
                    # get output logits for each image from gpt model
                    logits_target = get_logits(generated_target, temperature)
                    distr_logits = [get_logits(distr, temperature) for distr in generated_distr]

                    # sort token indices based on logit value 
                    _, sorted_indices_target = torch.sort(logits_target, descending=True)
                    # get top-k indices

                    top_indices = sorted_indices_target.squeeze(0)[:top_k]
                    #print("top_indices:", top_indices)

                    # set values of other tokens to - infinity 
                    top_logits_target = logits_target[:, top_indices] 
                    distr_top_logits = [logits[:, top_indices] for logits in distr_logits]

                    # apply softmax function to remaining logits
                    probabilities_target = nnf.softmax(top_logits_target, dim=-1)             
                    distr_probs = [nnf.softmax(logits, dim=-1) for logits in distr_top_logits]
                    if l0_prob_target < t: 
                        if pos_decoding: 
                            pos = get_pos(logits_target, probabilities_target)
                        if pos: 
                            # pragmatic reasoning 
                            s0_probs = [probabilities_target] + distr_probs
                            #print("s0_probs size:", s0_probs[0].size())

                            l0_probs_no_n = [prob * image_priors[i] for i, prob in enumerate(s0_probs)]

                            l0_sum = sum(l0_probs_no_n)

                            l0_probs = [prob/l0_sum for prob in l0_probs_no_n]
                            u = [torch.log(prob) for prob in l0_probs]

                            #print("l0_probs size:", l0_probs[0].size())
                            s1_probs_no_n = [torch.exp(a*utility - -torch.log(s0_probs[i])) for i, utility in enumerate(u)]
                            s1_sum = torch.tensor([torch.sum(w) for w in s1_probs_no_n])
                            s1_probs = [s1_prob/s1_sum[i] for i, s1_prob in enumerate(s1_probs_no_n)]

                            target_probs = s1_probs[0]

                            # apply argmax function to target probabilities to extract the most probable next word 
                            next_token = torch.argmax(target_probs, -1).unsqueeze(0)
                            #print("l0_probs", l0_probs)
                            l0_prob_target = l0_probs[0][0][next_token]


                            if print_logits: 
                                print("s0_probs", s0_probs)
                                print("l0_probs before normalization", l0_probs_no_n)
                                print("l0_sum", l0_sum)
                                print("l0_probs after normalization", l0_probs)
                                print("U", u)
                                print("s1_probs before normalization", s1_probs_no_n)
                                print("s1_sum", s1_sum)
                                print("s1_probs after normalization", s1_probs)

                                print("next token index:", next_token)
                        else:
                            no_rsa_pos.append(i)
                            next_token = torch.argmax(probabilities_target, -1).unsqueeze(0)
                    else:
                        if RSA:
                            if t > 0:
                                if not tokens==None:
                                    if d:
                                        print("stopped RSA decoding after", tokenizer.decode(list(tokens.squeeze().cpu().numpy())))
                                else:
                                    if d: 
                                        print("stopped RSA decoding immediately.")
                                
                            rsa_stop = i 
                            RSA=False
                        # greedy decoding
                        next_token = torch.argmax(probabilities_target, -1).unsqueeze(0)

                    # extract embeddings for output word 
                    next_token_index = top_indices[next_token]
                    next_token_embed = model.gpt.transformer.wte(next_token_index)
                    if tokens == None:
                        tokens = next_token_index
                    else:
                        # add token to output
                        tokens = torch.cat((tokens, next_token_index), dim=1)
                    # add next token embedding to image embeddings for each image
                    generated_target = torch.cat((generated_target, next_token_embed), dim=1)
                    generated_distr = [torch.cat((gen, next_token_embed), dim=1) for gen in generated_distr]
                    # stop generating when the stop token is generated
                    if stop_token_index == next_token_index.item():
                        break
                # get list of generated output sequence
                output_list = list(tokens.squeeze().cpu().numpy())
                # convert token IDs to tokens 
                output_text = tokenizer.decode(output_list)
                # append output to list of output sentences
                generated_list.append((file, output_text))
                rsa_stop_list.append(rsa_stop)
                no_rsa_pos_list.append(no_rsa_pos)

    return (generated_list, rsa_stop_list, no_rsa_pos_list)
)


def rearrange_RSA(logits, distr_logits, image_priors, l0_probs_tokens = 0, top_k=100, t = 1, a=3, i = 0, RSA=True, pos = False, print_stops=False):

    # sort token indices based on logit value 
    _, sorted_indices_target = torch.sort(logits, descending=True)
    # get top-k indices
    top_indices = sorted_indices_target.squeeze(0)[:top_k]

    
    #print("logits size in function:", logits.shape)
    topk_scores, topk_tokens = logits.topk(top_k, -1)

    #print("topk_tokens:", topk_tokens.shape)
    top_logits_target = logits.gather(1, topk_tokens)

    pos_no_rsa = None 
    if pos:
        top_beam_words = [[re.sub("^ ", "", tokenizer.decode(ind)) for ind in beam] for beam in topk_tokens]
        top_beam_words_tagged = [nltk.pos_tag(top_words) for top_words in top_beam_words]
        beam_tags = [[pos for w, pos in top_words_tagged] for top_words_tagged in top_beam_words_tagged]

        c_tag_probs = 0
        nc_tag_probs = 0
        for beam_num, beam in enumerate(beam_tags): 
            for tc, tag in enumerate(beam): 
                if tag in content_tags: 
                    c_tag_probs += topk_scores[beam_num][tc]
                else:
                    nc_tag_probs += topk_scores[beam_num][tc]
        
        #print("c_tag_probs: {}\nnc_tag_probs: {}".format(c_tag_probs, nc_tag_probs))
        if c_tag_probs < nc_tag_probs: 
            RSA = False
            pos_no_rsa = i
            
                    
    #print("size top_logits_target:", top_logits_target.shape)
    distr_top_logits = [lg[:, topk_tokens].squeeze(0) for lg in distr_logits]
    lookup_l0 = torch.zeros(topk_tokens.shape[0],50257).to(device)
    # print("topk_tokens: {}\n topk_scores: {}\n, top_logits_target: {}".format(topk_tokens.shape, topk_scores.shape, top_logits_target.shape))
    #print("maximum of l0_probs:", torch.max(torch.max(l0_probs_tokens, -1)[0]))
    #print("size of l0_probs_tokens:",  l0_probs_tokens.shape)
    top_l0_prob = torch.max(l0_probs_tokens)
    #print("top_l0_prob:", top_l0_prob)
    #print("top_l0_prob size:", top_l0_prob.shape)
    #print("top_l0_prob:", top_l0_prob)
    rsa_stop = None

    if RSA and (top_l0_prob.item() < t): 
        # pragmatic reasoning 
        s0_probs = torch.stack([top_logits_target] + distr_top_logits)
        #print("s0_probs:", s0_probs)
        #print("s0_probs size:", s0_probs.size())
        l0_probs_no_n = s0_probs * image_priors[:,None,None]

        #print("l0_probs_no_n size:", l0_probs_no_n.shape)
        l0_sum = torch.sum(l0_probs_no_n, 2)
        #print("size l0_sum:", l0_sum.shape)
        l0_probs = torch.div(l0_probs_no_n,l0_sum[:,:,None])
        #print("l0_probs size:", l0_probs.shape)
        u = torch.log(l0_probs)#[torch.log(prob) for prob in l0_probs]
        #print("u:", u.shape)

        #print("l0_probs size:", l0_probs[0].size())
        #print("-torch.log(s0_probs[i]):", [-torch.log(s0_probs[i]) for i, ut in enumerate(u)])
        s1_probs_no_n = torch.exp(a*u - -torch.log(s0_probs))
        #print(s1_probs_no_n.shape)
        #print("s1_probs before normalization:", s1_probs_no_n)
        s1_sum = torch.sum(s1_probs_no_n, 2)
        #print("s1_sum size:", s1_sum.shape)
        s1_probs = torch.div(s1_probs_no_n, s1_sum[:,:,None])
        #print("s1_probs:", s1_probs)

        top_logits_target = s1_probs[0]
        #print("top_logits_target size:", top_logits_target.shape)
        #print("topk_tokens size:", topk_tokens.shape)

        l0_probs_target = l0_probs[0]#[next_tokens]
        #print("size l0_probs_target:", l0_probs_target.shape)
        #print("l0_probs_target size:", l0_probs_target.shape)
        
        #print("lookup_l0 size:", lookup_l0.shape)
        for dim in range(lookup_l0.shape[0]):
            lookup_l0[dim][topk_tokens[dim]] = l0_probs_target[dim]

    elif pos==False:
        if RSA == True:
            if print_stops:
                print("stopped RSA decoding after {} words with highest l0 probability = {}".format(i, top_l0_prob))
            RSA = False
            rsa_stop = i 
    
    else:
        if print_stops:
            print("RSA disabled at index {} due to POS decoding".format(i))
        # set RSA to True again to keep checking for POS at next step
        RSA = True
        
    lookup = torch.zeros(topk_tokens.shape[0], 50257).to(device)
    #print("topk_tokens size: {} \n top_logits_target size: {}".format(topk_tokens.shape, top_logits_target.shape))
    for dim in range(lookup.shape[0]):
        lookup[dim][topk_tokens[dim]] = top_logits_target[dim]
    #("lookup size:", lookup.shape)

    return(lookup, lookup_l0, RSA, rsa_stop, pos_no_rsa)



def generate_beam_RSA(df, entry_length=67, temperature = 1, stop_token = ".", top_k = 100, t= 1, a=3, beam_size=5, 
                      image_path = "./data/AbstractScenes_v1.1/RenderedScenes/", d=False, pos=False, print_stops = False):
    generation_model.eval()
    generated_num = 0
    generated_list = []
    rsa_stop_list = []
    pos_no_rsa_scene = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    #filter_value = 0
    device = next(generation_model.parameters()).device
    pos_no_rsa_sum = 0
    length_sum = 0
        
    for scene in tqdm(list(set(df["scene_idx"]))):
        
        df_scene = df[df["scene_idx"]==scene]
        for file in df_scene["file"]:
            pos_no_rsa_list = []
            tokens = None
            RSA = True

            #dnames = ["d_" + str(num) for num in range(6)]
            embed = preproc_img(image_path + file, d=d, proj_model=generation_model)
            distr_embeds = [preproc_img(image_path + img, d=d, proj_model=generation_model) for img in list(df_scene["file"]) if not img==file]
            image_num = len(distr_embeds) + 1 
            image_priors =  torch.tensor([1/image_num]*image_num).to(device)
            l0_probs_tokens = torch.tensor([[0]*5]).to(device)
            generation_model.eval()
            stop_token_index = tokenizer.encode(stop_token)[0]
            scores = None
            device = next(generation_model.parameters()).device
            seq_lengths = torch.ones(beam_size, device=device)
            is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
            pos_no_rsa_sent = []
            #print("seq_lenghths {}\n is stopped {}".format(seq_lengths, is_stopped))
            with torch.no_grad():
                if embed is not None:
                    generated = embed
                    generated_distr = distr_embeds
                else:
                    print("please input embeds!")
                for i in range(entry_length):
                    logits = get_logits(generated, temperature=temperature)
                    #print("logits size:", logits.shape)
                    #outputs = model.gpt(inputs_embeds=generated)
                    #distr_outputs = (model.gpt(input_embeds=distr) for distr in generated_distr)
                    #logits = outputs.logits
                    distr_logits = [get_logits(embed, temperature=temperature) for embed in generated_distr]
                    #logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                    #logits = logits.softmax(-1).log()
                    logits = logits.softmax(-1)
                    #print("logits size after softmax:", logits.shape)
                    #distr_logits = [lg.softmax(-1).log() for lg in distr_logits]
                    distr_logits = [lg.softmax(-1) for lg in distr_logits]
                    if scores is None:
                        top_logits_target, l0_probs_tokens, RSA, rsa_stop, pos_no_rsa = rearrange_RSA(logits, distr_logits, image_priors, 
                                                                                                      l0_probs_tokens=l0_probs_tokens, top_k=top_k, 
                                                                                                      t=t, i =i, RSA=RSA, pos=pos, 
                                                                                                      print_stops=print_stops)
                        scores, next_tokens = top_logits_target.topk(beam_size, -1)
                        #print("l0_probs_tokens size: {}\nnext_tokens_size: {}".format(l0_probs_tokens.shape, next_tokens.shape))
                        l0_probs_tokens = l0_probs_tokens.gather(1, next_tokens)
                        #print("new l0_probs_tokens size:", l0_probs_tokens.shape)
                        scores = scores.squeeze(0)
                        #print("scores size {}\n next_tokens size {}".format(scores.shape, next_tokens.shape))
                        generated = generated.expand(beam_size, *generated.shape[1:])
                        #print("size of ggitenerated after expansion: ", generated.shape)
                        next_tokens = next_tokens.permute(1, 0)
                        #print("size of next_tokens after permute:", next_tokens.shape)
                        #next_tokens, scores = next_tokens.squeeze(0), scores.squeeze(0)
                        if tokens is None:
                            tokens = next_tokens
                        else:
                            tokens = tokens.expand(beam_size, *tokens.shape[1:])
                            tokens = torch.cat((tokens, next_tokens), dim=1)
                            #print("tokens after concatenation: ", tokens) 
                    else:



                        #topk_scores, topk_tokens = logits.topk(top_k, -1)
                        #top_logits_target = logits[:, topk_tokens].squeeze(0).squeeze(0)
                        #distr_top_logits = [lg[:, topk_tokens].squeeze(0).squeeze(0) for lg in distr_logits]
                        top_logits_target, l0_probs_tokens, RSA, rsa_stop, pos_no_rsa = rearrange_RSA(logits, distr_logits, image_priors, 
                                                                                                      l0_probs_tokens=l0_probs_tokens, top_k=top_k, t=t, 
                                                                                                      i=i, RSA=RSA, pos=pos, print_stops=print_stops)
                        top_logits_target[is_stopped] = -float(np.inf)
                        top_logits_target[is_stopped, 0] = 0
                        #scores, next_tokens = top_logits_target.topk(beam_size, -1)
                        #print(scores.shape, top_logits_target.shape)
                        #print("scores:", scores)
                        #scores_sum = torch.add(scores[0][:,None], top_logits_target)
                        scores_sum = scores[:, None] + top_logits_target
                        #print("scores_sum:", scores_sum)
                        seq_lengths[~is_stopped] += 1
                        #print("seq_lengths after adding 1", seq_lengths)
                        scores_sum_average = scores_sum / seq_lengths[:, None]
                        scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                        #print("next_tokens:", next_tokens)
                        #print("next_tokens size:", next_tokens.shape)
                        
                        #print("scores_sum_average {}\n next_tokens after view operation {}".format(scores_sum_average, next_tokens))
                        next_tokens_source = next_tokens // scores_sum.shape[1]
                        
                        seq_lengths = seq_lengths[next_tokens_source]
                        next_tokens = next_tokens % scores_sum.shape[1]
                    
                        #print("scores_sum.shape[1]", scores_sum.shape[1])
                        next_tokens = next_tokens.unsqueeze(1)
                        #print("next_tokens:", next_tokens)
                        l0_probs_tokens = l0_probs_tokens.gather(1, next_tokens)
                        #print("next_tokens_source", next_tokens_source)
                        tokens = tokens[next_tokens_source]
                        tokens = torch.cat((tokens, next_tokens), dim=1)
                        #print("tokens after adding next_tokens", tokens)
                        generated = generated[next_tokens_source]
                        #print("size generated: ", generated.shape)
                        scores = scores_sum_average * seq_lengths
                        
                        #print("scores after multiplying by seq_lenghts", scores)
                        is_stopped = is_stopped[next_tokens_source]
                        #print("new is_stopped:", is_stopped)
                    next_token_embed = generation_model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
                    generated = torch.cat((generated, next_token_embed), dim=1)
                    #print("size of generated after adding next_token_embed", generated.shape)
                    is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
                    #print("is_stopped after scanning for stop_token_index", is_stopped)
                    if rsa_stop !=None:
                        rsa_stop_list.append(rsa_stop)
                    if pos_no_rsa != None:
                        pos_no_rsa_sent.append(pos_no_rsa)
                    if is_stopped.all():
                        break
                pos_no_rsa_list.append(pos_no_rsa_sent)
                
            scores = scores / seq_lengths
            #print("final scores:", scores)
            output_list = tokens.cpu().numpy()
            output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
            order = scores.argsort(descending=True)
            #print("order:", order)
            output_text = [output_texts[i] for i in order][0]
            # add seq_lenght of chosen output to overall seq lengths of outputs to calculate proportion of tokens that were generated using 
            # RSA decoding later
            seq_length  =seq_lengths[order[0]]
            length_sum += seq_length
            
            generated_list.append(output_text)
            
            pos_no_rsa_output = pos_no_rsa_list[order[0]]
            pos_no_rsa_len = len(pos_no_rsa_output)
            # add number of tokens that were generated using RSA decoding to overall number 
            pos_no_rsa_sum += pos_no_rsa_len
            
            pos_no_rsa_scene.append(pos_no_rsa_output)
            
        pos_no_rsa_average = pos_no_rsa_sum/length_sum
        no_rsa_t_average = sum(rsa_stop_list)/length_sum
        
    return(generated_list, rsa_stop_list, pos_no_rsa_scene, no_rsa_t_average, pos_no_rsa_average)



def get_pos(logits_target, probabilities_target, top_k=100):
    _, sorted_indices_target = torch.sort(logits_target, descending=True)
    # get top-k indices
    #top_indices = sorted_indices_target.squeeze(0)[top_k+1:]
    top_indices = sorted_indices_target.squeeze(0)[:top_k]
    top_words = [re.sub("^ ", "", tokenizer.decode(ind)) for ind in top_indices]
    top_words_tagged = nltk.pos_tag(top_words)
    tags = [pos for w, pos in top_words_tagged]

    c_tag_probs = 0
    nc_tag_probs = 0
    for tc, tag in enumerate(tags): 
        if tag in content_tags: 
            c_tag_probs += probabilities_target[0][tc]
        else:
            nc_tag_probs += probabilities_target[0][tc]

    return(c_tag_probs > nc_tag_probs)


# select decoding method (beam search, greedy search or RSA-based search), including temperature parameter and 
# rationality parameter a (only for RSA-based search)
def generate_method(model, tokenizer, embeds, method="RSA", temperature=0.9, a=0.05, t=0.5, top_k = 100):
    if method=="beam_search":
        generated_text_prefix = generate_beam(model, tokenizer, embed=embeds[0], temperature=temperature)[0]
    elif method=="greedy":
        generated_text_prefix = generate2(model, tokenizer, embed=embeds[0], temperature=temperature)
    elif method=="nie":
        generated_text_prefix = generate_RSA_nie(model, tokenizer, target_embed=embeds[0], distr_embeds = embeds[1:],
                                                 temperature = temperature, a = a, t=t, top_k = top_k)
    elif method=="wf":
        generated_text_prefix = generate_RSA_wf(model, tokenizer, target_embed=embeds[0], distr_embeds = embeds[1:],
                                                 temperature = temperature, a = a, t=t, top_k = top_k)
    else: 
        generated_text_prefix = generate_RSA_new(model, tokenizer, target_embed=embeds[0], distr1_embed=embeds[1], distr2_embed=embeds[2], 
                                             temperature = temperature, a = a)

    return generated_text_prefix








