## GENERATION ##

import pandas as pd
import clip
import os     
from os.path import dirname, abspath
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm.notebook import tqdm
import skimage.io as io
import PIL.Image

import gdown
import nltk
import re 

#os.environ["CUDA_LAUNCH_BLOCKING"]="1" 
#os.environ["TORCH_USE_CUDA_DSA"] = "1" 

#nltk.download()
# NLTK tags of nouns, adjectives and verbs 
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


# check if cuda is available
def get_device(device_id: int) -> D:
    if not torch.cuda.is_available():
        return CPU
    device_id = min(torch.cuda.device_count() - 1, device_id)
    return torch.device(f'cuda:{device_id}')


CUDA = get_device   
is_gpu = True
device = CUDA(0) if is_gpu else "cpu"
print(f"device: {device}")


d = dirname(dirname(abspath(__file__)))
#print("d:", d)
save_path =  d + "/pretrained_models/"
#print("save_path:", save_path)

# path of clipcap model
model_path = os.path.join(save_path, 'conceptual_weights.pt')

# load clip model
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

# load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


# unchanged clipcap model from Mokady et al. 2021
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
    
def download_model(name, model_path = model_path): 
    if name == 'Conceptual captions':
        id = "14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT"
    elif name == "COCO":
        id = "14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT"
    else:
        print("unknown model type!")
        return
        
    gdown.download(id=id, output=model_path, quiet=False)
    


def load_clip_model(model_path, prefix_length=10):
    model = ClipCaptionModel(prefix_length)
    model.load_state_dict(torch.load(model_path, map_location=CPU)) 
    model = model.eval() 
    device = CUDA(0) if is_gpu else "cpu"
    model = model.to(device)
    return(model)


# set flag to False if generation model is already downloaded
download_generation_model = True
if download_generation_model:
    download_model("Conceptual captions", model_path=model_path)
generation_model = load_clip_model(model_path)



## generation
# get clip prefix embedding from image path 
def preproc_img(path, proj_model=None, prefix_length = 10):
    # read image using scikit-image
    img = io.imread(path)
    # convert to PIL format 
    pil_img = PIL.Image.fromarray(img)

    # preprocess image using CLIP 
    pil_img = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        # extract CLIP prefix embedding
        prefix = clip_model.encode_image(pil_img).to(device, dtype=torch.float32)
        prefix_embed = proj_model.clip_project(prefix).reshape(1, prefix_length, -1)
    return prefix_embed


def get_logits(embeds, temperature, model=None):
    # get logits from gpt2, modify given temperature
    outputs = model.gpt(inputs_embeds=embeds)
    logits = outputs.logits
    logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
    return(logits)

# greedy RSA decoding with, set t to 1 to disable threshold based RSA decoding, set t to 0 to disable RSA decoding alltogether
def generate_RSA(
        model,
        tokenizer,
        df,
        entry_length=30,  # maximum number of words
        #top_k=100,
        temperature=1.,
        stop_token: str = '.',
        a = 1,
        pos_decoding = False, # set flag to True for POS decoding
        print_logits = False,
        t = 0.5, # threshold value 
        image_path = "./data/AbstractScenes_v1.1/RenderedScenes/",
        d = False,
        group_by = "scene_idx",
        cname = "caps_RSA" # name of output column

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
        # get list of unique scenes 
        scenes = [scene for scene in list(df[group_by]) if not scene in ["None", None]]
        scenes = list(set(scenes))
        for scene in tqdm(scenes, total = len(scenes)):
            # get subset of rows (pictures) belonging to scene 
            df_scene = df[df[group_by]==scene]
            for file in df_scene["file"]:
                
                tokens = None
                l0_prob_target = 0 
                RSA = True
                pos = True
                rsa_stop = None
                no_rsa_pos = []

                # get clip embeddings of target and distractors
                target_embed = preproc_img(image_path + file,  proj_model=model)
                distr_embeds = [preproc_img(image_path + img, proj_model=model) for img in list(df_scene["file"]) if not img==file]

                if not None in [target_embed] + distr_embeds:
                    # one embedding per image (1 target, 2 distractors)
                    generated_target = target_embed
                    generated_distr = distr_embeds
                else:
                    print("Embedding missing, generation stopped!")
                    break
                
                # get flat distribution for image priors 
                image_priors = [1/(len(distr_embeds)+1) for embed in range(len(distr_embeds)+1)]
                for i in range(entry_length):
                    # get output logits for each image from gpt-2
                    logits_target = get_logits(generated_target, temperature, model=model)
                    distr_logits = [get_logits(distr, temperature, model=model) for distr in generated_distr]

                    # sort token indices based on logit value 
                    _, sorted_indices_target = torch.sort(logits_target, descending=True)
                    
                    # reduce dimensionality
                    top_indices = sorted_indices_target.squeeze(0)#[:top_k]

                    top_logits_target = logits_target[:, top_indices] 
                    distr_top_logits = [logits[:, top_indices] for logits in distr_logits]

                    # apply softmax function to logits
                    probabilities_target = nnf.softmax(top_logits_target, dim=-1)             
                    distr_probs = [nnf.softmax(logits, dim=-1) for logits in distr_top_logits]
                    # if l0 probablity for target < threshold: apply RSA decoding 
                    if l0_prob_target < t: 
                        if pos_decoding: 
                            # check if probablities of nouns, adjectives and verbs is higher than that of other parts of speech in distribution
                            pos = get_pos(logits_target, probabilities_target)
                        # if check is positive: apply RSA decoding
                        if pos: 
                            # pragmatic reasoning 
                            # S0 distribution is just output distribution of GPT-2 model given each image
                            s0_probs = [probabilities_target] + distr_probs
                            # L0 distribution is normalized S0 distribution multiplied by image priors
                            l0_probs_no_n = [prob * image_priors[i] for i, prob in enumerate(s0_probs)]
                            l0_sum = sum(l0_probs_no_n)
                            l0_probs = [prob/l0_sum for prob in l0_probs_no_n]
                            # utility function
                            u = [torch.log(prob) for prob in l0_probs]

                            # S1 distributions given each image
                            s1_probs_no_n = [torch.exp(a*utility - -torch.log(s0_probs[i])) for i, utility in enumerate(u)]
                            s1_sum = torch.tensor([torch.sum(w) for w in s1_probs_no_n])
                            s1_probs = [s1_prob/s1_sum[i] for i, s1_prob in enumerate(s1_probs_no_n)]

                            # S1 distribution of target image 
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
                            # append index to list of tokens where no RSA decoding was applied
                            no_rsa_pos.append(i)
                            # next token is highest probability token from GPT-2 output distribution 
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

                            # record point where RSA decoding was stopped based on threshold     
                            rsa_stop = i 
                            # disable RSA decoding from this token on 
                            RSA=False
                        # greedy decoding
                        next_token = torch.argmax(probabilities_target, -1).unsqueeze(0)

                    # extract embeddings for output word 
                    next_token_index = top_indices[next_token]
                    next_token_embed = model.gpt.transformer.wte(next_token_index)
                    if tokens == None:
                        tokens = next_token_index
                    else:
                        # add token to input
                        tokens = torch.cat((tokens, next_token_index), dim=1)
                    # add next token embedding to input embeddings for each image
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
                # append points where RSA decoding was disabled to list 
                rsa_stop_list.append(rsa_stop)
                no_rsa_pos_list.append(no_rsa_pos)
        for cap in generated_list: 
            # save generated sequence to larger dataframe 
            df.loc[df["file"]==cap[0], cname] = cap[1] 
    return (df, rsa_stop_list, no_rsa_pos_list)


# apply RSA decoding to logits (only used in beam search function)
def rearrange_RSA(logits, distr_logits, image_priors, l0_probs_tokens = None, 
                  top_k=100, t = 1, a=3, i = 0, RSA=True, pos = False, 
                  print_stops=False, device="CUDA:0"):

    # get tensor of zeros for L0 probabilities if none are in the input 
    if l0_probs_tokens == None: 
        l0_probs_tokens = torch.zeros(1,logits.shape[0]).to(device)
    top_logits_target = logits
    distr_top_logits = distr_logits
    
    pos_no_rsa = None 
    # apply POS based RSA decoding
    if pos:
        # get k highest probability tokens 
        _, topk_tokens = top_logits_target.topk(top_k, -1)
        # decode indexes 
        top_beam_words = [[re.sub("^ ", "", tokenizer.decode(ind)) for ind in beam] for beam in topk_tokens]
        # NLTK tagging 
        top_beam_words_tagged = [nltk.pos_tag(top_words) for top_words in top_beam_words]
        beam_tags = [[pos for w, pos in top_words_tagged] for top_words_tagged in top_beam_words_tagged]

        # add up the probabilities of nouns, adjectives and verbs and of other POS 
        c_tag_probs = 0
        nc_tag_probs = 0
        for beam_num, beam in enumerate(beam_tags): 
            for tc, tag in enumerate(beam): 
                if tag in content_tags: 
                    c_tag_probs += top_logits_target[beam_num][tc]
                else:
                    nc_tag_probs += top_logits_target[beam_num][tc]
        
        # disable RSA if the probalities for nouns, adjectives and verbs is lower
        if c_tag_probs < nc_tag_probs: 
            RSA = False
            pos_no_rsa = i
            
    # get highest probablity in L0 for target image            
    top_l0_prob = torch.max(l0_probs_tokens)
    rsa_stop = None

    # if highest probablitiy in L0 is lower than threshold, RSA decoding is applied 
    if RSA and (top_l0_prob.item() < t): 
        
        # S0 probs are equal to output distribution of gpt-2 
        s0_probs = torch.stack([top_logits_target] + distr_top_logits)
        # L0 probs are normalized S0 probs multiplied by image priors 
        l0_probs_no_n = s0_probs * image_priors[:,None,None]
        # l0 distribution is normalized along first dimension (dimension of images)
        l0_sum = torch.sum(l0_probs_no_n, 0)
        l0_probs = torch.div(l0_probs_no_n,l0_sum[None])
        
        # utility function 
        u = torch.log(l0_probs)

        # S1 distribution 
        s1_probs_no_n = torch.exp(a*u - -torch.log(s0_probs))
        # S1 distribution is normalized along third dimension (dimension of tokens)
        s1_sum = torch.sum(s1_probs_no_n, 2)
        s1_probs = torch.div(s1_probs_no_n, s1_sum.unsqueeze(2))

        top_logits_target = s1_probs[0]
        l0_probs_target = l0_probs[0]

    else:
        # if no RSA decoding is applied, L0 distribution is set as a zero vector 
        l0_probs_target = torch.zeros(*top_logits_target.shape).to(device)
        # indexes where RSA decoding was disabled based on threshold are collected
        if pos==False:
            if RSA == True:
                if print_stops:
                    print("stopped RSA decoding after {} words with highest l0 probability = {}".format(i, top_l0_prob))
                # RSA decoding is disabled from this token on 
                RSA = False
                rsa_stop = i 
    
        else:
            # points where RSA decoding was disabled based on POS are collected 
            if print_stops:
                print("RSA disabled at index {} due to POS decoding".format(i))
            # set RSA to True again to keep checking for POS at next step
            RSA = True
    
    # nan values which are caused by log function are substituted by zeros
    top_logits_target = top_logits_target.nan_to_num()
    l0_probs_target = l0_probs_target.nan_to_num()

    return(top_logits_target, l0_probs_target, RSA, rsa_stop, pos_no_rsa)



def generate_beam_RSA(df, generation_model = None, tokenizer=None, entry_length=30, temperature = 1, stop_token = ".", 
                      top_k = 100, # only used for POS based RSA decoding 
                      t=1, # set threshold to 0 to disable RSA decoding, set to 1 to disable threshold based RSA decoding, otherwise set to 0.7
                      a=3, # rationality hyperparameter
                      beam_size=5, image_path = "./data/AbstractScenes_v1.1/RenderedScenes/", 
                      pos=False, #set flag to True for POS based RSA decoding 
                      print_stops = False, cname="caps_RSA" # name of output colunm
                      ):
    generation_model.eval()
    generated_list = []
    rsa_stop_list = []
    pos_no_rsa_scene = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    device = next(generation_model.parameters()).device
    pos_no_rsa_sum = 0
    length_sum = 0
    
    # iterate over set of scenes 
    for scene in tqdm(list(set(df["scene_idx"]))):
        # get subset of rows (images) belonging to the current scene 
        df_scene = df[df["scene_idx"]==scene]
        for file in df_scene["file"]:
            pos_no_rsa_list = []
            tokens = None
            RSA = True
            # get clip embeddings for each image 
            embed = preproc_img(image_path + file, proj_model=generation_model)
            distr_embeds = [preproc_img(image_path + img, proj_model=generation_model) for img in list(df_scene["file"]) if not img==file]
            image_num = len(distr_embeds) + 1 
            # set image priors to flat distrbution 
            image_priors = torch.tensor([1/image_num]*image_num).to(device)
            scores = None
            
            # length of sequences for each beam
            seq_lengths = torch.ones(beam_size, device=device)
            # 0 or 1 for each beam based on whether it has stopped yet 
            is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
            pos_no_rsa_sent = []

            with torch.no_grad():
                generated = embed
                generated_distr = distr_embeds

                for i in range(entry_length):
                    # get logits for each image from gpt-2 model
                    logits = get_logits(generated, temperature=temperature, model=generation_model)
                    distr_logits = [get_logits(embed, temperature=temperature, model=generation_model) for embed in generated_distr]
                    # apply softmax function to logits 
                    logits = logits.softmax(-1)
                    distr_logits = [lg.softmax(-1) for lg in distr_logits]
                    if scores is None:
                        # if first word of the sequence
                        # apply RSA decoding to softmax vectors 
                        top_logits_target, l0_probs_target, RSA, rsa_stop, pos_no_rsa = rearrange_RSA(logits, distr_logits, image_priors, top_k=top_k, 
                                                                                                      a=a, t=t, i=i, RSA=RSA, pos=pos, 
                                                                                                      print_stops=print_stops, device=device)
                        # select top k highest probably tokens to append to k beams 
                        scores, next_tokens = top_logits_target.topk(beam_size, -1)
                        # get LO probabilities for the selected tokens
                        l0_probs_tokens = l0_probs_target.gather(1, next_tokens)
                        # reduce dimensionality of scores
                        scores = scores.squeeze(0)
                        # append selected tokens to beams for each image 
                        generated = generated.expand(beam_size, *generated.shape[1:])
                        generated_distr = [distr.expand(beam_size, *distr.shape[1:]) for distr in generated_distr]
                        next_tokens = next_tokens.permute(1, 0)
                        if tokens is None:
                            tokens = next_tokens
                        else:
                            # add tokens to outputs
                            tokens = tokens.expand(beam_size, *tokens.shape[1:])
                            tokens = torch.cat((tokens, next_tokens), dim=1)

                    else:
                        # apply RSA decoding to softmax vectors 
                        top_logits_target, l0_probs_target, RSA, rsa_stop, pos_no_rsa = rearrange_RSA(logits, distr_logits, image_priors, 
                                                                                                      l0_probs_tokens=l0_probs_tokens, top_k=top_k, a=a, 
                                                                                                      t=t, i=i, RSA=RSA, pos=pos, print_stops=print_stops, device=device)
                        # set probabilities of stopped beams to -infinite
                        top_logits_target[is_stopped] = -float(np.inf)
                        top_logits_target[is_stopped, 0] = 0

                        # add probablities of tokens to sequence probablities 
                        scores_sum = scores[:, None] + top_logits_target
                        # add 1 to seq lengths of sequences which haven't stopped 
                        seq_lengths[~is_stopped] += 1

                        # divide probablity sum of sequences by their length
                        scores_sum_average = scores_sum / seq_lengths[:, None]
                        scores_sum_average = scores_sum_average
                        # select k next tokens to append to beams 
                        scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                        next_tokens_source = next_tokens // scores_sum.shape[1]
                        
                        # select sequence lengths of next tokens 
                        seq_lengths = seq_lengths[next_tokens_source]
                        next_tokens = next_tokens % scores_sum.shape[1]
                        next_tokens = next_tokens.unsqueeze(1)

                        # get L0 probabilities of next tokens 
                        l0_probs_tokens = l0_probs_target.gather(1, next_tokens)
                        
                        # add selected tokens to respective beams 
                        tokens = tokens[next_tokens_source]
                        tokens = torch.cat((tokens, next_tokens), dim=1)

                        # select embeddings of chosen beams 
                        generated = generated[next_tokens_source]
                        generated_distr = [distr[next_tokens_source] for distr in generated_distr]

                        # divide sums of probalities by sequence lengths
                        scores = scores_sum_average * seq_lengths
                        
                        # update vector with information about stopped beams 
                        is_stopped = is_stopped[next_tokens_source]

                    # get token embeddings for next tokens 
                    next_token_embed = generation_model.gpt.transformer.wte(next_tokens.squeeze())
                    next_token_embed = next_token_embed.view(generated.shape[0], 1, -1)
                    
                    # append token embeddings to input for each image 
                    generated = torch.cat((generated, next_token_embed), dim=1)
                    generated_distr = [torch.cat((distr, next_token_embed), dim=1) for distr in generated_distr]

                    # check if stop tokens have been generated 
                    is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
                    
                    # collect stops if RSA decoding has been stopped based on threshold or POS 
                    if rsa_stop !=None:
                        rsa_stop_list.append(rsa_stop)
                    if pos_no_rsa != None:
                        pos_no_rsa_sent.append(pos_no_rsa)
                    if is_stopped.all(): # if all beams have been stopped: stop generation 
                        break
                    
                pos_no_rsa_list.append(pos_no_rsa_sent)
            
            # divide sum sequence probablities by sequence lengths 
            scores = scores / seq_lengths
            output_list = tokens.cpu().numpy()
            # decode output beams 
            output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
            # sort beams by sequence probability 
            order = scores.argsort(descending=True)
            # select highest probability beam 
            output_text = [output_texts[i] for i in order][0]
            # add seq_lenght of chosen output to overall seq lengths of outputs to calculate proportion of tokens that were generated using 
            # RSA decoding later
            seq_length = seq_lengths[order[0]]
            length_sum += seq_length
            
            # append selected beam to outputs 
            generated_list.append((file, output_text))
            
            pos_no_rsa_output = pos_no_rsa_list[order[0]]
            pos_no_rsa_len = len(pos_no_rsa_output)
            # add number of tokens that were generated using RSA decoding to overall number 
            pos_no_rsa_sum += pos_no_rsa_len
            
            pos_no_rsa_scene.append(pos_no_rsa_output)

    for cap in generated_list: 
        # add output to large dataframe
        df.loc[df["file"]==cap[0], cname] = cap[1] 
        
    return(df, rsa_stop_list, pos_no_rsa_scene)



# for POS based RSA decoding: 
# check if sum of probalities for nouns, adjectives and verbs is higher then sum probabilities of other POS among top-k tokens
def get_pos(logits_target, probabilities_target, top_k=100):
    _, sorted_indices_target = torch.sort(logits_target, descending=True)
    # get top-k indices
    top_indices = sorted_indices_target.squeeze(0)[:top_k]
    # decode indices
    top_words = [re.sub("^ ", "", tokenizer.decode(ind)) for ind in top_indices]
    # tag using NLTK 
    top_words_tagged = nltk.pos_tag(top_words)
    tags = [pos for w, pos in top_words_tagged]

    # get probability sums of the two categories 
    c_tag_probs = 0
    nc_tag_probs = 0
    for tc, tag in enumerate(tags): 
        if tag in content_tags: 
            c_tag_probs += probabilities_target[0][tc]
        else:
            nc_tag_probs += probabilities_target[0][tc]

    return(c_tag_probs > nc_tag_probs)


# for debugging
if __name__=="__main__":
    
    current_folder = dirname(abspath(__file__))
    image_path=current_folder + "/data/AbstractScenes_v1.1/RenderedScenes/"
    df = pd.read_csv(current_folder + "/data/AbstractScenes_v1.1/processed_data/train_df_r3.csv")
    test_scenes = list(df["scene_idx"])[:10]
    df = df[df["scene_idx"].isin(test_scenes)]                   
    df, rsa_stop_list, pos_no_rsa_scene = generate_beam_RSA(df, generation_model, tokenizer, image_path=image_path, t=0.7, pos=True)
    df.to_csv(current_folder + "/temp/test_df_t_pos.csv")




