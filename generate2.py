# CLIP Captioning mit RSA-basiertem pragmatischem Decoding 
## von Hannes Gröner 
### Code abgewandelt von https://github.com/rmokady/CLIP_prefix_caption/tree/main/notebooks
# error: line 380
#@title Imports
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

os.environ["CUDA_LAUNCH_BLOCKING"]="1" 
os.environ["TORCH_USE_CUDA_DSA"] = "1" 


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
is_gpu = True
device = CUDA(0) if is_gpu else "cpu"


d = dirname(dirname(abspath(__file__)))
print("d:", d)
#current_directory = os.getcwd()
save_path =  d + "/pretrained_models/"
print("save_path:", save_path)
#os.makedirs(save_path, exist_ok=True)
model_path = os.path.join(save_path, 'conceptual_weights.pt')

clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


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
    
#@title Choose pretrained model - COCO or Coneptual captions
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


download_generation_model = False
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
        group_by = "scene_idx",
        cname = "caps_RSA"

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
        scenes = list(set(scenes))
        for scene in tqdm(scenes, total = len(scenes)):
            
            df_scene = df[df[group_by]==scene]
            for file in df_scene["file"]:
                
                tokens = None
                l0_prob_target = 0 
                RSA = True
                pos = True
                rsa_stop = None
                no_rsa_pos = []

                #dnames = ["d_" + str(num) for num in range(6)]
                target_embed = preproc_img(image_path + file,  proj_model=model)
                distr_embeds = [preproc_img(image_path + img, proj_model=model) for img in list(df_scene["file"]) if not img==file]

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
                    logits_target = get_logits(generated_target, temperature, model=model)
                    distr_logits = [get_logits(distr, temperature, model=model) for distr in generated_distr]

                    # sort token indices based on logit value 
                    _, sorted_indices_target = torch.sort(logits_target, descending=True)
                    
                    # get top-k indices
                    top_indices = sorted_indices_target.squeeze(0)[:top_k]
                    #print("top_indices:", top_indices)

                    # select top-k logits for target and distractors
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
        for cap in generated_list: 
            df.loc[df["file"]==cap[0], cname] = cap[1] 
    return (df, rsa_stop_list, no_rsa_pos_list)


def rearrange_RSA(logits, distr_logits, image_priors, l0_probs_tokens = 0, top_k=100, t = 1, a=3, i = 0, RSA=True, pos = False, print_stops=False):

    #topk_scores, topk_tokens = logits.topk(top_k, -1)
    top_logits_target = logits

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
        
        if c_tag_probs < nc_tag_probs: 
            RSA = False
            pos_no_rsa = i
            
                    
    distr_top_logits = distr_logits
    #lookup_l0 = torch.zeros(topk_tokens.shape[0],50257).to(device)

    top_l0_prob = torch.max(l0_probs_tokens)

    rsa_stop = None

    if RSA and (top_l0_prob.item() < t): 
        # pragmatic reasoning 
        # RuntimeError: stack expects each tensor to be equal size, but got [5, 50257] at entry 0 and [1, 50257] at entry 1
        s0_probs = torch.stack([top_logits_target] + distr_top_logits)
        l0_probs_no_n = s0_probs * image_priors[:,None,None]

        l0_sum = torch.sum(l0_probs_no_n, 2)

        l0_probs = torch.div(l0_probs_no_n,l0_sum[:,None])
        u = torch.log(l0_probs)

        s1_probs_no_n = torch.exp(a*u - -torch.log(s0_probs))
        s1_sum = torch.sum(s1_probs_no_n, 2)
        s1_probs = torch.div(s1_probs_no_n, s1_sum[:,None])

        top_logits_target = s1_probs[0]
        l0_probs_target = l0_probs[0]#generate.py
        '''
        for dim in range(lookup_l0.shape[0]):
            lookup_l0[dim][topk_tokens[dim]] = l0_probs_target[dim]
        '''
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
        
    return(top_logits_target, l0_probs_target, RSA, rsa_stop, pos_no_rsa)



def generate_beam_RSA(df, generation_model = None, tokenizer=None, entry_length=67, temperature = 1, stop_token = ".", top_k = 100, t= 1, a=3, beam_size=5, 
                      image_path = "./data/AbstractScenes_v1.1/RenderedScenes/", d=False, pos=False, print_stops = False,
                      cname="caps_RSA"):
    generation_model.eval()
    generated_list = []
    rsa_stop_list = []
    pos_no_rsa_scene = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    device = next(generation_model.parameters()).device
    pos_no_rsa_sum = 0
    length_sum = 0
        
    for scene in tqdm(list(set(df["scene_idx"]))):
        
        df_scene = df[df["scene_idx"]==scene]
        for file in df_scene["file"]:
            pos_no_rsa_list = []
            tokens = None
            RSA = True

            embed = preproc_img(image_path + file, proj_model=generation_model)
            distr_embeds = [preproc_img(image_path + img, proj_model=generation_model) for img in list(df_scene["file"]) if not img==file]
            image_num = len(distr_embeds) + 1 
            image_priors = torch.tensor([1/image_num]*image_num).to(device)
            l0_probs_tokens = torch.tensor([[0]*5]).to(device)
            generation_model.eval()
            stop_token_index = tokenizer.encode(stop_token)[0]
            scores = None
            device = next(generation_model.parameters()).device
            seq_lengths = torch.ones(beam_size, device=device)
            is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
            pos_no_rsa_sent = []

            with torch.no_grad():
                generated = embed
                generated_distr = distr_embeds

                for i in range(entry_length):
                    logits = get_logits(generated, temperature=temperature, model=generation_model)
                    distr_logits = [get_logits(embed, temperature=temperature, model=generation_model) for embed in generated_distr]
                    logits = logits.softmax(-1)
                    distr_logits = [lg.softmax(-1) for lg in distr_logits]

                    if scores is None:
                        top_logits_target, l0_probs_tokens, RSA, rsa_stop, pos_no_rsa = rearrange_RSA(logits, distr_logits, image_priors, 
                                                                                                      l0_probs_tokens=l0_probs_tokens, top_k=top_k, 
                                                                                                      a=a, t=t, i=i, RSA=RSA, pos=pos, 
                                                                                                      print_stops=print_stops)
                        scores, next_tokens = top_logits_target.topk(beam_size, -1)
                        l0_probs_tokens = l0_probs_tokens.gather(1, next_tokens)
                        scores = scores.squeeze(0)
                        generated = generated.expand(beam_size, *generated.shape[1:])
                        tokens = next_tokens.permute(1, 0)

                    else:
                        top_logits_target, l0_probs_tokens, RSA, rsa_stop, pos_no_rsa = rearrange_RSA(logits, distr_logits, image_priors, 
                                                                                                      l0_probs_tokens=l0_probs_tokens, top_k=top_k, a=a, 
                                                                                                      t=t, i=i, RSA=RSA, pos=pos, print_stops=print_stops)
                        top_logits_target[is_stopped] = -float(np.inf)
                        top_logits_target[is_stopped, 0] = 0

                        scores_sum = scores[:, None] + top_logits_target

                        seq_lengths[~is_stopped] += 1

                        scores_sum_average = scores_sum / seq_lengths[:, None]
                        scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                        
                        next_tokens_source = next_tokens // scores_sum.shape[1]
                        
                        seq_lengths = seq_lengths[next_tokens_source]
                        next_tokens = next_tokens % scores_sum.shape[1]
                        next_tokens = next_tokens.unsqueeze(1)

                        l0_probs_tokens = l0_probs_tokens.gather(1, next_tokens)
                        tokens = tokens[next_tokens_source]
                        tokens = torch.cat((tokens, next_tokens), dim=1)
                        generated = generated[next_tokens_source]
                        scores = scores_sum_average * seq_lengths
                        
                        is_stopped = is_stopped[next_tokens_source]

                    next_token_embed = generation_model.gpt.transformer.wte(next_tokens.squeeze())
                    next_token_embed = next_token_embed.view(generated.shape[0], 1, -1)

                    generated = torch.cat((generated, next_token_embed), dim=1)
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
            output_list = tokens.cpu().numpy()
            output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
            order = scores.argsort(descending=True)
            output_text = [output_texts[i] for i in order][0]
            # add seq_lenght of chosen output to overall seq lengths of outputs to calculate proportion of tokens that were generated using 
            # RSA decoding later
            seq_length = seq_lengths[order[0]]
            length_sum += seq_length
            
            generated_list.append((file, output_text))
            
            pos_no_rsa_output = pos_no_rsa_list[order[0]]
            pos_no_rsa_len = len(pos_no_rsa_output)
            # add number of tokens that were generated using RSA decoding to overall number 
            pos_no_rsa_sum += pos_no_rsa_len
            
            pos_no_rsa_scene.append(pos_no_rsa_output)
            
        #pos_no_rsa_average = pos_no_rsa_sum/length_sum
        #no_rsa_t_average = sum(rsa_stop_list)/length_sum

    for cap in generated_list: 
        df.loc[df["file"]==cap[0], cname] = cap[1] 
        
    #return(df, rsa_stop_list, pos_no_rsa_scene, no_rsa_t_average, pos_no_rsa_average)
    return(df, rsa_stop_list, pos_no_rsa_scene)



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


if __name__=="__main__":
    
    current_folder = dirname(abspath(__file__))
    image_path=current_folder + "/data/AbstractScenes_v1.1/RenderedScenes/"
    df = pd.read_csv(current_folder + "/data/AbstractScenes_v1.1/processed_data/train_df.csv")
    test_scenes = list(df["scene_idx"])[:10]
    df = df[df["scene_idx"].isin(test_scenes)]
    '''
    df, rsa_stop_list, _ = generate_RSA(generation_model, tokenizer, df, temperature=0.8, a=3, t = 0.7, cname="caps_RSA_t", 
                                        image_path=image_path)
    df, _, no_rsa_pos_list = generate_RSA(generation_model, tokenizer, df, temperature=0.8, a=3, t = 1, pos_decoding=True, cname="caps_RSA_pos", image_path=image_path)
    '''
    df, rsa_stop_list, pos_no_rsa_scene = generate_beam_RSA(df, generation_model, tokenizer, image_path=image_path)
    df.to_csv(current_folder + "/temp/test_df.csv")




