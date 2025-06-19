import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 
import re
from .ablate import AblateGPT 

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()
            # print((W==0).sum().item())
            # print(sub_params)
            # exit()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

import re

def parse_line(line):
    # 定义字段名
    keys = ['q', 'k', 'v', 'o_proj', 'gate', 'up', 'down', 'layer']
    
    # 初始化一个空字典
    parsed_dict = {}
    
    # 遍历字段名并提取对应的值
    for key in keys:
        if key == 'layer':
            pattern = f'{key}(\\d+)'
        else:
            pattern = f'{key}(\\d+\\.\\d+)'
        match = re.search(pattern, line)
        if match:
            parsed_dict[key] = match.group(1)
    
    # 将 'layer' 键的值改为 'LayerNumber'，并转换为字符串
    parsed_dict['LayerNumber'] = parsed_dict.pop('layer')
    
    return parsed_dict

class LineParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = open(file_path, 'r')
        self.counter = 0

    def parse_next_line(self):
        line = self.file.readline().strip()
        if line:
            self.counter += 1
            return parse_line(line)
        else:
            self.file.close()
            return None







def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, **kwargs):
    layers = model.model.layers 
    weight_dict = {}
    LayerNumber = float(kwargs['LayerNumber'])
    # layer_prate = [0.51, 0.51, 0.51, 0.51, 0.51, 0.51, 0.71, 0.51, 0.71, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51, 0.71, 0.51, 0.51, 0.51, 0.51, 0.51, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.51, 0.51, 0.51]
    layer_prate = [0.546, 0.522, 0.47, 0.47, 0.488, 0.525, 0.683, 0.462, 0.669, 0.552, 0.502, 0.508, 0.489, 0.535, 0.534, 0.478, 0.699, 0.552, 0.517, 0.558, 0.522, 0.527, 0.742, 0.696, 0.732, 0.725, 0.678, 0.683, 0.754, 0.702, 0.704, 0.76, 0.677, 0.734, 0.722, 0.756, 0.665, 0.514, 0.486, 0.462]
    
    file_path = 'name_inner_layer_final_config.txt'
    parser = LineParser(file_path)    
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        kwargs = parser.parse_next_line()
        print(f"第 {parser.counter} 条记录: kwargs is {kwargs}")
        LayerNumber = float(kwargs['LayerNumber'])
        # 初始化一个空的张量列表来存储权重
        # weights_list = []
        for name in subset:
            W = subset[name].weight.data 
            
            
            # if(('q_proj' in name) or ('k_proj' in name) or ('v_proj' in name)):
            #     sparsity_ratio = args.sparsity_ratio - 0.05
            # if(('gate_proj' in name) or ('up_proj' in name) or ('down_proj' in name) or ('o_proj' in name)):
            #     sparsity_ratio = args.sparsity_ratio + 0.0167            
            
            
            pratio = 0.985

            if(i==LayerNumber):
                for keyword in kwargs:
                    if keyword in name:
                        sparsity_ratio = float(kwargs[keyword]) / 10 * pratio 
                        print(sparsity_ratio)
                        break                
            else:
                # sparsity_ratio = 0.6 
                sparsity_ratio = layer_prate[i]
                           
                           
                           
                           
                           
                           
                           
                           
                           
                           
            # sparsity_ratio = float(kwargs[str(i)])
            
            
            
            # pratio = 0.985
            # if(i==0):
            #     if(('q' in name)or('k' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.783*pratio
            #     else:
            #         sparsity_ratio = 0.42*pratio         
            # if(i==1):
            #     if(('v' in name)or('k' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.32*pratio
            #     else:
            #         sparsity_ratio = 0.5726*pratio
            # if(i==2):
            #     if(('gate' in name)or('up' in name)):
            #         sparsity_ratio = 0.6713*pratio
            #     else:
            #         sparsity_ratio = 0.38*pratio  
            # if(i==3):
            #     if(('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.36*pratio
            #     else:
            #         sparsity_ratio = 0.5397   *pratio                                  
            # if(i==4):
            #     if(('q' in name)or('k' in name)or('v' in name)):
            #         sparsity_ratio = 0.32*pratio
            #     else:
            #         sparsity_ratio = 0.5726*pratio
            # if(i==5):
            #     if(('gate' in name)or('down' in name)):
            #         sparsity_ratio = 0.5472*pratio
            #     else:
            #         sparsity_ratio = 0.48*pratio 
            # if(i==6):# 111
            #     if(('gate' in name)):
            #         sparsity_ratio = 0.7448*pratio
            #     else:
            #         sparsity_ratio = 0.7      *pratio                           
            # if(i==7):
            #     if(('gate' in name)or('up' in name)):
            #         sparsity_ratio = 0.6713 *pratio
            #     else:
            #         sparsity_ratio = 0.38*pratio
            # if(i==8):#1111
            #     if(('k' in name)or('q' in name)):
            #         sparsity_ratio = 0.7605*pratio
            #     else:
            #         sparsity_ratio = 0.7    *pratio
            # if(i==9):
            #     if(('gate' in name)):
            #         sparsity_ratio = 0.5448*pratio
            #     else:
            #         sparsity_ratio = 0.5*pratio 
            # if(i==10):
            #     if(('gate' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.594 *pratio
            #     else:
            #         sparsity_ratio = 0.34*pratio  
            # if(i==11):
            #     if(('q' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.26*pratio
            #     else:
            #         sparsity_ratio = 0.5924     *pratio          
            # if(i==12):
            #     if(('gate' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.6137*pratio
            #     else:
            #         sparsity_ratio = 0.3*pratio 
            # if(i==13):
            #     if(('v' in name)or('o_proj' in name)or('down' in name)):
            #         sparsity_ratio = 0.28*pratio
            #     else:
            #         sparsity_ratio = 0.6561  *pratio                                   
            # if(i==14):
            #     if(('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.1*pratio
            #     else:
            #         sparsity_ratio = 0.5912*pratio
            # if(i==15):
            #     if(('gate' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.5544*pratio
            #     else:
            #         sparsity_ratio = 0.42*pratio 
            # if(i==16):
            #     if(('q' in name)or('k' in name)or('gate' in name)):
            #         sparsity_ratio = 0.7257*pratio
            #     else:
            #         sparsity_ratio = 0.7      *pratio                             
            # if(i==17):
            #     if(('gate' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.5347*pratio
            #     else:
            #         sparsity_ratio = 0.46*pratio
            # if(i==18):
            #     if(('gate' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.5544*pratio
            #     else:
            #         sparsity_ratio = 0.42*pratio    
            # if(i==19):
            #     if(('q' in name)or('k' in name)or('v' in name)):
            #         sparsity_ratio = 0.46*pratio
            #     else:
            #         sparsity_ratio = 0.5265*pratio
            # if(i==20):
            #     if(('gate' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.5643*pratio
            #     else:
            #         sparsity_ratio = 0.4*pratio 
            # if(i==21):
            #     if(('q' in name)or('k' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.38*pratio
            #     else:
            #         sparsity_ratio = 0.5742*pratio
            # if(i==22):
            #     if(('q' in name)or('k' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.6*pratio
            #     else:
            #         sparsity_ratio = 0.7643 *pratio
            # if(i==23):#111
            #     if(('q' in name)or('k' in name)or('v' in name)):
            #         sparsity_ratio = 0.7403*pratio
            #     else:
            #         sparsity_ratio = 0.7       *pratio                               
            # if(i==24):
            #     if(('o_proj' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.7*pratio
            #     else:
            #         sparsity_ratio = 0.7212*pratio
            # if(i==25):#111
            #     if(('q' in name)or('k' in name)or('v' in name)or('o_proj' in name)or('gate' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.71*pratio
            #     else:
            #         sparsity_ratio = 0.0 *pratio
            # if(i==26):#111
            #     if(('gate' in name)or('up' in name)):
            #         sparsity_ratio = 0.772*pratio
            #     else:
            #         sparsity_ratio = 0.66    *pratio                                   
            # if(i==27):#111
            #     if(('k' in name)or('gate' in name)or('down' in name)):
            #         sparsity_ratio = 0.66*pratio
            #     else:
            #         sparsity_ratio = 0.7661*pratio
            # if(i==28):#111
            #     if(('k' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.7403*pratio
            #     else:
            #         sparsity_ratio = 0.7  *pratio
            # if(i==29):#111
            #     if(('o_proj' in name)or('v' in name)):
            #         sparsity_ratio = 0.68 *pratio
            #     else:
            #         sparsity_ratio = 0.7159 *pratio
            # if(i==30):#111
            #     if(('q' in name)or('k' in name)or('v' in name)or('down' in name)):
            #         sparsity_ratio = 0.64*pratio
            #     else:
            #         sparsity_ratio = 0.7723*pratio
            # if(i==31):  #111
            #     if(('v' in name)or('k' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.801*pratio
            #     else:
            #         sparsity_ratio = 0.68*pratio
            # if(i==32):
            #     if(('q' in name)or('k' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.64*pratio
            #     else:
            #         sparsity_ratio = 0.7331 *pratio
            # if(i==33):
            #     if(('q' in name)or('k' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.5*pratio
            #     else:
            #         sparsity_ratio = 0.8137    *pratio                                 
            # if(i==34):
            #     if(('q' in name)or('gate' in name)or('down' in name)):
            #         sparsity_ratio = 0.66*pratio
            #     else:
            #         sparsity_ratio = 0.7661*pratio
            # if(i==35):
            #     if(('q' in name)or('k' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.54*pratio
            #     else:
            #         sparsity_ratio = 0.766 *pratio
            # if(i==36):
            #     if(('gate' in name)or('q' in name)or('k' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.64*pratio
            #     else:
            #         sparsity_ratio = 0.7969    *pratio                               
            # if(i==37):
            #     if(('q' in name)or('k' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.32*pratio
            #     else:
            #         sparsity_ratio = 0.6038*pratio
            # if(i==38):
            #     if(('down' in name)or('q' in name)or('k' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.4*pratio
            #     else:
            #         sparsity_ratio = 0.6465   *pratio
            # if(i==39):
            #     if(('gate' in name)or('k' in name)or('q' in name)or('down' in name)):
            #         sparsity_ratio = 0.4*pratio
            #     else:
            #         sparsity_ratio = 0.6832*pratio


                                                   
            
            
            
            
            
            
            # pratio = 0.985
            # if(i==0):
            #     if(('q' in name)or('k' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.4599*pratio
            #     else:
            #         sparsity_ratio = 0.5265*pratio         
            # if(i==1):
            #     if(('v' in name)or('k' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.0681*pratio
            #     else:
            #         sparsity_ratio = 0.6557*pratio
            # if(i==2):
            #     if(('gate' in name)or('up' in name)):
            #         sparsity_ratio = 0.5746*pratio
            #     else:
            #         sparsity_ratio = 0.458*pratio  
            # if(i==3):
            #     if(('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.1*pratio
            #     else:
            #         sparsity_ratio = 0.5912   *pratio                                  
            # if(i==4):
            #     if(('q' in name)or('k' in name)or('v' in name)):
            #         sparsity_ratio = 0.0066*pratio
            #     else:
            #         sparsity_ratio = 0.676*pratio
            # if(i==5):
            #     if(('gate' in name)or('down' in name)):
            #         sparsity_ratio = 0.8704*pratio
            #     else:
            #         sparsity_ratio = 0.2195*pratio 
            # if(i==6):# 111
            #     if(('gate' in name)):
            #         sparsity_ratio = 0.7541*pratio
            #     else:
            #         sparsity_ratio = 0.6973      *pratio                           
            # if(i==7):
            #     if(('gate' in name)or('up' in name)):
            #         sparsity_ratio = 0.4773 *pratio
            #     else:
            #         sparsity_ratio = 0.5363*pratio
            # if(i==8):#1111
            #     if(('k' in name)or('q' in name)):
            #         sparsity_ratio = 0.8599*pratio
            #     else:
            #         sparsity_ratio = 0.6803    *pratio
            # if(i==9):
            #     if(('gate' in name)):
            #         sparsity_ratio = 0.9921*pratio
            #     else:
            #         sparsity_ratio = 0.3715*pratio 
            # if(i==10):
            #     if(('gate' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.5725 *pratio
            #     else:
            #         sparsity_ratio = 0.3835*pratio  
            # if(i==11):
            #     if(('q' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.0243*pratio
            #     else:
            #         sparsity_ratio = 0.6701     *pratio          
            # if(i==12):
            #     if(('gate' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.5571*pratio
            #     else:
            #         sparsity_ratio = 0.4147*pratio 
            # if(i==13):
            #     if(('v' in name)or('o_proj' in name)or('down' in name)):
            #         sparsity_ratio = 0.3435*pratio
            #     else:
            #         sparsity_ratio = 0.6157  *pratio                                   
            # if(i==14):
            #     if(('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.1*pratio
            #     else:
            #         sparsity_ratio = 0.5912*pratio
            # if(i==15):
            #     if(('gate' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.6345*pratio
            #     else:
            #         sparsity_ratio = 0.2579*pratio 
            # if(i==16):
            #     if(('q' in name)or('k' in name)or('gate' in name)):
            #         sparsity_ratio = 0.7115*pratio
            #     else:
            #         sparsity_ratio = 0.709      *pratio                             
            # if(i==17):
            #     if(('gate' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.5259*pratio
            #     else:
            #         sparsity_ratio = 0.4778*pratio
            # if(i==18):
            #     if(('gate' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.6331*pratio
            #     else:
            #         sparsity_ratio = 0.2607*pratio    
            # if(i==19):
            #     if(('q' in name)or('k' in name)or('v' in name)):
            #         sparsity_ratio = 0.1*pratio
            #     else:
            #         sparsity_ratio = 0.6452*pratio
            # if(i==20):
            #     if(('gate' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.6386*pratio
            #     else:
            #         sparsity_ratio = 0.2496*pratio 
            # if(i==21):
            #     if(('q' in name)or('k' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.2466*pratio
            #     else:
            #         sparsity_ratio = 0.6401*pratio
            # if(i==22):
            #     if(('q' in name)or('k' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.6*pratio
            #     else:
            #         sparsity_ratio = 0.7643 *pratio
            # if(i==23):#111
            #     if(('q' in name)or('k' in name)or('v' in name)):
            #         sparsity_ratio = 0.6078*pratio
            #     else:
            #         sparsity_ratio = 0.7437       *pratio                               
            # if(i==24):
            #     if(('o_proj' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.7102*pratio
            #     else:
            #         sparsity_ratio = 0.7098*pratio
            # if(i==25):#111
            #     if(('q' in name)or('k' in name)or('v' in name)or('o_proj' in name)or('gate' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.71*pratio
            #     else:
            #         sparsity_ratio = 0.0 *pratio
            # if(i==26):#111
            #     if(('gate' in name)or('up' in name)):
            #         sparsity_ratio = 0.8465*pratio
            #     else:
            #         sparsity_ratio = 0.6    *pratio                                   
            # if(i==27):#111
            #     if(('k' in name)or('gate' in name)or('down' in name)):
            #         sparsity_ratio = 0.883*pratio
            #     else:
            #         sparsity_ratio = 0.5157*pratio
            # if(i==28):#111
            #     if(('k' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.5694*pratio
            #     else:
            #         sparsity_ratio = 0.7564  *pratio
            # if(i==29):#111
            #     if(('o_proj' in name)or('v' in name)):
            #         sparsity_ratio = 0.717 *pratio
            #     else:
            #         sparsity_ratio = 0.7086 *pratio
            # if(i==30):#111
            #     if(('q' in name)or('k' in name)or('v' in name)or('down' in name)):
            #         sparsity_ratio = 0.6513*pratio
            #     else:
            #         sparsity_ratio = 0.7623*pratio
            # if(i==31):  #111
            #     if(('v' in name)or('k' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.5678*pratio
            #     else:
            #         sparsity_ratio = 0.7569*pratio
            # if(i==32):
            #     if(('q' in name)or('k' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.6*pratio
            #     else:
            #         sparsity_ratio = 0.7463 *pratio
            # if(i==33):
            #     if(('q' in name)or('k' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.6*pratio
            #     else:
            #         sparsity_ratio = 0.7643    *pratio                                 
            # if(i==34):
            #     if(('q' in name)or('gate' in name)or('down' in name)):
            #         sparsity_ratio = 0.7693*pratio
            #     else:
            #         sparsity_ratio = 0.6434*pratio
            # if(i==35):
            #     if(('q' in name)or('k' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.6*pratio
            #     else:
            #         sparsity_ratio = 0.7463 *pratio
            # if(i==36):
            #     if(('gate' in name)or('q' in name)or('k' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.6391*pratio
            #     else:
            #         sparsity_ratio = 0.798    *pratio                               
            # if(i==37):
            #     if(('q' in name)or('k' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.013*pratio
            #     else:
            #         sparsity_ratio = 0.7554*pratio
            # if(i==38):
            #     if(('down' in name)or('q' in name)or('k' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.4652*pratio
            #     else:
            #         sparsity_ratio = 0.5655   *pratio
            # if(i==39):
            #     if(('gate' in name)or('k' in name)or('q' in name)or('down' in name)):
            #         sparsity_ratio = 0.5613*pratio
            #     else:
            #         sparsity_ratio = 0.4292*pratio            
            
            
            
            
            
            
            
            
            
            
            
            
            # pratio = 0.985
            # if(i==0):
            #     if(('q' in name)or('k' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.8069*pratio
            #     else:
            #         sparsity_ratio = 0.46*pratio         
            # if(i==1):
            #     if(('v' in name)or('k' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.5*pratio
            #     else:
            #         sparsity_ratio = 0.5293*pratio
            # if(i==2):
            #     if(('gate' in name)or('up' in name)):
            #         sparsity_ratio = 0.6561*pratio
            #     else:
            #         sparsity_ratio = 0.32*pratio  
            # if(i==3):
            #     if(('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.36*pratio
            #     else:
            #         sparsity_ratio = 0.4918   *pratio                                  
            # if(i==4):
            #     if(('q' in name)or('k' in name)or('v' in name)):
            #         sparsity_ratio = 0.3*pratio
            #     else:
            #         sparsity_ratio = 0.55*pratio
            # if(i==5):
            #     if(('gate' in name)or('down' in name)):
            #         sparsity_ratio = 0.6056*pratio
            #     else:
            #         sparsity_ratio = 0.46*pratio 
            # if(i==6):# 111
            #     if(('gate' in name)):
            #         sparsity_ratio = 0.6934*pratio
            #     else:
            #         sparsity_ratio = 0.68      *pratio                           
            # if(i==7):
            #     if(('gate' in name)or('up' in name)):
            #         sparsity_ratio = 0.5389 *pratio
            #     else:
            #         sparsity_ratio = 0.4*pratio
            # if(i==8):#1111
            #     if(('k' in name)or('q' in name)):
            #         sparsity_ratio = 0.7144*pratio
            #     else:
            #         sparsity_ratio = 0.66    *pratio
            # if(i==9):
            #     if(('gate' in name)):
            #         sparsity_ratio = 0.6634*pratio
            #     else:
            #         sparsity_ratio = 0.52*pratio 
            # if(i==10):
            #     if(('gate' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.5129 *pratio
            #     else:
            #         sparsity_ratio = 0.48*pratio  
            # if(i==11):
            #     if(('q' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.26*pratio
            #     else:
            #         sparsity_ratio = 0.5898     *pratio          
            # if(i==12):
            #     if(('gate' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.5922*pratio
            #     else:
            #         sparsity_ratio = 0.28*pratio 
            # if(i==13):
            #     if(('v' in name)or('o_proj' in name)or('down' in name)):
            #         sparsity_ratio = 0.42*pratio
            #     else:
            #         sparsity_ratio = 0.608  *pratio                                   
            # if(i==14):
            #     if(('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.1*pratio
            #     else:
            #         sparsity_ratio = 0.6199*pratio
            # if(i==15):
            #     if(('gate' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.5165*pratio
            #     else:
            #         sparsity_ratio = 0.4*pratio 
            # if(i==16):
            #     if(('q' in name)or('k' in name)or('gate' in name)):
            #         sparsity_ratio = 0.7289*pratio
            #     else:
            #         sparsity_ratio = 0.68      *pratio                             
            # if(i==17):
            #     if(('gate' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.5678*pratio
            #     else:
            #         sparsity_ratio = 0.52*pratio
            # if(i==18):
            #     if(('gate' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.5649*pratio
            #     else:
            #         sparsity_ratio = 0.42*pratio    
            # if(i==19):
            #     if(('q' in name)or('k' in name)or('v' in name)):
            #         sparsity_ratio = 0.5*pratio
            #     else:
            #         sparsity_ratio = 0.5771*pratio
            # if(i==20):
            #     if(('gate' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.5822*pratio
            #     else:
            #         sparsity_ratio = 0.4*pratio 
            # if(i==21):
            #     if(('q' in name)or('k' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.38*pratio
            #     else:
            #         sparsity_ratio = 0.5996*pratio
            # if(i==22):
            #     if(('q' in name)or('k' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.64*pratio
            #     else:
            #         sparsity_ratio = 0.7924 *pratio
            # if(i==23):#111
            #     if(('q' in name)or('k' in name)or('v' in name)):
            #         sparsity_ratio = 0.7445*pratio
            #     else:
            #         sparsity_ratio = 0.68       *pratio                               
            # if(i==24):
            #     if(('o_proj' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.72*pratio
            #     else:
            #         sparsity_ratio = 0.7455*pratio
            # if(i==25):#111
            #     if(('q' in name)or('k' in name)or('v' in name)or('o_proj' in name)or('gate' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.725*pratio
            #     else:
            #         sparsity_ratio = 0.0 *pratio
            # if(i==26):#111
            #     if(('gate' in name)or('up' in name)):
            #         sparsity_ratio = 0.7251*pratio
            #     else:
            #         sparsity_ratio = 0.64    *pratio                                   
            # if(i==27):#111
            #     if(('k' in name)or('gate' in name)or('down' in name)):
            #         sparsity_ratio = 0.62*pratio
            #     else:
            #         sparsity_ratio = 0.7537*pratio
            # if(i==28):#111
            #     if(('k' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.9785*pratio
            #     else:
            #         sparsity_ratio = 0.68  *pratio
            # if(i==29):#111
            #     if(('o_proj' in name)or('v' in name)):
            #         sparsity_ratio = 0.68 *pratio
            #     else:
            #         sparsity_ratio = 0.7064 *pratio
            # if(i==30):#111
            #     if(('q' in name)or('k' in name)or('v' in name)or('down' in name)):
            #         sparsity_ratio = 0.64*pratio
            #     else:
            #         sparsity_ratio = 0.761*pratio
            # if(i==31):  #111
            #     if(('v' in name)or('k' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.942*pratio
            #     else:
            #         sparsity_ratio = 0.7*pratio
            # if(i==32):
            #     if(('q' in name)or('k' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.64*pratio
            #     else:
            #         sparsity_ratio = 0.6892 *pratio
            # if(i==33):
            #     if(('q' in name)or('k' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.66*pratio
            #     else:
            #         sparsity_ratio = 0.7705    *pratio                                 
            # if(i==34):
            #     if(('q' in name)or('gate' in name)or('down' in name)):
            #         sparsity_ratio = 0.66*pratio
            #     else:
            #         sparsity_ratio = 0.7916*pratio
            # if(i==35):
            #     if(('q' in name)or('k' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.68*pratio
            #     else:
            #         sparsity_ratio = 0.7811 *pratio
            # if(i==36):
            #     if(('gate' in name)or('q' in name)or('k' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.62*pratio
            #     else:
            #         sparsity_ratio = 0.7208    *pratio                               
            # if(i==37):
            #     if(('q' in name)or('k' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.44*pratio
            #     else:
            #         sparsity_ratio = 0.5505*pratio
            # if(i==38):
            #     if(('down' in name)or('q' in name)or('k' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.36*pratio
            #     else:
            #         sparsity_ratio = 0.6423   *pratio
            # if(i==39):
            #     if(('gate' in name)or('k' in name)or('q' in name)or('down' in name)):
            #         sparsity_ratio = 0.32*pratio
            #     else:
            #         sparsity_ratio = 0.6856*pratio            
    
    
    
    
            
            
            
            
            
            
            
            
            
            # if(('q_proj' in name) or ('k_proj' in name) or ('v_proj' in name)):
            #     sparsity_ratio = prunelist[i] - 0.05
            # if(('gate_proj' in name) or ('up_proj' in name) or ('down_proj' in name) or ('o_proj' in name)):
            #     sparsity_ratio = prunelist[i] + 0.0167           
            
            
            
            # sparsity_ratio = args.sparsity_ratio
            
            
            
            W_metric = torch.abs(W)
            # weights_list.append(W)
            
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)
                mask = W_mask.bool().cpu()
                # if "q_proj" in name or "v_proj" in name:
                weight_dict["layers."+str(i)+"."+name] = mask
            W[W_mask] = 0
    # torch.save(weight_dict,  "/data/fhl/wanda/lora_ft/llama2_13b_groups_and_sens_7.7_weight_mask_allmodules.pth")
    # torch.save(weight_dict,  "/data/fhl/wanda/lora_ft/llama_7b05_mag_based_weight_mask_allmodules.pth")
def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, **kwargs):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    weight_dict = {}
    # LayerNumber = float(kwargs['LayerNumber'])
    # layer_prate = [0.546, 0.522, 0.47, 0.47, 0.488, 0.525, 0.683, 0.462, 0.669, 0.552, 0.502, 0.508, 0.489, 0.535, 0.534, 0.478, 0.699, 0.552, 0.517, 0.558, 0.522, 0.527, 0.742, 0.696, 0.732, 0.725, 0.678, 0.683, 0.754, 0.702, 0.704, 0.76, 0.677, 0.734, 0.722, 0.756, 0.665, 0.514, 0.486, 0.462]
    # layer_prate = [0.55, 0.47, 0.5, 0.44, 0.47, 0.56, 0.61, 0.52, 0.61, 0.54, 0.51, 0.5, 0.52, 0.53, 0.52, 0.52, 0.60, 0.57, 0.49, 0.47, 0.53, 0.55, 0.61, 0.62, 0.67, 0.89, 0.78, 0.84, 0.98, 0.78, 0.77, 0.69, 0.74, 0.66, 0.61, 0.69, 0.61, 0.54, 0.51, 0.42]
    print("loading calibdation data")# 用wikitext2来校验得到的wikitext2 ppl更好，用c4得到的结果略低。
    dataloader, _ = get_loaders("wikitext2",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
    file_path = 'name_inner_layer_final_config.txt'
    parser = LineParser(file_path) 
    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        # # ***
        # kwargs = parser.parse_next_line()
        # print(f"第 {parser.counter} 条记录: kwargs is {kwargs}")
        # LayerNumber = float(kwargs['LayerNumber'])
        # # ***

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            
            
            
            
            
            
            
            
            
            
            # pratio = 0.985
            # sparsity_ratio = layer_prate[i] * pratio
            
            
            
            
            
            
            # pratio = 1.0

            # if(i==LayerNumber):
            #     for keyword in kwargs:
            #         if keyword in name:
            #             sparsity_ratio = float(kwargs[keyword]) / 10 * pratio 
            #             print(sparsity_ratio)
            #             break                
            # else:
            #     sparsity_ratio = 0.6 
            #     # sparsity_ratio = layer_prate[i]            
            
            
            
            
            
            
            sparsity_ratio = float(kwargs[str(i)]) * 1
            
            
            
            
            
            
            # if(i==0):
            #     if(('q' in name)or('k' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.8069*pratio
            #         sparsity_ratio = 0.8069*pratio
            #         sparsity_ratio = 0.9889*pratio
            #         sparsity_ratio = 0.5642*pratio
            #     else:
            #         sparsity_ratio = 0.46*pratio     
            #         sparsity_ratio = 0.46*pratio   
            #         sparsity_ratio = 0.4*pratio    
            #         sparsity_ratio = 0.54*pratio
            # if(i==1):
            #     if(('v' in name)or('k' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.5*pratio
            #         sparsity_ratio = 0.3*pratio
            #         sparsity_ratio = 0.1*pratio
            #         sparsity_ratio = 0.52*pratio
            #     else:
            #         sparsity_ratio = 0.5293*pratio
            #         sparsity_ratio = 0.5952*pratio
            #         sparsity_ratio = 0.6611*pratio
            #         sparsity_ratio = 0.5227*pratio
            # if(i==2):
            #     if(('gate' in name)or('up' in name)):
            #         sparsity_ratio = 0.6561*pratio
            #         sparsity_ratio = 0.7057*pratio
            #         sparsity_ratio = 0.9291*pratio
            #         sparsity_ratio = 0.4824*pratio
            #     else:
            #         sparsity_ratio = 0.32*pratio  
            #         sparsity_ratio = 0.28*pratio  
            #         sparsity_ratio = 0.1*pratio
            #         sparsity_ratio = 0.46*pratio
            # if(i==3):
            #     if(('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.36*pratio
            #         sparsity_ratio = 0.26*pratio
            #         sparsity_ratio = 0.1*pratio
            #         sparsity_ratio = 0.46*pratio
            #     else:
            #         sparsity_ratio = 0.4918   *pratio    
            #         sparsity_ratio = 0.5116   *pratio       
            #         sparsity_ratio = 0.5433*pratio     
            #         sparsity_ratio = 0.472*pratio                  
            # if(i==4):
            #     if(('q' in name)or('k' in name)or('v' in name)):
            #         sparsity_ratio = 0.3*pratio
            #         sparsity_ratio = 0.3*pratio
            #         sparsity_ratio = 0.1*pratio
            #         sparsity_ratio = 0.48*pratio
            #     else:
            #         sparsity_ratio = 0.55*pratio
            #         sparsity_ratio = 0.55*pratio
            #         sparsity_ratio = 0.6159*pratio
            #         sparsity_ratio = 0.4906*pratio
            # if(i==5):
            #     if(('gate' in name)or('down' in name)):
            #         sparsity_ratio = 0.6056*pratio
            #         sparsity_ratio = 0.7545*pratio
            #         sparsity_ratio = 0.9779*pratio
            #         sparsity_ratio = 0.5312*pratio
            #     else:
            #         sparsity_ratio = 0.46*pratio 
            #         sparsity_ratio = 0.34*pratio 
            #         sparsity_ratio = 0.16*pratio
            #         sparsity_ratio = 0.52*pratio
            # if(i==6):# 111
            #     if(('gate' in name)):
            #         sparsity_ratio = 0.6934*pratio
            #         sparsity_ratio = 0.8327*pratio
            #         sparsity_ratio = 0.972*pratio
            #         sparsity_ratio = 0.6934*pratio
            #     else:
            #         sparsity_ratio = 0.68      *pratio    
            #         sparsity_ratio = 0.64      *pratio   
            #         sparsity_ratio = 0.6*pratio     
            #         sparsity_ratio = 0.68      *pratio                   
            # if(i==7):
            #     if(('gate' in name)or('up' in name)):
            #         sparsity_ratio = 0.5389 *pratio
            #         sparsity_ratio = 0.6878 *pratio
            #         sparsity_ratio = 0.9111*pratio
            #         sparsity_ratio = 0.4645      *pratio
            #     else:
            #         sparsity_ratio = 0.4*pratio
            #         sparsity_ratio = 0.28*pratio
            #         sparsity_ratio = 0.1*pratio
            #         sparsity_ratio = 0.46      *pratio
            # if(i==8):#1111
            #     if(('k' in name)or('q' in name)):
            #         sparsity_ratio = 0.7144*pratio
            #         sparsity_ratio = 0.8154*pratio
            #         sparsity_ratio = 0.9164*pratio
            #         sparsity_ratio = 0.7144*pratio
            #     else:
            #         sparsity_ratio = 0.66    *pratio
            #         sparsity_ratio = 0.64    *pratio
            #         sparsity_ratio = 0.62*pratio
            #         sparsity_ratio = 0.66    *pratio
            # if(i==9):
            #     if(('gate' in name)):
            #         sparsity_ratio = 0.6634*pratio
            #         sparsity_ratio = 0.733*pratio
            #         sparsity_ratio = 0.9419*pratio
            #         sparsity_ratio = 0.5938    *pratio
            #     else:
            #         sparsity_ratio = 0.52*pratio 
            #         sparsity_ratio = 0.50*pratio 
            #         sparsity_ratio = 0.44*pratio
            #         sparsity_ratio = 0.54    *pratio
            # if(i==10):
            #     if(('gate' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.5129 *pratio
            #         sparsity_ratio = 0.6116 *pratio
            #         sparsity_ratio = 0.7005*pratio
            #         sparsity_ratio = 0.503    *pratio
            #     else:
            #         sparsity_ratio = 0.48*pratio  
            #         sparsity_ratio = 0.28*pratio  
            #         sparsity_ratio = 0.1*pratio
            #         sparsity_ratio = 0.5    *pratio
            # if(i==11):
            #     if(('q' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.26*pratio
            #         sparsity_ratio = 0.30*pratio
            #         sparsity_ratio = 0.1*pratio
            #         sparsity_ratio = 0.5    *pratio
            #     else:
            #         sparsity_ratio = 0.5898     *pratio    
            #         sparsity_ratio = 0.5766     *pratio          
            #         sparsity_ratio = 0.6425*pratio
            #         sparsity_ratio = 0.5106    *pratio
            # if(i==12):
            #     if(('gate' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.5922*pratio
            #         sparsity_ratio = 0.5922*pratio
            #         sparsity_ratio = 0.6811*pratio
            #         sparsity_ratio = 0.4934    *pratio
            #     else:
            #         sparsity_ratio = 0.28*pratio 
            #         sparsity_ratio = 0.28*pratio 
            #         sparsity_ratio = 0.1*pratio 
            #         sparsity_ratio = 0.48    *pratio
            # if(i==13):
            #     if(('v' in name)or('o_proj' in name)or('down' in name)):
            #         sparsity_ratio = 0.42*pratio
            #         sparsity_ratio = 0.3*pratio
            #         sparsity_ratio = 0.1*pratio
            #         sparsity_ratio = 0.52    *pratio
            #     else:
            #         sparsity_ratio = 0.608  *pratio     
            #         sparsity_ratio = 0.6843  *pratio   
            #         sparsity_ratio = 0.8113*pratio  
            #         sparsity_ratio = 0.5445    *pratio                              
            # if(i==14):
            #     if(('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.1*pratio
            #         sparsity_ratio = 0.32*pratio
            #         sparsity_ratio = 0.1*pratio
            #         sparsity_ratio = 0.52    *pratio
            #     else:
            #         sparsity_ratio = 0.6199*pratio
            #         sparsity_ratio = 0.5764*pratio
            #         sparsity_ratio = 0.6199*pratio
            #         sparsity_ratio = 0.5368    *pratio
            # if(i==15):
            #     if(('gate' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.5165*pratio
            #         sparsity_ratio = 0.5857*pratio
            #         sparsity_ratio = 0.6647*pratio
            #         sparsity_ratio = 0.4869    *pratio
            #     else:
            #         sparsity_ratio = 0.4*pratio 
            #         sparsity_ratio = 0.26*pratio 
            #         sparsity_ratio = 0.1*pratio
            #         sparsity_ratio = 0.46    *pratio
            # if(i==16):
            #     if(('q' in name)or('k' in name)or('gate' in name)):
            #         sparsity_ratio = 0.7289*pratio
            #         sparsity_ratio = 0.8549*pratio
            #         sparsity_ratio = 0.9808*pratio
            #         sparsity_ratio = 0.7289*pratio
            #     else:
            #         sparsity_ratio = 0.68      *pratio  
            #         sparsity_ratio = 0.6    *pratio      
            #         sparsity_ratio = 0.52*pratio  
            #         sparsity_ratio = 0.68      *pratio                     
            # if(i==17):
            #     if(('gate' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.5678*pratio
            #         sparsity_ratio = 0.6666*pratio
            #         sparsity_ratio = 0.7752*pratio
            #         sparsity_ratio = 0.5579      *pratio  
            #     else:
            #         sparsity_ratio = 0.52*pratio
            #         sparsity_ratio = 0.32*pratio
            #         sparsity_ratio = 0.1*pratio
            #         sparsity_ratio = 0.54      *pratio  
            # if(i==18):
            #     if(('gate' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.5649*pratio
            #         sparsity_ratio = 0.6242*pratio
            #         sparsity_ratio = 0.7229*pratio
            #         sparsity_ratio = 0.5254      *pratio  
            #     else:
            #         sparsity_ratio = 0.42*pratio    
            #         sparsity_ratio = 0.3*pratio
            #         sparsity_ratio = 0.1*pratio
            #         sparsity_ratio = 0.5      *pratio  
            # if(i==19):
            #     if(('q' in name)or('k' in name)or('v' in name)):
            #         sparsity_ratio = 0.5*pratio
            #         sparsity_ratio = 0.32*pratio
            #         sparsity_ratio = 0.1*pratio
            #         sparsity_ratio = 0.54      *pratio  
            #     else:
            #         sparsity_ratio = 0.5771*pratio
            #         sparsity_ratio = 0.6365*pratio
            #         sparsity_ratio = 0.709*pratio
            #         sparsity_ratio = 0.5639      *pratio  
            # if(i==20):
            #     if(('gate' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.5822*pratio
            #         sparsity_ratio = 0.6316*pratio
            #         sparsity_ratio = 0.7304*pratio
            #         sparsity_ratio = 0.523      *pratio  
            #     else:
            #         sparsity_ratio = 0.4*pratio 
            #         sparsity_ratio = 0.3*pratio
            #         sparsity_ratio = 0.1*pratio
            #         sparsity_ratio = 0.52      *pratio  
            # if(i==21):
            #     if(('q' in name)or('k' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.38*pratio
            #         sparsity_ratio = 0.32*pratio
            #         sparsity_ratio = 0.1*pratio
            #         sparsity_ratio = 0.52      *pratio  
            #     else:
            #         sparsity_ratio = 0.5996*pratio
            #         sparsity_ratio = 0.6292*pratio
            #         sparsity_ratio = 0.7379*pratio
            #         sparsity_ratio = 0.5305      *pratio  
            # if(i==22):
            #     if(('q' in name)or('k' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.64*pratio
            #         sparsity_ratio = 0.46*pratio
            #         sparsity_ratio = 0.22*pratio
            #         sparsity_ratio = 0.74      *pratio  
            #     else:
            #         sparsity_ratio = 0.7924 *pratio
            #         sparsity_ratio = 0.8813*pratio
            #         sparsity_ratio = 0.9998*pratio
            #         sparsity_ratio = 0.743      *pratio  
            # if(i==23):#111
            #     if(('q' in name)or('k' in name)or('v' in name)):
            #         sparsity_ratio = 0.7445*pratio
            #         sparsity_ratio = 0.8659*pratio
            #         sparsity_ratio = 0.9872*pratio
            #         sparsity_ratio = 0.7445*pratio
            #     else:
            #         sparsity_ratio = 0.68       *pratio     
            #         sparsity_ratio = 0.64*pratio  
            #         sparsity_ratio = 0.6*pratio 
            #         sparsity_ratio = 0.68       *pratio                       
            # if(i==24):
            #     if(('o_proj' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.72*pratio
            #         sparsity_ratio = 0.62*pratio  
            #         sparsity_ratio = 0.5*pratio  
            #         sparsity_ratio = 0.72*pratio
            #     else:
            #         sparsity_ratio = 0.7455*pratio
            #         sparsity_ratio = 0.8578*pratio  
            #         sparsity_ratio = 0.9925*pratio  
            #         sparsity_ratio = 0.7455*pratio
            # if(i==25):#111
            #     if(('q' in name)or('k' in name)or('v' in name)or('o_proj' in name)or('gate' in name)or('up' in name)or('down' in name)):
            #         sparsity_ratio = 0.725*pratio
            #     else:
            #         sparsity_ratio = 0.0 *pratio
            # if(i==26):#111
            #     if(('gate' in name)or('up' in name)):
            #         sparsity_ratio = 0.7251*pratio
            #         sparsity_ratio = 0.8492*pratio    
            #         sparsity_ratio = 0.9981*pratio
            #         sparsity_ratio = 0.7003*pratio
            #     else:
            #         sparsity_ratio = 0.64    *pratio  
            #         sparsity_ratio = 0.54*pratio       
            #         sparsity_ratio = 0.42*pratio      
            #         sparsity_ratio = 0.66*pratio                        
            # if(i==27):#111
            #     if(('k' in name)or('gate' in name)or('down' in name)):
            #         sparsity_ratio = 0.62*pratio
            #         sparsity_ratio = 0.54*pratio  
            #         sparsity_ratio = 0.42*pratio  
            #         sparsity_ratio = 0.68*pratio
            #     else:
            #         sparsity_ratio = 0.7537*pratio
            #         sparsity_ratio = 0.8436*pratio  
            #         sparsity_ratio = 0.9783*pratio  
            #         sparsity_ratio = 0.6864*pratio
            # if(i==28):#111
            #     if(('k' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.9785*pratio
            #         sparsity_ratio = 0.9178*pratio  
            #         sparsity_ratio = 0.9785*pratio  
            #         sparsity_ratio = 0.7965*pratio
            #     else:
            #         sparsity_ratio = 0.68  *pratio
            #         sparsity_ratio = 0.7*pratio    
            #         sparsity_ratio = 0.68  *pratio
            #         sparsity_ratio = 0.74*pratio
            # if(i==29):#111
            #     if(('o_proj' in name)or('v' in name)):
            #         sparsity_ratio = 0.68 *pratio
            #         sparsity_ratio = 0.36*pratio   
            #         sparsity_ratio = 0.1*pratio 
            #         sparsity_ratio = 0.70*pratio
            #     else:
            #         sparsity_ratio = 0.7064 *pratio
            #         sparsity_ratio = 0.7697*pratio    
            #         sparsity_ratio = 0.8212*pratio
            #         sparsity_ratio = 0.7024*pratio
            # if(i==30):#111
            #     if(('q' in name)or('k' in name)or('v' in name)or('down' in name)):
            #         sparsity_ratio = 0.64*pratio
            #         sparsity_ratio = 0.54*pratio   
            #         sparsity_ratio = 0.38*pratio 
            #         sparsity_ratio = 0.70*pratio
            #     else:
            #         sparsity_ratio = 0.761*pratio
            #         sparsity_ratio = 0.8501*pratio  
            #         sparsity_ratio = 0.9926*pratio  
            #         sparsity_ratio = 0.7076*pratio
            # if(i==31):  #111
            #     if(('v' in name)or('k' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.942*pratio
            #         sparsity_ratio = 0.8813*pratio    
            #         sparsity_ratio = 0.942*pratio
            #         sparsity_ratio = 0.8207*pratio
            #     else:
            #         sparsity_ratio = 0.7*pratio
            #         sparsity_ratio = 0.72*pratio    
            #         sparsity_ratio = 0.7*pratio
            #         sparsity_ratio = 0.74*pratio
            # if(i==32):
            #     if(('q' in name)or('k' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.64*pratio
            #         sparsity_ratio = 0.36*pratio   
            #         sparsity_ratio = 0.1*pratio 
            #         sparsity_ratio = 0.66*pratio
            #     else:
            #         sparsity_ratio = 0.6892 *pratio
            #         sparsity_ratio = 0.7815*pratio    
            #         sparsity_ratio = 0.8672*pratio
            #         sparsity_ratio = 0.6826*pratio
            # if(i==33):
            #     if(('q' in name)or('k' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.66*pratio
            #         sparsity_ratio = 0.44*pratio  
            #         sparsity_ratio = 0.2*pratio  
            #         sparsity_ratio = 0.72*pratio
            #     else:
            #         sparsity_ratio = 0.7705    *pratio    
            #         sparsity_ratio = 0.8792*pratio  
            #         sparsity_ratio = 0.9977*pratio  
            #         sparsity_ratio = 0.7409*pratio                             
            # if(i==34):
            #     if(('q' in name)or('gate' in name)or('down' in name)):
            #         sparsity_ratio = 0.66*pratio
            #         sparsity_ratio = 0.60*pratio  
            #         sparsity_ratio = 0.48*pratio  
            #         sparsity_ratio = 0.72*pratio
            #     else:
            #         sparsity_ratio = 0.7916*pratio
            #         sparsity_ratio = 0.859*pratio    
            #         sparsity_ratio = 0.9937*pratio
            #         sparsity_ratio = 0.7242*pratio
            # if(i==35):
            #     if(('q' in name)or('k' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.68*pratio
            #         sparsity_ratio = 0.4*pratio  
            #         sparsity_ratio = 0.1*pratio  
            #         sparsity_ratio = 0.74*pratio
            #     else:
            #         sparsity_ratio = 0.7811 *pratio
            #         sparsity_ratio = 0.8734*pratio    
            #         sparsity_ratio = 0.9723*pratio
            #         sparsity_ratio = 0.7613*pratio
            # if(i==36):
            #     if(('gate' in name)or('q' in name)or('k' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.62*pratio
            #         sparsity_ratio = 0.54*pratio   
            #         sparsity_ratio = 0.4*pratio 
            #         sparsity_ratio = 0.66*pratio
            #     else:
            #         sparsity_ratio = 0.7208    *pratio     
            #         sparsity_ratio = 0.8201*pratio      
            #         sparsity_ratio = 0.9938*pratio    
            #         sparsity_ratio = 0.6712*pratio                    
            # if(i==37):
            #     if(('q' in name)or('k' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.44*pratio
            #         sparsity_ratio = 0.3*pratio   
            #         sparsity_ratio = 0.1*pratio 
            #         sparsity_ratio = 0.5*pratio
            #     else:
            #         sparsity_ratio = 0.5505*pratio
            #         sparsity_ratio = 0.6197*pratio  
            #         sparsity_ratio = 0.7184*pratio  
            #         sparsity_ratio = 0.5209*pratio
            # if(i==38):
            #     if(('down' in name)or('q' in name)or('k' in name)or('v' in name)or('o_proj' in name)):
            #         sparsity_ratio = 0.36*pratio
            #         sparsity_ratio = 0.3*pratio 
            #         sparsity_ratio = 0.1*pratio   
            #         sparsity_ratio = 0.48*pratio
            #     else:
            #         sparsity_ratio = 0.6423   *pratio
            #         sparsity_ratio = 0.7168*pratio    
            #         sparsity_ratio = 0.9649*pratio
            #         sparsity_ratio = 0.4934*pratio
            # if(i==39):
            #     if(('gate' in name)or('k' in name)or('q' in name)or('down' in name)):
            #         sparsity_ratio = 0.32*pratio
            #         sparsity_ratio = 0.3*pratio   
            #         sparsity_ratio = 0.14*pratio 
            #         sparsity_ratio = 0.46*pratio
            #     else:
            #         sparsity_ratio = 0.6856*pratio             
            #         sparsity_ratio = 0.7171*pratio    
            #         sparsity_ratio = 0.969*pratio
            #         sparsity_ratio = 0.4651*pratio
            
            
            
            
            
            # sparsity_ratio = args.sparsity_ratio
            
            
            
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            # 先屏蔽
            dev = torch.device('cpu')            
            # 先屏蔽
            W_metric = W_metric.to(dev)

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
                    mask = W_mask.bool().cpu()
                    weight_dict["layers."+str(i)+"."+name] = mask

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    # torch.save(weight_dict,  "/data/fhl/wanda/lora_ft/llama_7b05_wanda_weight_mask_allmodules.pth")


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()



@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None 

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()