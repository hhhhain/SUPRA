import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers
from lib.eval import eval_ppl, eval_zero_shot
import re
from torch.profiler import profile, ProfilerActivity

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
        
    )

    # model.seqlen = model.config.max_position_embeddings 
    model.seqlen = 2048
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument("--sheet_name", type=str, help="sheet for storing")
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", 
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str ) # 这里指的是下载的权重应该存到哪，默认是下载到~/.cache里。
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')

    parser.add_argument("--eval_zero_shot", action="store_true")
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured" and args.sparsity_ratio != 0:
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    # print(model)
    # exit()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)






    s = args.save_model
    s_ori = s
    kwargs={}
    
    
    # if args.eval_zero_shot:
    #     print("Zero-shot evaluation is enabled.") 
    # else:  
    #     # 定义每个内容的正则表达式模式
    #     pattern_1 = r'12504(\w+)(?=\d\.\d+)'
    #     match_1 = re.search(pattern_1, s)
    #     if match_1:
    #         Module_1 = match_1.group(1)
    #         s = s[match_1.end():]
        
    #     pattern_2 = r'(\d\.\d+)'   
    #     match_2 = re.search(pattern_2, s)
    #     if match_2:
    #         Module_2 = match_2.group(1)
    #         s = s[match_2.end():]    
            
    #     pattern_3 = r'(\w+)(?=\d\.\d+)' 
    #     match_3 = re.search(pattern_3, s)
    #     if match_3:
    #         Module_3 = match_3.group(1)
    #         s = s[match_3.end():]     
        
    #     pattern_4 = r'(\d\.\d+)'
    #     match_4 = re.search(pattern_4, s)
    #     if match_4:
    #         Module_4 = match_4.group(1)
    #         s = s[match_4.end():]    

    #     pattern_5 = r'(\w+)(?=\d\.\d+)' 
    #     match_5 = re.search(pattern_5, s)
    #     if match_5:
    #         Module_5 = match_5.group(1)
    #         s = s[match_5.end():]     
        
    #     pattern_6 = r'(\d\.\d+)'
    #     match_6 = re.search(pattern_6, s)
    #     if match_6:
    #         Module_6 = match_6.group(1)
    #         s = s[match_6.end():]
            
    #     pattern_7 = r'(\w+)(?=\d\.\d+)' 
    #     match_7 = re.search(pattern_7, s)
    #     if match_7:
    #         Module_7 = match_7.group(1)
    #         s = s[match_7.end():]     
        
    #     pattern_8 = r'(\d\.\d+)'
    #     match_8 = re.search(pattern_8, s)
    #     if match_8:
    #         Module_8 = match_8.group(1)
    #         s = s[match_8.end():]
            
    #     pattern_9 = r'(\w+)(?=\d\.\d+)' 
    #     match_9 = re.search(pattern_9, s)
    #     if match_9:
    #         Module_9 = match_9.group(1)
    #         s = s[match_9.end():]     
        
    #     pattern_10 = r'(\d\.\d+)'
    #     match_10 = re.search(pattern_10, s)
    #     if match_10:
    #         Module_10 = match_10.group(1)
    #         s = s[match_10.end():]
            
    #     pattern_11 = r'(\w+)(?=\d\.\d+)' 
    #     match_11 = re.search(pattern_11, s)
    #     if match_11:
    #         Module_11 = match_11.group(1)
    #         s = s[match_11.end():]     
        
    #     pattern_12 = r'(\d\.\d+)'
    #     match_12 = re.search(pattern_12, s)
    #     if match_12:
    #         Module_12 = match_12.group(1)
    #         s = s[match_12.end():]
            
    #     pattern_13 = r'(\w+)(?=\d\.\d+)' 
    #     match_13 = re.search(pattern_13, s)
    #     if match_13:
    #         Module_13 = match_13.group(1)
    #         s = s[match_13.end():]     
        
    #     pattern_14 = r'(\d\.\d+)'
    #     match_14 = re.search(pattern_14, s)
    #     if match_14:
    #         Module_14 = match_14.group(1)
    #         s = s[match_14.end():]






        
    #     pattern_15 = r'(\d+)$'                 
    #     pattern_16 = r'12504(.*)' 
        
    #     match_15 = re.search(pattern_15, s)
    #     LayerNumber = match_15.group(1) if match_15 else None
        
    #     match_16 = re.search(pattern_16, s_ori)
    #     FullName = match_16.group(1) if match_16 else None
        
        
    #     kwargs = {Module_1: Module_2}
    #     kwargs[Module_3] = Module_4
    #     kwargs[Module_5] = Module_6
    #     kwargs[Module_7] = Module_8
    #     kwargs[Module_9] = Module_10
    #     kwargs[Module_11] = Module_12
    #     kwargs[Module_13] = Module_14
    #     kwargs['LayerNumber'] = LayerNumber
    #     kwargs['FullName'] = FullName
        

    #     # print(kwargs)
    #     # exit()






    if "7b" in args.model or "6.7b" in args.model:
        layer_count = 31  # 32层 - 1
    elif "8b" in args.model:
        layer_count = 31  # 假设32层 - 1
    elif "13b" in args.model:
        layer_count = 39  # 40层 - 1
    elif "30b" in args.model:
        layer_count = 59  # 60层 - 1
    elif "65b" in args.model or "70b" in args.model:
        layer_count = 79  # 80层 - 1
    elif "400b" in args.model:
        layer_count = -1  # 未知层数
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # 正则表达式模式，匹配n个浮点数
    pattern = fr"(?<=mag_based_opt_12504)\d*\.\d+(?:_\d*\.\d+){{{layer_count}}}"

    # 使用re.findall进行匹配
    matches = re.findall(pattern, s_ori)

    kwargs={}
    # 打印所有匹配的浮点数
    for match in matches:
        for index, number in enumerate(match.split('_')):
            kwargs[str(index)] = number
    # print(kwargs)
    # exit()





    





    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, **kwargs)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, **kwargs)
            # prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    
    if args.eval_zero_shot:
        print("Zero-shot evaluation is enabled.") 
    else:      
        datasets_user = ["wikitext2"]
        # datasets_user = ["wikitext2", "ptb", "c4"]
        # datasets_user = ["ptb"]
        # datasets_user = ["ptb", "c4"]
        for dataset1 in datasets_user:
            
            # # 使用 profiler 捕获执行过程
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            #             on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
            #             with_stack=True, record_shapes=True) as prof:
            
            #     ppl_test = eval_ppl(args, model, tokenizer, dataset1, device)
            #     print(f"{dataset1} perplexity: {ppl_test}")
                
            ppl_test = eval_ppl(args, model, tokenizer, dataset1, device)
            print(f"{dataset1} perplexity: {ppl_test}")         
        
        # ppl_test = eval_ppl(args, model, tokenizer, device)
        # print(f"wikitext perplexity {ppl_test}")


        


        if not os.path.exists(args.save):
            os.makedirs(args.save)
        save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
        with open(save_filepath, "w") as f:
            print("method\tactual_sparsity\tppl_test", file=f, flush=True)
            print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)

    if args.eval_zero_shot:
        accelerate=False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate=True

        task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)

    if args.save_model:
        pass
        # model.save_pretrained(args.save_model)
        # tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()