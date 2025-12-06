import os
import sys
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

import nnsight
nnsight.CONFIG.API.APIKEY = "dummy"

from lib.models import create_model
from lib.datasets import GSM8K


def smart_resolve(obj):
    """Recursively extracts real values from NNsight Proxies."""
    if hasattr(obj, 'value'):
        obj = obj.value
    if isinstance(obj, (tuple, list)):
        return type(obj)(smart_resolve(x) for x in obj)
    return obj

def recursive_to_device(obj, device):
    """Recursively moves Tensors to device, ignores others."""
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, (tuple, list)):
        return type(obj)(recursive_to_device(x, device) for x in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to_device(v, device) for k, v in obj.items()}
    return obj




def plot_causal_effects(effects_matrix):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data_np = effects_matrix.numpy()
    
    
    mask = np.tril(np.ones_like(data_np, dtype=bool))
    data_masked = np.ma.array(data_np, mask=mask)
    
    
    im = ax.imshow(data_masked, cmap='viridis', interpolation="nearest", origin='upper')
                    
    plt.xlabel("Effect @ layer")
    plt.ylabel("Layer skipped")
    plt.title("Local effect of layer on later layers")
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, label='Relative change')
    return fig

def plot_logit_diffs(d):
    fig = plt.figure(figsize=(6,3))
    plt.bar([0], [d.detach().item()])
    plt.ylabel("Logit diff norm")
    return fig




def get_layers(llm):
    root = llm.model if hasattr(llm, "model") else llm
    if hasattr(root, "model") and hasattr(root.model, "layers"):
        return root.model.layers
    if hasattr(root, "layers"):
        return root.layers
    if hasattr(root, "transformer") and hasattr(root.transformer, "h"):
        return root.transformer.h
    available = list(root._modules.keys()) if hasattr(root, "_modules") else "Unknown"
    raise RuntimeError(f"Could not find model layers. Available submodules: {available}")




def get_causal_effects(llm, prompt):
    layers = get_layers(llm)
    n_layers = len(layers)
    
    
    inputs_saved = []
    outputs_saved = []

    with llm.trace(prompt):
        for layer in layers:
            inputs_saved.append(layer.inputs.save())
            outputs_saved.append(layer.output.save())
        logits = llm.output.logits.save()

    
    device = next(layers[0].parameters()).device
    
    
    inputs_resolved = [smart_resolve(x) for x in inputs_saved]
    outputs_resolved = [smart_resolve(x) for x in outputs_saved]
    
    
    
    full_inputs = [recursive_to_device(x, device) for x in inputs_resolved]
    full_outs = [recursive_to_device(x, device) for x in outputs_resolved]

    effect_matrix = torch.zeros((n_layers, n_layers))

    
    with torch.no_grad():
        for t in range(n_layers):
            
            
            
            args_t = full_inputs[t]
            if isinstance(args_t, torch.Tensor):
                args_t = (args_t,)
            
            x_t = args_t[0] 
            other_args = args_t[1:] 
            
            
            
            out_t_val = full_outs[t]
            if isinstance(out_t_val, tuple):
                out_t_val = out_t_val[0] 
            
            
            update_t_original = out_t_val - x_t
            norm_orig = update_t_original.norm() + 1e-6 

            
            for s in range(t): 
                args_s = full_inputs[s]
                if isinstance(args_s, torch.Tensor): args_s = (args_s,)
                x_s = args_s[0]
                out_s_val = full_outs[s]
                if isinstance(out_s_val, tuple): out_s_val = out_s_val[0]
                
                u_s = out_s_val - x_s
                x_t_ablated = x_t - u_s
                out_t_new = layers[t](x_t_ablated, *other_args)
                
                if isinstance(out_t_new, tuple):
                    out_t_new = out_t_new[0]
                
                update_t_new = out_t_new - x_t_ablated
                diff = (update_t_original - update_t_new).norm()
                score = diff / norm_orig
                
                effect_matrix[s, t] = score.item()

    return effect_matrix.cpu(), logits




def run(llm, model_name, dataset, suffix, n_examples):
    outdir = "out/causal_effects"
    os.makedirs(outdir, exist_ok=True)

    d_mean = None
    logits_max = torch.zeros([1])
    count = 0

    for idx, prompt in enumerate(tqdm(dataset, total=n_examples, desc=f"Processing {suffix}")):
        try:
            d, l_obj = get_causal_effects(llm, prompt)
            
            if d_mean is None:
                d_mean = d
            else:
                d_mean += d
            
            l = l_obj.softmax(-1).norm()
            logits_max = torch.max(logits_max, l.cpu())
            count += 1
            
        except Exception as e:
            
            print(f"\nSkipping prompt {idx} due to error: {repr(e)}")
            continue

        if idx + 1 >= n_examples:
            break
            
    if count > 0:
        d_mean = d_mean / count

        print(f"Saving plots to {outdir}...")
        
        fig = plot_causal_effects(d_mean)
        fig.savefig(os.path.join(outdir, f"{model_name}_{suffix}_causal.pdf"), bbox_inches="tight")
        
        fig = plot_logit_diffs(logits_max)
        fig.savefig(os.path.join(outdir, f"{model_name}_{suffix}_logits.pdf"), bbox_inches="tight")
    else:
        print("No successful runs.")




def main():
    random.seed(1234)
    model_name = "qwen2.5_1.5b" 
    llm = create_model(model_name)
    llm.eval()

    print("Running GSM8K Causal Analysis...")
    run(llm, model_name, GSM8K(), "gsm8k", 5) 

    print("DONE")

if __name__ == "__main__":
    main()