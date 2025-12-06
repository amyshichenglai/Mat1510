import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import os
import torch
import random
import nnsight
from tqdm import tqdm
from typing import Optional


nnsight.CONFIG.API.APIKEY = "dummy_key_local_only"
os.environ["NNSIGHT_API_KEY"] = "dummy"

from lib.models import create_model
from lib.nnsight_tokenize import tokenize
from lib.datasets import GSM8K, MQuake, LAMBADA, OpenEnded20
from lib.ndif_cache import ndif_cache_wrapper

N_EXAMPLES = 10

def plot_layer_diffs(dall):
    
    fig, ax = plt.subplots(figsize=(10, 10))

    data = dall.float().cpu().numpy()

    
    im = ax.imshow(
        data,
        vmin=0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal"   
    )

    
    ax.set_ylabel("Layer skipped")
    ax.set_xlabel("Effect @ layer")

    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, label="Relative change")

    
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    return fig

def plot_logit_diffs(dall):
    fig = plt.figure(figsize=(6,3))
    dall = dall.squeeze().float().cpu().numpy()
    plt.bar(list(range(dall.shape[0])), dall)
    plt.xlim(-1, dall.shape[0])
    plt.xlabel("Layer")
    plt.ylabel("Output change norm")
    return fig


def get_tensor(x):
    if hasattr(x, "value"):
        x = x.value
    if isinstance(x, tuple):
        return x[0]
    return x


def merge_io(intervened, orig, t: Optional[int] = None, no_skip_front: int = 1):
    if t is not None:
        return torch.cat([orig[:, :no_skip_front], intervened[:, no_skip_front:t], orig[:, t:]], dim=1)
    return torch.cat([orig[:, :no_skip_front], intervened[:, no_skip_front:]], dim=1)

def intervene_layer(layer, t, part, no_skip_front, baseline_inp=None, baseline_out=None):
    return merge_io(baseline_inp, baseline_out, t, no_skip_front)

@ndif_cache_wrapper
def test_effect(llm, prompt, positions, part, no_skip_front=1):
    """
    Optimized for Qwen2.5 local execution.
    Fixes:
    1. UnboundLocalError (initialized variables outside trace)
    2. OutOfOrderError (merged loops)
    3. Added inner progress bar
    """
    num_layers = len(llm.model.layers)
    
    baseline_inputs = []
    baseline_outputs = []
    baseline_logits = None
    
    with torch.no_grad():
        with llm.trace(prompt, remote=False) as tracer:
            for layer in llm.model.layers:
                baseline_inputs.append(layer.inputs[0].save())
                baseline_outputs.append(layer.output.save())
            baseline_logits = llm.output.logits.save()

    b_in = [get_tensor(x) for x in baseline_inputs]
    b_out = [get_tensor(x) for x in baseline_outputs]
    b_logits = get_tensor(baseline_logits)
    
    all_diffs_per_pos = []
    all_out_diffs_per_pos = []

    
    
    
    chunk_iterator = tqdm(positions, desc="  Chunks", leave=False, unit="chunk")
    
    for t in chunk_iterator:
        batched_prompts = [prompt] * num_layers
        intervention_tensors = []
        
        
        for i in range(num_layers):
            prefix = b_out[i][:, :no_skip_front]
            if part == "layer":
                replacement = b_in[i]
            else:
                replacement = torch.zeros_like(b_out[i])

            if t is not None:
                middle = replacement[:, no_skip_front:t]
                suffix = b_out[i][:, t:]
                merged = torch.cat([prefix, middle, suffix], dim=1)
            else:
                suffix = replacement[:, no_skip_front:]
                merged = torch.cat([prefix, suffix], dim=1)
            intervention_tensors.append(merged)

        stack = torch.cat(intervention_tensors, dim=0).to(llm.device)
        
        
        layer_effect_rows = []
        l_effect = None

        
        with torch.no_grad():
            with llm.trace(batched_prompts, remote=False) as tracer:
                
                
                for i, layer in enumerate(llm.model.layers):
                    
                    
                    if part == "layer":
                        layer.output[i] = stack[i].unsqueeze(0)
                    elif part == "mlp":
                        layer.mlp.output[i] = stack[i].unsqueeze(0)
                    elif part == "attention":
                        layer.self_attn.output[0][i] = stack[i].unsqueeze(0)

                    
                    diff = layer.output - b_out[i]
                    
                    if t is not None:
                        diff_future = diff[:, t:, :]
                        base_future = b_out[i][:, t:, :]
                    else:
                        diff_future = diff
                        base_future = b_out[i]
                    
                    d_norm = torch.linalg.norm(diff_future, dim=-1)
                    b_norm = torch.linalg.norm(base_future, dim=-1).clamp(min=1e-6)
                    
                    rel = d_norm / b_norm
                    max_rel = rel.max(dim=-1).values
                    layer_effect_rows.append(max_rel.save())

                
                final_logits = llm.output.logits
                p_intervened = final_logits.softmax(dim=-1)
                p_baseline = b_logits.softmax(dim=-1)
                
                if t is not None:
                    p_diff = p_intervened[:, t:, :] - p_baseline[:, t:, :]
                else:
                    p_diff = p_intervened - p_baseline
                
                l_effect = torch.linalg.norm(p_diff, dim=-1).max(dim=-1).values.save()

        
        
        if l_effect is None or len(layer_effect_rows) == 0:
            print(f"Warning: Trace failed for position {t}. Skipping.")
            continue
            
        res_matrix = torch.stack([get_tensor(x) for x in layer_effect_rows]).T
        l_effect_val = get_tensor(l_effect)
        
        all_diffs_per_pos.append(res_matrix.cpu())
        all_out_diffs_per_pos.append(l_effect_val.cpu())

    if len(all_diffs_per_pos) > 0:
        dall = torch.stack(all_diffs_per_pos).max(dim=0).values
        dall_out = torch.stack(all_out_diffs_per_pos).max(dim=0).values
    else:
        dall = torch.zeros(num_layers, num_layers)
        dall_out = torch.zeros(num_layers)

    return dall, dall_out

def test_future_max_effect(llm, prompt, N_CHUNKS=4, part="layer"):
    _, tokens = tokenize(llm, prompt)
    positions = list(range(8, len(tokens)-4, 8))
    random.shuffle(positions)
    positions = positions[:N_CHUNKS]
    return test_effect(llm, prompt, positions, part)

def run(llm, model_name, dataset, suffix):
    target_dir = "out/future_effects"
    os.makedirs(target_dir, exist_ok=True)

    for what in ["layer", "mlp", "attention"]:
        print(f"\n[Run] Processing part: {what}")
        dall = []
        d_max = torch.zeros([1])
        dout_max = torch.zeros([1])
        
        iterator = tqdm(enumerate(dataset), total=N_EXAMPLES, desc=f"{suffix}-{what}", unit="ex")
        
        for idx, prompt in iterator:
            if len(prompt) > 1024: prompt = prompt[:1024]
            
            try:
                diff_now, diff_out = test_future_max_effect(llm, prompt, part=what)
                
                if d_max.ndim == 1:
                    d_max = torch.zeros_like(diff_now)
                    dout_max = torch.zeros_like(diff_out)
                
                d_max = torch.max(d_max, diff_now)
                dout_max = torch.max(dout_max, diff_out)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    tqdm.write("OOM detected! Skipping.")
                    torch.cuda.empty_cache()
                else:
                    raise e

            if idx == N_EXAMPLES - 1:
                break

        fig = plot_layer_diffs(d_max)
        fig.savefig(os.path.join(target_dir, f"{model_name}_{suffix}_future_max_effect_{what}.pdf"), bbox_inches="tight")
        fig = plot_logit_diffs(dout_max)
        fig.savefig(os.path.join(target_dir, f"{model_name}_{suffix}_future_max_effect_out_{what}.pdf"), bbox_inches="tight")
        plt.close('all')

def main():
    if len(sys.argv) > 1: model_name = sys.argv[1]
    else: model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    print(f"[Main] Loading {model_name} locally...")
    llm = create_model(model_name, force_local=True)
    llm.eval()

    print("\n[Main] Running GSM8K …")
    run(llm, model_name, GSM8K(), "gsm8k")
    
    print("\n[Main] Running OpenEnded20 …")
    run(llm, model_name, OpenEnded20(), "openended20")

    print("\n[Main] Running OpenEnded20 …")
    run(llm, model_name, LAMBADA(), "lambda")

    print("\n[Main] Running OpenEnded20 …")
    run(llm, model_name, MQuake(), "mquake")
    
    print("\n[Main] DONE")

if __name__ == "__main__":
    main()