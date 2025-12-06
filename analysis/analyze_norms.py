from lib.matplotlib_config import sort_zorder
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os
import sys
import nnsight
nnsight.CONFIG.API.APIKEY = os.environ.get("NDIF_TOKEN", "dummy_key_for_local")
import torch
import torch.nn.functional as F
import random
import datasets
from nnsight import LanguageModel

from lib.models import create_model
from lib.nnsight_tokenize import tokenize
from lib.datasets import GSM8K, MQuake, LAMBADA, OpenEnded20
from lib.ndif_cache import ndif_cache_wrapper


N_EXAMPLES = 21

if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    raise ValueError("Please provide a model name")

llm = create_model(model_name)
target_dir = "out/norms"
os.makedirs(target_dir, exist_ok=True)

llm.eval()

@ndif_cache_wrapper
def analyze_norms(llm, prompts):
    n_layers = len(llm.model.layers)
    results = {}

    with llm.session(remote=llm.remote) as session:
        with torch.no_grad():
            res_norms_all = 0
            att_norms_all = 0
            mlp_norms_all = 0
            att_cos_acc = 0
            mlp_cos_acc = 0
            layer_cos_acc = 0
            layer_io_cos_acc = 0
            mean_rel_att_acc = 0
            mean_rel_mlp_acc = 0
            mean_rel_layer_acc = 0
            max_res_norms = torch.zeros(n_layers + 1)
            max_att_norms = torch.zeros(n_layers)
            max_mlp_norms = torch.zeros(n_layers)
            max_relative_contribution_att = torch.zeros(n_layers)
            max_relative_contribution_mlp = torch.zeros(n_layers)
            max_relative_contribution_layer = torch.zeros(n_layers)
            cnt = 0

            for i, prompt in enumerate(prompts):
                print(i)
                with llm.trace(prompt, remote=llm.remote):
                    layer_inputs = []
                    layer_outputs = []
                    att_outs_trace = []
                    mlp_outs_trace = []

                    for i_layer, layer in enumerate(llm.model.layers):
                        layer_inputs.append(layer.inputs[0][0].save())
                        att_outs_trace.append(layer.self_attn.output[0].save())
                        mlp_outs_trace.append(layer.mlp.output.save())
                        layer_outputs.append(layer.output[0].save())

                layer_inputs = [x.cpu().float() for x in layer_inputs]
                layer_outputs = [x.cpu().float() for x in layer_outputs]
                att_outs = [x.cpu().float() for x in att_outs_trace]
                mlp_outs = [x.cpu().float() for x in mlp_outs_trace]

                print(f"  Example {i}: Got {len(layer_inputs)} layer inputs, {len(layer_outputs)} layer outputs")

                att_outputs, mlp_outputs = [], []
                att_cos, mlp_cos = [], []
                layer_cos, layer_io_cos = [], []
                relative_contribution_att = []
                relative_contribution_mlp = []
                relative_contribution_layer = []

                for j in range(len(layer_inputs)):
                    layer_diff = layer_outputs[j] - layer_inputs[j]
                    att_approx = att_outs[j]
                    mlp_approx = mlp_outs[j]

                    att_outputs.append(att_approx)
                    mlp_outputs.append(mlp_approx)

                    relative_contribution_att.append(
                        att_approx.norm(dim=-1) / layer_inputs[j].norm(dim=-1).clamp(min=1e-6)
                    )
                    relative_contribution_mlp.append(
                        mlp_approx.norm(dim=-1) / layer_inputs[j].norm(dim=-1).clamp(min=1e-6)
                    )
                    relative_contribution_layer.append(
                        layer_diff.norm(dim=-1) / layer_inputs[j].norm(dim=-1).clamp(min=1e-6)
                    )

                    att_cos.append(F.cosine_similarity(att_approx, layer_inputs[j], dim=-1).mean(1))
                    mlp_cos.append(F.cosine_similarity(mlp_approx, layer_inputs[j], dim=-1).mean(1))
                    layer_cos.append(F.cosine_similarity(layer_diff, layer_inputs[j], dim=-1).mean(1))
                    layer_io_cos.append(F.cosine_similarity(layer_outputs[j], layer_inputs[j], dim=-1).mean(1))

                residual_log = [layer_inputs[0]] + layer_outputs

                r_norms = torch.stack([x.norm(dim=-1).mean() for x in residual_log])
                a_norms = torch.stack([x.norm(dim=-1).mean() for x in att_outputs])
                m_norms = torch.stack([x.norm(dim=-1).mean() for x in mlp_outputs])

                res_norms_all = res_norms_all + r_norms
                att_norms_all = att_norms_all + a_norms
                mlp_norms_all = mlp_norms_all + m_norms
                cnt += 1

                r_max = torch.stack([x.norm(dim=-1).max() for x in residual_log])
                a_max = torch.stack([x.norm(dim=-1).max() for x in att_outputs])
                m_max = torch.stack([x.norm(dim=-1).max() for x in mlp_outputs])

                max_res_norms = torch.maximum(max_res_norms, r_max)
                max_att_norms = torch.maximum(max_att_norms, a_max)
                max_mlp_norms = torch.maximum(max_mlp_norms, m_max)

                rel_att = torch.stack([x.mean() for x in relative_contribution_att])
                rel_mlp = torch.stack([x.mean() for x in relative_contribution_mlp])
                rel_layer = torch.stack([x.mean() for x in relative_contribution_layer])

                mean_rel_att_acc = mean_rel_att_acc + rel_att
                mean_rel_mlp_acc = mean_rel_mlp_acc + rel_mlp
                mean_rel_layer_acc = mean_rel_layer_acc + rel_layer

                max_relative_contribution_att = torch.maximum(
                    max_relative_contribution_att,
                    torch.stack([x.max() for x in relative_contribution_att])
                )
                max_relative_contribution_mlp = torch.maximum(
                    max_relative_contribution_mlp,
                    torch.stack([x.max() for x in relative_contribution_mlp])
                )
                max_relative_contribution_layer = torch.maximum(
                    max_relative_contribution_layer,
                    torch.stack([x.max() for x in relative_contribution_layer])
                )

                att_cos_acc = att_cos_acc + torch.stack([x.mean() for x in att_cos])
                mlp_cos_acc = mlp_cos_acc + torch.stack([x.mean() for x in mlp_cos])
                layer_cos_acc = layer_cos_acc + torch.stack([x.mean() for x in layer_cos])
                layer_io_cos_acc = layer_io_cos_acc + torch.stack([x.mean() for x in layer_io_cos])

            # Finalize outputs
            results['res_norms'] = (res_norms_all / cnt).cpu().clone()
            results['att_norms'] = (att_norms_all / cnt).cpu().clone()
            results['mlp_norms'] = (mlp_norms_all / cnt).cpu().clone()
            results['max_res_norms'] = max_res_norms.cpu().clone()
            results['max_att_norms'] = max_att_norms.cpu().clone()
            results['max_mlp_norms'] = max_mlp_norms.cpu().clone()
            results['mean_relative_contribution_att'] = (mean_rel_att_acc / cnt).cpu().clone()
            results['mean_relative_contribution_mlp'] = (mean_rel_mlp_acc / cnt).cpu().clone()
            results['mean_relative_contribution_layer'] = (mean_rel_layer_acc / cnt).cpu().clone()
            results['max_relative_contribution_att'] = max_relative_contribution_att.cpu().clone()
            results['max_relative_contribution_mlp'] = max_relative_contribution_mlp.cpu().clone()
            results['layer_cos_all'] = (layer_cos_acc / cnt).cpu().clone()
            results['att_cos_all'] = (att_cos_acc / cnt).cpu().clone()
            results['mlp_cos_all'] = (mlp_cos_acc / cnt).cpu().clone()
            results['layer_io_cos_all'] = (layer_io_cos_acc / cnt).cpu().clone()

    return (
        results['att_norms'], results['mlp_norms'], results['res_norms'],
        results['max_att_norms'], results['max_mlp_norms'], results['max_res_norms'],
        results['mean_relative_contribution_att'], results['mean_relative_contribution_mlp'],
        results['mean_relative_contribution_layer'], results['max_relative_contribution_att'],
        results['max_relative_contribution_mlp'], results['layer_cos_all'],
        results['att_cos_all'], results['mlp_cos_all'], results['layer_io_cos_all']
    )



###############################################################################
# 1) ========================= RUN FOR GSM8K ==================================
###############################################################################

prompts_gsm = []
for i, prompt in enumerate(GSM8K()):
    if i >= N_EXAMPLES:
        break
    prompts_gsm.append(prompt)

print("\nRunning norms for GSM8K …")
gsm_results = analyze_norms(llm, prompts_gsm)



###############################################################################
# 2) ======================= RUN FOR OPENENDED20 ===============================
###############################################################################

### === ADDED FOR OPENENDED20 ===
prompts_oe = []
for i, prompt in enumerate(OpenEnded20()):
    if i >= N_EXAMPLES:
        break
    prompts_oe.append(prompt)

print("\nRunning norms for OpenEnded20 …")
oe_results = analyze_norms(llm, prompts_oe)

prompts_lamb = []
for i, prompt in enumerate(LAMBADA()):
    if i >= N_EXAMPLES:
        break
    prompts_lamb.append(prompt)

print("\nRunning norms for Lambada …")
lamb_results = analyze_norms(llm, prompts_lamb)

prompts_mquake = []
for i, prompt in enumerate(MQuake()):
    if i >= N_EXAMPLES:
        break
    prompts_mquake.append(prompt)

print("\nRunning norms for mquake …")
mquake_results = analyze_norms(llm, prompts_mquake)



###############################################################################
# ========== PLOTTING (unchanged), repeated for BOTH DATASETS =================
###############################################################################

def make_plots(results, suffix):
    (
        att_norms, mlp_norms, res_norms,
        max_att_norms, max_mlp_norms, max_res_norms,
        mean_relative_contribution_att, mean_relative_contribution_mlp,
        mean_relative_contribution_layer,
        max_relative_contribution_att, max_relative_contribution_mlp,
        layer_cos_all, att_cos_all, mlp_cos_all, layer_io_cos_all
    ) = results

    W_SCALE = 0.2
    W_BAR = 1.1
    W = 6
    H = 3

    def set_xlim(l):
        plt.xlim(-0.5, l - 0.5)

    ############################################################################
    # ========== ORIGINAL COMMENTED-OUT PLOT BLOCKS (PRESERVED AS IS) ==========
    ############################################################################

    # plt.figure(figsize=(W,H))
    # bars = []
    # bars.append(plt.bar([x for x in range(len(llm.model.layers))],
    #                     att_norms.float().cpu().numpy(),
    #                     label="Attention: $||\\bm{a}_l||_2$", width=W_BAR))
    # bars.append(plt.bar([x for x in range(len(llm.model.layers))],
    #                     mlp_norms.float().cpu().numpy(),
    #                     label="MLP: $||\\bm{m}_l||_2$", width=W_BAR))
    # bars.append(plt.bar([x for x in range(len(llm.model.layers))],
    #                     res_norms[:-1].float().cpu().numpy(),
    #                     label="Residual: $||\\bm{h}_{l}||_2$", width=W_BAR))
    # plt.xlabel("Layer index ($l$)")
    # plt.ylabel("Mean Norm")
    # plt.legend()
    # sort_zorder(bars)
    # set_xlim(len(llm.model.layers))
    # plt.savefig(os.path.join(target_dir, f"{model_name}_{suffix}_mean_norms.pdf"),
    #             bbox_inches="tight")

    # plt.figure(figsize=(W,H))
    # bars = []
    # bars.append(plt.bar(range(len(llm.model.layers)),
    #                     max_att_norms.float().cpu().numpy(),
    #                     label="Attention $\\bm{a}_l$", width=W_BAR))
    # bars.append(plt.bar(range(len(llm.model.layers)),
    #                     max_mlp_norms.float().cpu().numpy(),
    #                     label="MLP $\\bm{m}_l$", width=W_BAR))
    # bars.append(plt.bar(range(len(llm.model.layers)),
    #                     max_res_norms[:-1].float().cpu().numpy(),
    #                     label="Residual $\\bm{h}_{l}$", width=W_BAR))
    # plt.xlabel("Layer index ($l$)")
    # plt.ylabel("Max Norm")
    # plt.legend()
    # sort_zorder(bars)
    # set_xlim(len(llm.model.layers))
    # plt.savefig(os.path.join(target_dir, f"{model_name}_{suffix}_max_norms.pdf"),
    #             bbox_inches="tight")

    ############################################################################
    # ======================= ACTIVE PLOTS (UNCOMMENTED) =======================
    ############################################################################

    # === Mean Relative Contribution ===
    plt.figure(figsize=(W,H))
    bars = []
    bars.append(plt.bar(range(len(llm.model.layers)),
                        mean_relative_contribution_att.cpu().numpy(),
                        label="Attention: $||\\bm{a}_l||_2/||\\bm{h}_l||_2$", width=W_BAR))
    bars.append(plt.bar(range(len(llm.model.layers)),
                        mean_relative_contribution_mlp.cpu().numpy(),
                        label="MLP: $||\\bm{m}_l||_2/||\\bm{h}_l||_2$", width=W_BAR))
    bars.append(plt.bar(range(len(llm.model.layers)),
                        mean_relative_contribution_layer.cpu().numpy(),
                        label="Attention + MLP: $||\\bm{a}_l + \\bm{m}_l||_2/||\\bm{h}_{l}||_2$",
                        width=W_BAR))

    plt.legend()
    sort_zorder(bars)
    set_xlim(len(llm.model.layers))

    if max(
        mean_relative_contribution_att.max().item(),
        mean_relative_contribution_mlp.max().item(),
        mean_relative_contribution_layer.max().item()
    ) > 1.5:
        plt.ylim(0, 1.5)

    plt.xlabel("Layer index ($l$)")
    plt.ylabel("Mean Relative Contribution")
    plt.savefig(os.path.join(target_dir,
                f"{model_name}_{suffix}_mean_relative_contribution.pdf"),
                bbox_inches="tight")

    ############################################################################
    # ===================== ORIGINAL COMMENTED PC BLOCKS =======================
    ############################################################################

    # plt.figure(figsize=(W,H))
    # bars = []
    # bars.append(plt.bar(range(len(llm.model.layers)),
    #                     max_relative_contribution_att.cpu().numpy(),
    #                     label="Attention $\\bm{a}_l$", width=W_BAR))
    # bars.append(plt.bar(range(len(llm.model.layers)),
    #                     max_relative_contribution_mlp.cpu().numpy(),
    #                     label="MLP $\\bm{m}_l$", width=W_BAR))
    # plt.ylim(0, 2)
    # plt.xlabel("Layer index ($l$)")
    # plt.ylabel("Max Relative Contribution")
    # plt.legend()
    # sort_zorder(bars)
    # set_xlim(len(llm.model.layers))
    # plt.savefig(os.path.join(target_dir,
    #             f"{model_name}_{suffix}_max_relative_contribution.pdf"),
    #             bbox_inches="tight")

    ############################################################################
    # ======================== Cosine Similarity Plot ==========================
    ############################################################################

    plt.figure(figsize=(W,H))
    bars = []
    bars.append(plt.bar(range(len(llm.model.layers)),
                        att_cos_all.cpu().numpy(),
                        label="Attention: $\\text{cossim}(\\bm{a}_l, \\bm{h}_l)$",
                        width=W_BAR))
    bars.append(plt.bar(range(len(llm.model.layers)),
                        mlp_cos_all.cpu().numpy(),
                        label="MLP: $||\\bm{m}_l||_2/||\\bm{h}_l||_2$",
                        width=W_BAR))
    bars.append(plt.bar(range(len(llm.model.layers)),
                        layer_cos_all.cpu().numpy(),
                        label="Attention + MLP: $\\text{cossim}(\\bm{a}_l + \\bm{m}_l, \\bm{h}_l)$",
                        width=W_BAR))

    plt.xlabel("Layer index ($l$)")
    plt.ylabel("Cosine similarity")
    plt.legend()
    sort_zorder(bars)
    set_xlim(len(llm.model.layers))
    plt.savefig(os.path.join(target_dir, 
                f"{model_name}_{suffix}_avg_cossims.pdf"),
                bbox_inches="tight")

    ############################################################################
    # ======================= Original I/O Cosine Plot =========================
    ############################################################################

    # plt.figure(figsize=(W,H))
    # plt.bar(range(len(llm.model.layers)),
    #         layer_io_cos_all.cpu().numpy(),
    #         label="Attention + MLP $\\bm{a}_l + \\bm{m}_l$")
    # plt.xlabel("Layer index ($l$)")
    # plt.ylabel("Cosine similarity")
    # set_xlim(len(llm.model.layers))
    # plt.savefig(os.path.join(target_dir,
    #             f"{model_name}_{suffix}_avg_io_cossims.pdf"),
    #             bbox_inches="tight")




###############################################################################
# ========================= GENERATE PLOTS ====================================
###############################################################################

make_plots(gsm_results, "gsm8k")
make_plots(oe_results, "openended20")
make_plots(lamb_results, "lambada")
make_plots(mquake_results, "mquake_results")

print("\nDONE")
