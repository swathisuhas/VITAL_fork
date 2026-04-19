from __future__ import division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import copy
import collections
import os
import random

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as Tr
from torchvision.models.feature_extraction import create_feature_extractor

from utils.utils import lr_cosine_policy, clip, denormalize
from utils.lrp import LRPModel, LRPModelRestricted
from utils.guided_backprop import LayerGuidedBackprop


def split_network(model, req_name):
    layers = []
    feat_ext = []

    def get_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layers and layers[-1] == req_name:
                    break
                if layer is None:
                    continue
                if layer.__class__.__name__ == 'Sequential':
                    get_layers(layer, prefix=prefix + [name])
                else:
                    layers.append("_".join(prefix + [name]))
                    feat_ext.append(layer)

    get_layers(model)
    return torch.nn.Sequential(*feat_ext)


def sort_matching(target, input):
    B, C, W, H = target.size()
    input = input.squeeze(0)
    _, index_content = torch.sort(input.view(C, -1))
    inverse_index = index_content.argsort(-1)
    value_style, _ = torch.sort(target.view(B, C, -1))
    value_style = value_style.mean(0)
    mse_loss = (input.view(C, -1) - value_style.gather(-1, inverse_index)) ** 2
    return mse_loss.unsqueeze(0)


def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (
        (diff1.abs() / 255.0).mean()
        + (diff2.abs() / 255.0).mean()
        + (diff3.abs() / 255.0).mean()
        + (diff4.abs() / 255.0).mean()
    )
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2


def obtain_real_imgs(channel, num_real_img, main_dir):
    normalize = Tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transforms_all = Tr.Compose([
        Tr.Resize(256),
        Tr.CenterCrop(224),
        Tr.PILToTensor(),
        Tr.ConvertImageDtype(torch.float),
        normalize,
    ])

    txt_file = os.path.join(main_dir, str(channel), 'topk_files.txt')
    with open(txt_file) as f:
        datafiles = [line.rstrip() for line in f]

    datafiles = list(dict.fromkeys(datafiles))

    img_real = []
    inc_img = 0
    for file in datafiles:
        if inc_img == num_real_img:
            break
        img = Image.open(file)
        num_channel = len(img.split())
        if num_channel != 3:
            continue
        inc_img += 1
        img_p = transforms_all(img)
        img_real.append(img_p)

    img_real = torch.stack(img_real, 0).to('cuda')
    return img_real


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def check_format(arr):
    if isinstance(arr, torch.Tensor):
        arr = to_numpy(arr)
    if arr.shape[0] == 3:
        return np.moveaxis(arr, 0, -1)
    return arr


def normalize(image):
    image = np.array(image, dtype=np.float32)
    image -= image.min()
    image /= (image.max() + 1e-12)
    return image


def clip_percentile(img, percentile=0.1):
    return np.clip(img, np.percentile(img, percentile), np.percentile(img, 100 - percentile))


def save_maco(image, alpha, percentile_image=1.0, percentile_alpha=95, filename='image.png'):
    image, alpha = check_format(image), check_format(alpha)
    image = clip_percentile(image, percentile_image)
    image = normalize(image)

    alpha = np.mean(alpha, -1, keepdims=True)
    alpha = np.clip(alpha, None, np.percentile(alpha, percentile_alpha))
    alpha = alpha / alpha.max()

    image_np = np.concatenate([image, alpha], -1)
    pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
    pil_image.save(filename)


# ----------------------------
# Graph helpers on layer3 map
# ----------------------------

def visualize_points(iter_idx, syn_img, ref_img, syn_pts, ref_pts, save_dir):
    def _prepare_img(tensor):
        img = tensor.detach().cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img

    img_s = _prepare_img(syn_img)
    img_r = _prepare_img(ref_img)

    H, W, _ = img_s.shape

    def _map_pts(pts):
        pts = pts.detach().cpu().numpy()
        xs = (pts[:, 0] + 1) * 0.5 * (W - 1)
        ys = (pts[:, 1] + 1) * 0.5 * (H - 1)
        return xs, ys

    sx, sy = _map_pts(syn_pts)
    rx, ry = _map_pts(ref_pts)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img_s)
    axs[0].scatter(sx, sy, c='red', s=80, marker='x', linewidths=2)
    axs[0].set_title(f"Synthetic (Iter {iter_idx})")
    axs[0].axis('off')

    axs[1].imshow(img_r)
    axs[1].scatter(rx, ry, c='lime', s=60, marker='o', facecolors='none', linewidths=2)
    axs[1].set_title(f"Best Real Match (Iter {iter_idx})")
    axs[1].axis('off')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"viz_points_{iter_idx:05d}.png"), dpi=100)
    plt.close()


def _grid_positions(h, w, device):
    ys = torch.linspace(-1, 1, steps=h, device=device)
    xs = torch.linspace(-1, 1, steps=w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([xx, yy], dim=-1).view(-1, 2)


def _flatten_feats_positions(feat_map):
    # feat_map: [B, C, H, W]
    B, C, H, W = feat_map.shape
    feats = feat_map.permute(0, 2, 3, 1).reshape(B, H * W, C)
    pos = _grid_positions(H, W, feat_map.device).unsqueeze(0).repeat(B, 1, 1)
    return feats, pos, (H, W)


def _build_graph_from_featmap_importance(feat_map, K_parts=8):
    feats_flat, pos_flat, _ = _flatten_feats_positions(feat_map)
    feats_flat = feats_flat[0]
    pos_flat = pos_flat[0]

    f = feats_flat
    f_std = (f - f.mean(0, keepdim=True)) / (f.std(0, keepdim=True) + 1e-6)

    N, C = f_std.shape
    K_eff = min(K_parts, N)

    # Important: build graph points from LRP-/GBP-weighted layer3 embedding magnitude.
    mags = f.norm(p=2, dim=1)
    _, top_indices = torch.topk(mags, k=K_eff)

    nodes_f = f_std[top_indices]
    nodes_p = pos_flat[top_indices]

    diffs = nodes_p.unsqueeze(1) - nodes_p.unsqueeze(0)
    D = torch.linalg.norm(diffs + 1e-8, dim=-1)
    med = torch.median(D[D > 0]) if (D > 0).any() else D.new_tensor(1.0)
    Dnorm = D / (med + 1e-6)
    Ang = torch.atan2(diffs[..., 1], diffs[..., 0])

    return {
        "nodes_f": nodes_f,
        "nodes_p": nodes_p,
        "nodes_p_raw": pos_flat[top_indices],
        "Dnorm": Dnorm,
        "Ang": Ang,
    }


def _sinkhorn(C, eps=0.05, iters=50, a=None, b=None):
    K = torch.exp(-C / max(eps, 1e-6)) + 1e-9
    n, m = K.shape
    if a is None:
        a = torch.full((n,), 1.0 / n, device=K.device)
    if b is None:
        b = torch.full((m,), 1.0 / m, device=K.device)

    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(iters):
        u = a / (K @ v + 1e-8)
        v = b / (K.t() @ u + 1e-8)

    return torch.diag(u) @ K @ torch.diag(v)


def graph_matching_loss(
    syn_graph,
    ref_graph,
    sink_eps=0.005,
    sink_iters=200,
):
    # ONLY edge-length matching (no nodes, no angles)
    f_s, p_s, D_s = syn_graph["nodes_f"], syn_graph["nodes_p"], syn_graph["Dnorm"]
    f_r, p_r, D_r = ref_graph["nodes_f"], ref_graph["nodes_p"], ref_graph["Dnorm"]

    f_s_n = F.normalize(f_s, dim=-1)
    f_r_n = F.normalize(f_r, dim=-1)

    # feature cost only used for correspondence (not optimized directly)
    C_feat = 1.0 - (f_s_n @ f_r_n.t())
    sf = C_feat.median()
    C = C_feat / (sf + 1e-6)

    Pi = _sinkhorn(C, eps=sink_eps, iters=sink_iters)
    mass = Pi.sum() + 1e-8

    # EDGE LENGTH MATCHING ONLY
    Ds = D_s.unsqueeze(1).unsqueeze(3)
    Dr = D_r.unsqueeze(0).unsqueeze(2)
    diff2 = (Ds - Dr) ** 2

    Pij = Pi.unsqueeze(2).unsqueeze(3)
    Pkl = Pi.unsqueeze(0).unsqueeze(1)

    L_edges_dist = (diff2 * Pij * Pkl).sum() / (mass ** 2)

    return L_edges_dist


def softmin_over_refs(losses, tau=0.5):
    return -tau * torch.logsumexp(-losses / max(tau, 1e-6), dim=0)


class DeepFeaturesClass(object):
    def __init__(self,
                 model=None,
                 parameters=dict(),
                 coefficients=dict(),
                 exp_name=None,
                 folder_name=None):

        self.model = model

        self.image_resolution = parameters["resolution"]
        self.do_flip = parameters["do_flip"]
        self.setting_id = parameters["setting_id"]
        self.bs = parameters["bs"]
        self.jitter = parameters["jitter"]
        self.num_real_img = parameters["num_real_img"]
        self.epochs = parameters['epochs']
        self.channel = parameters["channel"]
        self.layer = parameters["layer"]
        self.arch_name = parameters['arch_name']
        self.method = parameters["method"]
        self.topk_dir = parameters["topk_dir"]

        self.print_every = 100

        self.var_scale_l1 = coefficients["tv_l1"]
        self.var_scale_l2 = coefficients["tv_l2"]
        self.l2_scale = coefficients["l2"]
        self.lr = coefficients["lr"]
        self.feat_dist = coefficients["feat_dist"]
        self.layer_weights = coefficients["layer_weights"]

        # Graph loss is applied on layer3 embeddings, while channel target stays in layer4.
        self.enable_graph = coefficients.get("enable_graph", False)
        self.graph_scale = coefficients.get("graph_scale", 0.001)
        self.graph_K = coefficients.get("graph_K", 10)
        self.graph_num_refs = coefficients.get("graph_num_refs", 8)
        self.graph_sink_eps = coefficients.get("graph_sink_eps", 0.005)
        self.graph_sink_iters = coefficients.get("graph_sink_iters", 30)
        self.graph_tau = coefficients.get("graph_tau", 0.5)

        self.num_generations = 0
        self.exp_name = exp_name
        self.folder_name = folder_name

    def get_images(self):
        print("get_images call")

        model = self.model
        model_split = split_network(copy.deepcopy(model), self.layer)
        model_split.eval()

        skip_connection_prop = "flows_skip"
        rel_pass_ratio = 1.0

        print_every = self.print_every
        img_original = self.image_resolution
        data_type = torch.float
        pooling_function = nn.modules.pooling.AvgPool2d(kernel_size=2)

        skipfirst = False if self.setting_id == 0 else True

        if 'resnet' in self.arch_name:
            return_nodes = {
                "conv1": "conv1",
                "layer1": "layer1",
                "layer2": "layer2",
                "layer3": "layer3",
                "layer4": "layer4",
                "fc": "fc"
            }

            # Keep LRP outputs through layer3. Target channel is still from layer4.
            return_nodes_lrp = {
                "fdim": [1024, 512, 256, 64],
                "res": [14, 28, 56, 112]
            }

            target_layers = {
                "layer3": model_split[16],
                "layer2": model_split[10],
                "layer1": model_split[6],
                "conv1": model_split[0]
            }
        elif 'densenet' in self.arch_name:
            return_nodes = {
                "features.conv0": "conv1",
                "features.denseblock1": "layer1",
                "features.denseblock2": "layer2",
                "features.denseblock3": "layer3",
                "features.denseblock4": "layer4",
                "classifier": "fc"
            }
            return_nodes_lrp = {
                "fdim": [1024, 512, 256, 64],
                "res": [14, 28, 56, 112]
            }
        else:
            raise Exception("Not Implemented Yet!")

        model2 = create_feature_extractor(copy.deepcopy(model), return_nodes=return_nodes)
        model2.eval()

        img_real = obtain_real_imgs(
            channel=self.channel,
            num_real_img=self.num_real_img,
            main_dir=self.topk_dir
        )

        inputs = torch.randn(
            (self.bs, 3, img_original, img_original),
            requires_grad=True,
            device='cuda',
            dtype=data_type
        )

        transparency_accumulator = torch.zeros(
            (self.bs, 3, img_original, img_original),
            device='cuda',
            dtype=data_type
        )

        iteration = 0
        for lr_it, lower_res in enumerate([2, 1]):
            if lr_it == 0:
                iterations_per_layer = 2000
            else:
                iterations_per_layer = 1000 if not skipfirst else self.epochs
                if self.setting_id == 2:
                    iterations_per_layer = 20000

            if lr_it == 0 and skipfirst:
                continue

            lim_0, lim_1 = self.jitter // lower_res, self.jitter // lower_res

            if self.setting_id == 0:
                optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, 0.9], eps=1e-8)
                do_clip = True
            elif self.setting_id == 1:
                optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, 0.9], eps=1e-8)
                do_clip = True
            elif self.setting_id == 2:
                optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.9, 0.999], eps=1e-8)
                do_clip = False

            lr_scheduler = lr_cosine_policy(self.lr, 100, iterations_per_layer)

            for iteration_loc in range(iterations_per_layer):
                iteration += 1
                lr_scheduler(optimizer, iteration_loc, iteration_loc)

                if lower_res != 1:
                    inputs_jit = pooling_function(inputs)
                    img_real_jit = pooling_function(img_real)
                else:
                    img_real_jit = img_real
                    inputs_jit = inputs

                off1 = random.randint(-lim_0, lim_0)
                off2 = random.randint(-lim_1, lim_1)
                inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

                flip = random.random() > 0.5
                if flip and self.do_flip:
                    inputs_jit = torch.flip(inputs_jit, dims=(3,))

                optimizer.zero_grad()
                img_real_jit.requires_grad = True

                syn_out = model2(inputs_jit)

                if iteration == 1:
                    real_out = model2(img_real_jit)

                    if self.method == 'LRP':
                        lrp_model_real = LRPModel(
                            model_split,
                            rel_pass_ratio=rel_pass_ratio,
                            skip_connection_prop=skip_connection_prop
                        )
                        attention_real = lrp_model_real.forward(
                            img_real_jit,
                            channel=self.channel,
                            return_nodes=copy.deepcopy(return_nodes_lrp)
                        )
                    elif self.method == 'LRPRestricted':
                        lrp_model_real = LRPModelRestricted(
                            model_split,
                            rel_pass_ratio=rel_pass_ratio,
                            skip_connection_prop=skip_connection_prop
                        )
                        attention_real = lrp_model_real.forward(
                            img_real_jit,
                            channel=self.channel,
                            return_nodes=copy.deepcopy(return_nodes_lrp)
                        )
                    elif self.method == 'GuidedBackprop':
                        assert "resnet" in self.arch_name
                        guided_bp = LayerGuidedBackprop(model_split)
                        attention_real = guided_bp.attribute(
                            inputs=img_real_jit,
                            target_layers=target_layers,
                            target=self.channel
                        )
                    else:
                        raise ValueError("Method not supported")

                    importance_neuron = []
                    for att_id in range(len(attention_real)):
                        imp_neuron = attention_real[att_id].amax((2, 3))
                        imp_neuron = imp_neuron - imp_neuron.amin(1, keepdim=True)
                        imp_neuron = imp_neuron / (imp_neuron.amax(1, keepdim=True) + 1e-6)
                        importance_neuron.append(imp_neuron)

                        attention_real[att_id] = attention_real[att_id] - attention_real[att_id].amin((2, 3), keepdim=True)
                        attention_real[att_id] = attention_real[att_id] / (attention_real[att_id].amax((2, 3), keepdim=True) + 1e-6)

                    attention_conv1 = attention_real[3] * real_out["conv1"]
                    attention_layer1 = attention_real[2] * real_out["layer1"]
                    attention_layer2 = attention_real[1] * real_out["layer2"]
                    attention_layer3 = attention_real[0] * real_out["layer3"]

                if self.method == 'LRP':
                    lrp_model_syn = LRPModel(
                        model_split,
                        rel_pass_ratio=rel_pass_ratio,
                        skip_connection_prop=skip_connection_prop
                    )
                    attention_syn = lrp_model_syn.forward(
                        inputs_jit,
                        channel=self.channel,
                        return_nodes=copy.deepcopy(return_nodes_lrp)
                    )
                elif self.method == 'LRPRestricted':
                    lrp_model_syn = LRPModelRestricted(
                        model_split,
                        rel_pass_ratio=rel_pass_ratio,
                        skip_connection_prop=skip_connection_prop
                    )
                    attention_syn = lrp_model_syn.forward(
                        inputs_jit,
                        channel=self.channel,
                        return_nodes=copy.deepcopy(return_nodes_lrp)
                    )
                elif self.method == 'GuidedBackprop':
                    assert "resnet" in self.arch_name
                    guided_bp = LayerGuidedBackprop(model_split)
                    attention_syn = guided_bp.attribute(
                        inputs=inputs_jit,
                        target_layers=target_layers,
                        target=self.channel
                    )
                else:
                    raise ValueError("Method not supported")

                for att_id in range(len(attention_syn)):
                    attention_syn[att_id] = attention_syn[att_id] - attention_syn[att_id].amin((2, 3), keepdim=True)
                    attention_syn[att_id] = attention_syn[att_id] / (attention_syn[att_id].amax((2, 3), keepdim=True) + 1e-6)

                attention_syn_conv1 = attention_syn[3] * syn_out["conv1"]
                attention_syn_layer1 = attention_syn[2] * syn_out["layer1"]
                attention_syn_layer2 = attention_syn[1] * syn_out["layer2"]
                attention_syn_layer3 = attention_syn[0] * syn_out["layer3"]

                loss_conv1 = sort_matching(input=attention_syn_conv1, target=attention_conv1) * importance_neuron[3].unsqueeze(-1)
                loss_layer1 = sort_matching(input=attention_syn_layer1, target=attention_layer1) * importance_neuron[2].unsqueeze(-1)
                loss_layer2 = sort_matching(input=attention_syn_layer2, target=attention_layer2) * importance_neuron[1].unsqueeze(-1)
                loss_layer3 = sort_matching(input=attention_syn_layer3, target=attention_layer3) * importance_neuron[0].unsqueeze(-1)

                loss_conv1 = loss_conv1.mean()
                loss_layer1 = loss_layer1.mean()
                loss_layer2 = loss_layer2.mean()
                loss_layer3 = loss_layer3.mean()

                loss_add = (
                    self.layer_weights[0] * loss_conv1
                    + self.layer_weights[1] * loss_layer1
                    + self.layer_weights[2] * loss_layer2
                    + self.layer_weights[3] * loss_layer3
                )

                # -----------------------------
                # Graph matching on layer3 only
                # Channel target stays in layer4
                # -----------------------------
                loss_graph = torch.tensor(0.0, device=inputs.device)
                graph_losses = []

                if self.enable_graph:
                    num_refs = min(self.graph_num_refs, attention_layer3.size(0))
                    ref_indices = random.sample(range(attention_layer3.size(0)), k=num_refs)

                    syn_graph = _build_graph_from_featmap_importance(
                        attention_syn_layer3[0:1],
                        K_parts=self.graph_K
                    )

                    best_loss_val = float('inf')
                    best_real_pts_raw = None
                    best_ref_img_idx = None

                    for r_idx in ref_indices:
                        ref_graph = _build_graph_from_featmap_importance(
                            attention_layer3[r_idx:r_idx + 1],
                            K_parts=self.graph_K
                        )

                        Lg = graph_matching_loss(
                            syn_graph,
                            ref_graph,
                            sink_eps=self.graph_sink_eps,
                            sink_iters=self.graph_sink_iters,
                        )
                        graph_losses.append(Lg)

                        if Lg.item() < best_loss_val:
                            best_loss_val = Lg.item()
                            best_real_pts_raw = ref_graph["nodes_p_raw"]
                            best_ref_img_idx = r_idx

                    if len(graph_losses) > 0:
                        loss_graph = softmin_over_refs(torch.stack(graph_losses), tau=self.graph_tau)

                    if iteration % print_every == 0 and best_real_pts_raw is not None:
                        visualize_points(
                            iter_idx=iteration,
                            syn_img=inputs_jit[0].detach(),
                            ref_img=img_real_jit[best_ref_img_idx].detach(),
                            syn_pts=syn_graph["nodes_p_raw"],
                            ref_pts=best_real_pts_raw,
                            save_dir=os.path.join(self.folder_name, "points_debug")
                        )

                loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)
                loss_l2 = torch.norm(inputs_jit.view(self.bs, -1), dim=1).mean()

                loss = (
                    self.var_scale_l2 * loss_var_l2
                    + self.var_scale_l1 * loss_var_l1
                    + self.l2_scale * loss_l2
                    + self.feat_dist * loss_add
                    + self.graph_scale * loss_graph
                )

                grad_input = torch.autograd.grad(
                    loss,
                    inputs,
                    grad_outputs=torch.ones_like(loss),
                    retain_graph=True
                )[0]
                transparency_accumulator += torch.abs(grad_input)

                if iteration % print_every == 0:
                    print("------------iteration {}----------".format(iteration))
                    print("total loss", loss.item())
                    print("feat loss", loss_add.item())
                    print("graph loss", loss_graph.item())

                loss.backward(retain_graph=True)
                optimizer.step()

                if do_clip:
                    inputs.data = clip(inputs.data, use_fp16=False)

        best_inputs = inputs.data.clone()
        best_inputs = denormalize(best_inputs)
        self.save_images(best_inputs)

        if self.exp_name is None:
            name = (
                f"ch_{self.channel}_c1_{self.layer_weights[0]}_"
                f"l1_{self.layer_weights[1]}_l2_{self.layer_weights[2]}_"
                f"l3_{self.layer_weights[3]}_l4_{self.layer_weights[4]}_masked.png"
            )
            place_to_store = os.path.join(self.folder_name, name)
        else:
            place_to_store = os.path.join(self.folder_name, self.exp_name + "_masked.png")

        save_maco(
            inputs.data.clone()[0],
            transparency_accumulator.data.clone()[0],
            percentile_image=1.0,
            percentile_alpha=98,
            filename=place_to_store
        )

        optimizer.state = collections.defaultdict(dict)

    def save_images(self, images):
        for id in range(images.shape[0]):
            if self.exp_name is None:
                name = (
                    f"ch_{self.channel}_c1_{self.layer_weights[0]}_"
                    f"l1_{self.layer_weights[1]}_l2_{self.layer_weights[2]}_"
                    f"l3_{self.layer_weights[3]}_l4_{self.layer_weights[4]}.png"
                )
                place_to_store = os.path.join(self.folder_name, name)
            else:
                place_to_store = os.path.join(self.folder_name, self.exp_name + ".png")

            image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
            pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
            pil_image.save(place_to_store)
