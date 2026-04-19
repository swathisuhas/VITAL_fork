import os
import heapq
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.utils as vutils
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


def list_of_ints(arg):
    return list(map(int, arg.split(',')))


parser = argparse.ArgumentParser()
parser.add_argument('--target_layer', type=str, default='layer4')
parser.add_argument('--arch', type=str, default='resnet50')
parser.add_argument('--patch_size', type=int, default=64)
parser.add_argument('--patch_stride', type=int, default=None,
                    help='Patch stride. Default: patch_size // 2')
parser.add_argument('--topk_patches', type=int, default=50)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--files_txt', type=str, required=True,
                    help='Path to files_all.txt')
parser.add_argument('--batch_size_images', type=int, default=8,
                    help='How many images to load per outer batch')
parser.add_argument('--max_patch_batch', type=int, default=4096,
                    help='Max number of patches forwarded at once')
parser.add_argument('--channels', type=str, default=None,
                    help='Optional comma-separated channel list, e.g. 0,1,2,10')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--resize_size', type=int, default=256)
parser.add_argument('--use_amp', action='store_true')
parser.add_argument('--save_patch_tensors', action='store_true',
                    help='Save top-k raw patch tensors per channel as .pt')
args = parser.parse_args()

print(args)


def get_module_by_name(model, layer_name):
    modules = dict(model.named_modules())
    if layer_name not in modules:
        raise ValueError(f"Layer '{layer_name}' not found. Available example keys: {list(modules.keys())[:20]}")
    return modules[layer_name]


def register_activation_hook(model, layer_name):
    activations = {}

    def hook_fn(module, inputs, output):
        activations['feat'] = output

    layer = get_module_by_name(model, layer_name)
    handle = layer.register_forward_hook(hook_fn)
    return activations, handle


def generate_model(arch='resnet50'):
    if arch not in models.__dict__:
        raise ValueError(f"Unknown architecture: {arch}")
    weights = None
    # torchvision compatibility
    try:
        if arch == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT
        elif arch == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT
        elif arch == 'resnet101':
            weights = models.ResNet101_Weights.DEFAULT
    except Exception:
        weights = None

    if weights is not None:
        net = models.__dict__[arch](weights=weights)
    else:
        try:
            net = models.__dict__[arch](pretrained=True)
        except TypeError:
            net = models.__dict__[arch](weights="DEFAULT")

    net = net.cuda().eval()
    net = net.to(memory_format=torch.channels_last)
    return net


def patchify(inputs, patch_size=64, stride=None):
    """
    inputs: [B, C, H, W]
    Returns:
        patches: [N, C, patch_size, patch_size]
        meta: list of tuples (img_idx_in_batch, patch_idx_within_image)
        num_patches_per_image: int
    """
    if stride is None:
        stride = patch_size // 2

    b, c, h, w = inputs.shape
    unfolded = F.unfold(inputs, kernel_size=patch_size, stride=stride)  # [B, C*K*K, L]
    num_patches_per_image = unfolded.shape[-1]

    patches = unfolded.transpose(1, 2).contiguous()
    patches = patches.view(b * num_patches_per_image, c, patch_size, patch_size)

    meta = []
    for img_idx in range(b):
        for patch_idx in range(num_patches_per_image):
            meta.append((img_idx, patch_idx))

    return patches, meta, num_patches_per_image


def chunked_forward_get_scores(model, act_map, patch_batch, max_patch_batch, use_amp):
    """
    patch_batch: [N, 3, P, P]
    Returns:
        pooled activations: [N, C]
    """
    outputs = []
    n = patch_batch.shape[0]

    for start in range(0, n, max_patch_batch):
        end = min(start + max_patch_batch, n)
        x = patch_batch[start:end].cuda(non_blocking=True)
        x = x.to(memory_format=torch.channels_last)

        with torch.inference_mode():
            if use_amp:
                with torch.cuda.amp.autocast():
                    _ = model(x)
            else:
                _ = model(x)

            feat = act_map['feat']               # [chunk, C, H', W']
            pooled = feat.mean(dim=(2, 3))      # [chunk, C]
            outputs.append(pooled.detach().cpu())

        del x, feat, pooled
        torch.cuda.empty_cache()

    return torch.cat(outputs, dim=0)


def load_image_paths(files_txt):
    with open(files_txt, 'r') as f:
        img_list = [line.strip() for line in f if line.strip()]
    return img_list


def batched(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


def safe_load_and_transform(paths, transform):
    images = []
    valid_paths = []
    for p in paths:
        try:
            img = Image.open(p).convert('RGB')
            img = transform(img)
            images.append(img)
            valid_paths.append(p)
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")
    if not images:
        return None, []
    return torch.stack(images, dim=0), valid_paths


def save_channel_results(channel, items, save_dir, topk_patches):
    """
    items: sorted descending by score
          each item = (score, img_path, patch_tensor_cpu)
    """
    ch_dir = os.path.join(save_dir, str(channel))
    os.makedirs(ch_dir, exist_ok=True)

    top_items = items[:topk_patches]

    with open(os.path.join(ch_dir, 'topk_files.txt'), 'w') as f:
        for score, img_path, _ in top_items:
            f.write(f"{img_path}\n")

    with open(os.path.join(ch_dir, 'topk_scores.txt'), 'w') as f:
        for score, img_path, _ in top_items:
            f.write(f"{score:.8f}\t{img_path}\n")

    if len(top_items) > 0:
        vis_items = top_items[:16]
        patch_grid = torch.stack([x[2] for x in vis_items], dim=0)
        vutils.save_image(
            patch_grid,
            os.path.join(ch_dir, 'patches.png'),
            normalize=True,
            scale_each=True,
            nrow=4
        )

    if args.save_patch_tensors:
        patch_tensors = [x[2] for x in top_items]
        torch.save(patch_tensors, os.path.join(ch_dir, 'topk_patches.pt'))


def main():
    os.makedirs(args.save_dir, exist_ok=True)

    stride = args.patch_stride if args.patch_stride is not None else args.patch_size // 2

    transform = transforms.Compose([
        transforms.Resize(args.resize_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
    ])

    model = generate_model(args.arch)
    act_map, hook_handle = register_activation_hook(model, args.target_layer)

    img_list = load_image_paths(args.files_txt)
    if len(img_list) == 0:
        raise ValueError(f"No image paths found in {args.files_txt}")

    # Infer channel count with one dummy forward
    dummy = torch.randn(1, 3, args.patch_size, args.patch_size).cuda()
    dummy = dummy.to(memory_format=torch.channels_last)
    with torch.inference_mode():
        if args.use_amp:
            with torch.cuda.amp.autocast():
                _ = model(dummy)
        else:
            _ = model(dummy)
    feat = act_map['feat']
    num_channels = feat.shape[1]
    print(f"[INFO] Feature channels at layer '{args.target_layer}': {num_channels}")
    del dummy, feat
    torch.cuda.empty_cache()

    if args.channels is not None:
        channel_ids = list_of_ints(args.channels)
        for ch in channel_ids:
            if ch < 0 or ch >= num_channels:
                raise ValueError(f"Channel {ch} out of range [0, {num_channels-1}]")
    else:
        channel_ids = list(range(num_channels))

    # Min-heaps per channel: stores (score, img_path, patch_tensor_cpu)
    topk_heaps = defaultdict(list)

    for batch_paths in tqdm(list(batched(img_list, args.batch_size_images)), desc='Image batches'):
        imgs_cpu, valid_paths = safe_load_and_transform(batch_paths, transform)
        if imgs_cpu is None or len(valid_paths) == 0:
            continue

        patches_cpu, meta, num_patches_per_image = patchify(
            imgs_cpu, patch_size=args.patch_size, stride=stride
        )

        # Forward all patches in chunks and get [N_patches, C]
        pooled_scores = chunked_forward_get_scores(
            model=model,
            act_map=act_map,
            patch_batch=patches_cpu,
            max_patch_batch=args.max_patch_batch,
            use_amp=args.use_amp
        )  # cpu tensor [N, C]

        # For each image in this batch, find best patch per channel
        bsz = len(valid_paths)
        for img_idx in range(bsz):
            start = img_idx * num_patches_per_image
            end = start + num_patches_per_image

            img_scores = pooled_scores[start:end]         # [L, C]
            img_patches = patches_cpu[start:end]          # [L, 3, P, P]
            img_path = valid_paths[img_idx]

            if args.channels is None:
                best_scores, best_patch_idx = img_scores.max(dim=0)  # [C], [C]
                for ch in channel_ids:
                    score = float(best_scores[ch].item())
                    pidx = int(best_patch_idx[ch].item())
                    patch_tensor = img_patches[pidx].clone()

                    heap = topk_heaps[ch]
                    item = (score, img_path, patch_tensor)

                    if len(heap) < args.topk_patches:
                        heapq.heappush(heap, item)
                    else:
                        if score > heap[0][0]:
                            heapq.heapreplace(heap, item)
            else:
                sub_scores = img_scores[:, channel_ids]                # [L, len(channel_ids)]
                best_scores, best_patch_idx = sub_scores.max(dim=0)    # [len(channel_ids)]
                for i, ch in enumerate(channel_ids):
                    score = float(best_scores[i].item())
                    pidx = int(best_patch_idx[i].item())
                    patch_tensor = img_patches[pidx].clone()

                    heap = topk_heaps[ch]
                    item = (score, img_path, patch_tensor)

                    if len(heap) < args.topk_patches:
                        heapq.heappush(heap, item)
                    else:
                        if score > heap[0][0]:
                            heapq.heapreplace(heap, item)

        del imgs_cpu, patches_cpu, pooled_scores
        torch.cuda.empty_cache()

    print("[INFO] Saving results...")
    for ch in tqdm(channel_ids, desc='Saving per-channel outputs'):
        items = sorted(topk_heaps[ch], key=lambda x: x[0], reverse=True)
        save_channel_results(
            channel=ch,
            items=items,
            save_dir=args.save_dir,
            topk_patches=args.topk_patches
        )

    hook_handle.remove()
    print("[INFO] Done.")


if __name__ == '__main__':
    main()