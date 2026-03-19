import os
import random
import argparse
from pathlib import Path

import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(
        description="Find top activating channels for an ImageNet class using random images."
    )

    parser.add_argument("--imagenet_root", type=str,
                        default="/BS/Databases23/imagenet/original",
                        help="Path to ImageNet root. Expected layout: root/train/<class_folder>/...")

    parser.add_argument("--split", type=str,
                        default="train",
                        choices=["train", "val"],
                        help="Which ImageNet split to use.")

    parser.add_argument("--class_idx", type=int,
                        default=0,
                        help="ImageNet class index in sorted folder order (0-based).")

    parser.add_argument("--num_images", type=int,
                        default=10,
                        help="Number of random images to sample from the class.")

    parser.add_argument("--topk_channels", type=int,
                        default=5,
                        help="How many top channels to return.")

    parser.add_argument("--arch", type=str,
                        default="resnet50",
                        help="Torchvision model name.")

    parser.add_argument("--target_layer", type=str,
                        default="layer4",
                        help="Layer name to hook, e.g. layer4.")

    parser.add_argument("--seed", type=int,
                        default=0,
                        help="Random seed for reproducible sampling.")

    parser.add_argument("--device", type=str,
                        default="cuda",
                        help="cuda or cpu")

    parser.add_argument("--use_weights_enum", action="store_true",
                        help="Use torchvision weights enum API if needed.")

    return parser.parse_args()

def get_model(arch: str, device: str, use_weights_enum: bool = False):
    if use_weights_enum:
        # For newer torchvision versions
        if arch == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif arch == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            raise ValueError(f"Add weights enum support for arch={arch}")
    else:
        # Works on many torchvision versions
        model = models.__dict__[arch](pretrained=True)

    model.eval().to(device)
    return model


def register_activation_hook(model, layer_name: str):
    activations = {}

    def hook_fn(module, inp, out):
        activations["feat"] = out.detach()

    named_modules = dict(model.named_modules())
    if layer_name not in named_modules:
        available = list(named_modules.keys())
        raise ValueError(
            f"Layer '{layer_name}' not found.\n"
            f"Example available layers: {available[:30]}"
        )

    handle = named_modules[layer_name].register_forward_hook(hook_fn)
    return activations, handle


def get_class_folders(split_dir: Path):
    class_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class folders found in: {split_dir}")
    return class_dirs


def list_images(class_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in class_dir.rglob("*") if p.suffix.lower() in exts]
    if not images:
        raise RuntimeError(f"No images found in class folder: {class_dir}")
    return images


def build_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])


def load_image(path: Path, transform, device: str):
    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    return img


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU.")
        device = "cpu"

    split_dir = Path(args.imagenet_root) / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    class_dirs = get_class_folders(split_dir)

    if args.class_idx < 0 or args.class_idx >= len(class_dirs):
        raise ValueError(
            f"class_idx={args.class_idx} out of range. "
            f"Found {len(class_dirs)} class folders."
        )

    class_dir = class_dirs[args.class_idx]
    image_paths = list_images(class_dir)

    num_images = min(args.num_images, len(image_paths))
    sampled_paths = random.sample(image_paths, num_images)

    print(f"Using class_idx      : {args.class_idx}")
    print(f"Class folder         : {class_dir.name}")
    print(f"Class path           : {class_dir}")
    print(f"Available images     : {len(image_paths)}")
    print(f"Sampled images       : {num_images}")
    print(f"Model                : {args.arch}")
    print(f"Target layer         : {args.target_layer}")
    print()

    model = get_model(args.arch, device, use_weights_enum=args.use_weights_enum)
    activations, hook_handle = register_activation_hook(model, args.target_layer)
    transform = build_transform()

    all_channel_scores = []

    with torch.no_grad():
        for img_path in sampled_paths:
            x = load_image(img_path, transform, device)
            _ = model(x)

            if "feat" not in activations:
                raise RuntimeError("Hook did not capture activations.")

            feat = activations["feat"]  # shape [1, C, H, W]
            scores = feat.mean(dim=(2, 3)).squeeze(0).cpu()  # shape [C]
            all_channel_scores.append(scores)

    hook_handle.remove()

    all_channel_scores = torch.stack(all_channel_scores, dim=0)  # [N, C]
    mean_scores = all_channel_scores.mean(dim=0)                 # [C]

    top_vals, top_idx = torch.topk(mean_scores, k=args.topk_channels)

    print("Top channels (mean activation across sampled images):")
    for rank, (ch, val) in enumerate(zip(top_idx.tolist(), top_vals.tolist()), start=1):
        print(f"{rank}. channel={ch:4d}  mean_activation={val:.6f}")

    print("\nSampled image paths:")
    for p in sampled_paths:
        print(p)


if __name__ == "__main__":
    main()