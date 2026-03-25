import os
import random
import argparse
from pathlib import Path

import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize channel activations for one ImageNet image."
    )

    parser.add_argument("--imagenet_root", type=str,
                        default="/BS/Databases23/imagenet/original",
                        help="Path to ImageNet root. Expected layout: root/train/<class_folder>/...")

    parser.add_argument("--split", type=str,
                        default="train",
                        choices=["train", "val"],
                        help="Which ImageNet split to use.")

    parser.add_argument("--class_idx", type=int,
                        default=420,
                        help="ImageNet class index in sorted folder order (0-based).")

    parser.add_argument("--image_index", type=int,
                        default=2,
                        help="Pick a specific image index from the class folder. If None, sample randomly.")

    parser.add_argument("--arch", type=str,
                        default="resnet50",
                        help="Torchvision model name.")

    parser.add_argument("--target_layer", type=str,
                        default="layer4",
                        help="Layer name to hook, e.g. layer4.")

    parser.add_argument("--topk_channels", type=int,
                        default=5,
                        help="How many top channels from this image to visualize.")

    parser.add_argument("--seed", type=int,
                        default=0,
                        help="Random seed.")

    parser.add_argument("--device", type=str,
                        default="cuda",
                        help="cuda or cpu")

    parser.add_argument("--use_weights_enum", action="store_true",
                        help="Use torchvision weights enum API if needed.")

    parser.add_argument("--outdir", type=str,
                        default="channel_viz",
                        help="Output directory.")

    return parser.parse_args()


def get_model(arch: str, device: str, use_weights_enum: bool = False):
    if use_weights_enum:
        if arch == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif arch == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            raise ValueError(f"Add weights enum support for arch={arch}")
    else:
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
    img_pil = Image.open(path).convert("RGB")
    x = transform(img_pil).unsqueeze(0).to(device)
    return img_pil, x


def tensor_to_numpy_img(x):
    # x: [1,3,H,W] in [0,1]
    img = x.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    return img


def normalize_map(m):
    m = m - m.min()
    if m.max() > 0:
        m = m / m.max()
    return m


def overlay_heatmap_on_image(img_np, heatmap, alpha=0.45):
    """
    img_np: [H,W,3] in [0,1]
    heatmap: [H,W] in [0,1]
    """
    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(heatmap)[..., :3]  # drop alpha
    overlay = (1 - alpha) * img_np + alpha * heat_rgb
    overlay = np.clip(overlay, 0, 1)
    return overlay


def save_channel_visualizations(img_np, feat, top_idx, outdir, prefix=""):
    """
    img_np: [H,W,3]
    feat: [1,C,Hf,Wf]
    top_idx: list of channel indices
    """
    os.makedirs(outdir, exist_ok=True)

    H, W = img_np.shape[:2]

    for rank, ch in enumerate(top_idx, start=1):
        act = feat[0, ch]  # [Hf, Wf]
        act_up = torch.nn.functional.interpolate(
            act.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False
        ).squeeze().cpu().numpy()

        act_norm = normalize_map(act_up)
        overlay = overlay_heatmap_on_image(img_np, act_norm)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(img_np)
        axes[0].set_title("Input image")
        axes[0].axis("off")

        axes[1].imshow(act_norm, cmap="jet")
        axes[1].set_title(f"Channel {ch} activation")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay channel {ch}")
        axes[2].axis("off")

        plt.tight_layout()
        save_path = os.path.join(outdir, f"{prefix}rank{rank}_channel{ch}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {save_path}")


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

    if args.image_index is None:
        img_path = random.choice(image_paths)
    else:
        if args.image_index < 0 or args.image_index >= len(image_paths):
            raise ValueError(f"image_index={args.image_index} out of range for {len(image_paths)} images")
        img_path = image_paths[args.image_index]

    print(f"Class folder : {class_dir.name}")
    print(f"Image path   : {img_path}")
    print(f"Model        : {args.arch}")
    print(f"Target layer : {args.target_layer}")

    model = get_model(args.arch, device, use_weights_enum=args.use_weights_enum)
    activations, hook_handle = register_activation_hook(model, args.target_layer)
    transform = build_transform()

    img_pil, x = load_image(img_path, transform, device)
    img_np = tensor_to_numpy_img(x)

    with torch.no_grad():
        _ = model(x)

    if "feat" not in activations:
        raise RuntimeError("Hook did not capture activations.")

    feat = activations["feat"]  # [1,C,H,W]
    hook_handle.remove()

    # mean spatial activation per channel for this single image
    channel_scores = feat.mean(dim=(2, 3)).squeeze(0).cpu()
    top_vals, top_idx = torch.topk(channel_scores, k=args.topk_channels)

    print("\nTop channels for this image:")
    for rank, (ch, val) in enumerate(zip(top_idx.tolist(), top_vals.tolist()), start=1):
        print(f"{rank}. channel={ch:4d}  mean_activation={val:.6f}")

    prefix = f"class{args.class_idx}_{class_dir.name}_"
    save_channel_visualizations(
        img_np=img_np,
        feat=feat,
        top_idx=top_idx.tolist(),
        outdir=args.outdir,
        prefix=prefix
    )


if __name__ == "__main__":
    main()