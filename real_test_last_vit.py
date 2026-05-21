import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.models import ViT_B_16_Weights

from visualization.patch_score import DenseViT


def load_checkpoint(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state_dict = {
        key[6:] if key.startswith("model.") else key: value
        for key, value in state_dict.items()
    }
    return checkpoint, state_dict


def build_model(state_dict, device: str):
    model = DenseViT(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model, missing, unexpected


def gaussian_kernel_1d(kernel_size: int, sigma: float, device):
    values = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, device=device).float()
    kernel = torch.exp(-0.5 * (values / sigma) ** 2)
    return kernel / torch.max(kernel)


@torch.no_grad()
def encode_tokens(model: DenseViT, tensor: torch.Tensor):
    x = model._process_input(tensor)
    n = x.shape[0]
    batch_class_token = model.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)
    return model.encoder(x)


def last_aggregate(model: DenseViT, encoded: torch.Tensor):
    x_detach = encoded[:, 1:]
    x_fft = torch.fft.fft(x_detach, dim=-1)
    kernel = gaussian_kernel_1d(
        x_detach.shape[-1],
        x_detach.shape[-1] ** 0.5,
        x_detach.device,
    ).unsqueeze(0).unsqueeze(0)
    x_fft = torch.fft.fftshift(x_fft, dim=-1)
    x_fft = x_fft * kernel
    x_fft = torch.fft.ifftshift(x_fft, dim=-1)
    x_filtered = torch.fft.ifft(x_fft, dim=-1).real

    diff = x_detach / (torch.abs(x_filtered - x_detach) + 1e-6)
    _, indices = torch.topk(diff, k=1, dim=1, largest=True)
    selected = torch.gather(x_detach, 1, indices)
    cls_token = torch.mean(selected, dim=1)
    logits = model.heads(cls_token)
    counts = torch.bincount(indices[0, 0].detach().cpu(), minlength=x_detach.shape[1])
    return logits, counts.numpy()


def standard_cls_logits(model: DenseViT, encoded: torch.Tensor):
    cls_token = encoded[:, 0]
    patch_tokens = encoded[:, 1:]
    logits = model.heads(cls_token)
    scores = torch.cosine_similarity(
        patch_tokens,
        cls_token.unsqueeze(1).expand(-1, patch_tokens.shape[1], -1),
        dim=-1,
    )
    return logits, scores[0].detach().cpu().numpy()


def top5_from_logits(logits: torch.Tensor, categories):
    probs = F.softmax(logits[0], dim=0)
    top_probs, top_classes = probs.topk(5)
    return [
        {
            "rank": rank,
            "class_id": int(class_id),
            "label": categories[int(class_id)],
            "probability": float(prob.detach().cpu()),
        }
        for rank, (prob, class_id) in enumerate(zip(top_probs, top_classes.detach().cpu()), start=1)
    ]


def make_heatmap(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    values = (values - values.min()) / (values.max() - values.min() + 1e-8)
    red = values
    green = np.clip(1.5 - np.abs(values - 0.55) * 3.0, 0, 1)
    blue = np.clip(1.0 - values * 1.8, 0, 1)
    heat = np.stack([red, green, blue], axis=-1)
    heat = (heat * 255).astype(np.uint8)
    return np.array(Image.fromarray(heat).resize((224, 224), Image.Resampling.BILINEAR))


def save_heat_overlay(display_img: Image.Image, values: np.ndarray, top_indices, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display = display_img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    display_np = np.array(display).astype(np.float32)
    heat_np = make_heatmap(values.reshape(14, 14)).astype(np.float32)
    overlay_np = np.clip(display_np * 0.55 + heat_np * 0.45, 0, 255).astype(np.uint8)
    overlay = Image.fromarray(overlay_np)

    draw = ImageDraw.Draw(overlay)
    patch_size = 16
    colors = ["cyan", "yellow", "lime", "magenta", "white"]
    for rank, idx in enumerate(top_indices[:5], start=1):
        row = int(idx) // 14
        col = int(idx) % 14
        x0 = col * patch_size
        y0 = row * patch_size
        color = colors[(rank - 1) % len(colors)]
        draw.rectangle([x0, y0, x0 + patch_size - 1, y0 + patch_size - 1], outline=color, width=2)
        draw.text((x0 + 2, y0 + 2), str(rank), fill=color)
    overlay.save(output_path)


def save_mask(display_img: Image.Image, mask: np.ndarray, output_path: Path):
    display = display_img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    base = np.array(display).astype(np.float32)
    mask_img = np.zeros((224, 224), dtype=bool)
    patch_size = 16
    for idx, value in enumerate(mask):
        if value:
            row = int(idx) // 14
            col = int(idx) % 14
            mask_img[row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size] = True

    red = np.array([255, 45, 45], dtype=np.float32)
    base[mask_img] = base[mask_img] * 0.45 + red * 0.55
    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for idx, value in enumerate(mask):
        if value:
            row = int(idx) // 14
            col = int(idx) % 14
            x0 = col * patch_size
            y0 = row * patch_size
            draw.rectangle([x0, y0, x0 + patch_size - 1, y0 + patch_size - 1], outline="red", width=2)
    image.save(output_path)


def top_patches(values: np.ndarray, value_name: str):
    order = np.argsort(values)[::-1][:10]
    return [
        {
            "rank": rank,
            "patch_index": int(idx),
            "row": int(idx) // 14,
            "col": int(idx) % 14,
            value_name: float(values[int(idx)]),
        }
        for rank, idx in enumerate(order, start=1)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ViT_190k.pth"))
    parser.add_argument("--image", type=Path, default=Path("imgs/tea.png"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/real_test"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    output_stem = args.image.stem

    checkpoint, state_dict = load_checkpoint(args.checkpoint)
    model, missing, unexpected = build_model(state_dict, args.device)

    image = Image.open(args.image).convert("RGB")
    display_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    model_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    display_img = display_transform(image)
    tensor = model_transform(image).unsqueeze(0).to(args.device)

    encoded = encode_tokens(model, tensor)
    last_logits, last_selection_counts = last_aggregate(model, encoded)
    cls_logits, cls_cosine_scores = standard_cls_logits(model, encoded)

    categories = ViT_B_16_Weights.IMAGENET1K_V1.meta["categories"]
    last_top_indices = np.argsort(last_selection_counts)[::-1][:10]
    cls_top_indices = np.argsort(cls_cosine_scores)[::-1][:10]
    last_mask = last_selection_counts > (0.3 * max(1, last_selection_counts.max()))

    save_heat_overlay(
        display_img,
        last_selection_counts.astype(np.float32),
        last_top_indices,
        args.output_dir / f"{output_stem}_last_selection_count_overlay.png",
    )
    save_mask(display_img, last_mask, args.output_dir / f"{output_stem}_last_selection_mask_0p3.png")
    save_heat_overlay(
        display_img,
        cls_cosine_scores.astype(np.float32),
        cls_top_indices,
        args.output_dir / f"{output_stem}_standard_cls_cosine_overlay.png",
    )

    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_iteration": checkpoint.get("iteration") if isinstance(checkpoint, dict) else None,
        "image": str(args.image),
        "device": args.device,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "last_logits_top5_classes": top5_from_logits(last_logits, categories),
        "standard_cls_top5_classes": top5_from_logits(cls_logits, categories),
        "last_selection": {
            "channels": int(last_selection_counts.sum()),
            "unique_selected_patches": int(np.count_nonzero(last_selection_counts)),
            "max_channel_count": int(last_selection_counts.max()),
            "top10_patches": [
                {
                    "rank": rank,
                    "patch_index": int(idx),
                    "row": int(idx) // 14,
                    "col": int(idx) % 14,
                    "selected_channels": int(last_selection_counts[int(idx)]),
                }
                for rank, idx in enumerate(last_top_indices, start=1)
            ],
            "mask_threshold": "count > 0.3 * max_count",
            "mask_patch_count": int(last_mask.sum()),
        },
        "standard_cls_cosine_top10_patches": top_patches(cls_cosine_scores, "cosine_score"),
        "last_selection_stats": {
            "min": float(last_selection_counts.min()),
            "max": float(last_selection_counts.max()),
            "mean": float(last_selection_counts.mean()),
            "std": float(last_selection_counts.std()),
        },
        "standard_cls_cosine_stats": {
            "min": float(cls_cosine_scores.min()),
            "max": float(cls_cosine_scores.max()),
            "mean": float(cls_cosine_scores.mean()),
            "std": float(cls_cosine_scores.std()),
        },
        "outputs": [
            str(args.output_dir / f"{output_stem}_last_selection_count_overlay.png"),
            str(args.output_dir / f"{output_stem}_last_selection_mask_0p3.png"),
            str(args.output_dir / f"{output_stem}_standard_cls_cosine_overlay.png"),
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / f"{output_stem}_real_test_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
