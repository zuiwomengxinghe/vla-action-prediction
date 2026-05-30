import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CacheManager:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_path(self, name: str) -> Path:
        return self.cache_dir / f"{name}.pt"
    
    def load(self, name: str, expected_count: int = None) -> dict:
        path = self._get_path(name)
        if not path.exists():
            print(f"  [Cache] {name} not found")
            return None
        try:
            saved_data = torch.load(path, weights_only=False)
            if expected_count is not None and saved_data.get('expected_count') != expected_count:
                print(f"  [Cache] {name} count mismatch: expected {expected_count}, got {saved_data.get('expected_count')}")
                return None
            print(f"  [Cache] Loaded {name} from {path}")
            return saved_data['data']
        except Exception as e:
            print(f"  [Cache] Failed to load {name}: {e}")
            return None
    
    def save(self, name: str, data, expected_count: int = None):
        path = self._get_path(name)
        try:
            torch.save({
                'data': data,
                'expected_count': expected_count or (len(data) if hasattr(data, '__len__') else None),
                'saved_count': len(data) if hasattr(data, '__len__') else None,
            }, path)
            print(f"  [Cache] Saved {name} to {path}")
        except Exception as e:
            print(f"  [Cache] Failed to save {name}: {e}")


class VideoFrameExtractor:
    def __init__(self, video_path: Path, fps: float):
        self.video_path = str(video_path)
        self.fps = fps
        self.backend = None
        self._init_reader()

    def _init_reader(self):
        try:
            import cv2

            self.cv2 = cv2
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video: {self.video_path}")
            self.backend = "cv2"
        except Exception:
            try:
                import imageio.v3 as iio

                self.iio = iio
                self.reader = iio.get_reader(self.video_path, format="ffmpeg")
                self.backend = "imageio"
            except Exception as exc:
                raise RuntimeError(
                    "Failed to initialize video reader. Install opencv-python or imageio[ffmpeg]."
                ) from exc

    def get_frame(self, video_frame_index: int) -> Image.Image:
        if self.backend == "cv2":
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, int(video_frame_index))
            success, frame = self.cap.read()
            if not success or frame is None:
                raise RuntimeError(f"Failed to read frame {video_frame_index} from {self.video_path}")
            frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame)

        if self.backend == "imageio":
            frame = self.reader.get_data(int(video_frame_index))
            return Image.fromarray(frame)

        raise RuntimeError("Unsupported video backend")

    def close(self):
        if self.backend == "cv2":
            self.cap.release()
        elif self.backend == "imageio":
            self.reader.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_data_frames(root: Path) -> pd.DataFrame:
    data_files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError("No data Parquet files found under data/chunk-*/file-*.parquet")
    frames = []
    for path in data_files:
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def load_episode_meta(root: Path) -> pd.DataFrame:
    meta_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(str(meta_path))
    meta = pd.read_parquet(meta_path)
    meta = meta.rename(columns={
        "videos/observation.images.top/from_timestamp": "video_from_timestamp",
        "videos/observation.images.top/to_timestamp": "video_to_timestamp",
    })
    return meta


def sample_frame_rows(data_df: pd.DataFrame, sample_ratio: float, seed: int, cache: CacheManager = None) -> pd.DataFrame:
    sample_count = max(1, int(len(data_df) * sample_ratio))
    
    if cache is not None:
        cached_indices = cache.load("selected_global_indices", expected_count=sample_count)
        if cached_indices is not None:
            print(f"  [Cache] Using {len(cached_indices)} cached sampled indices")
            result = data_df.iloc[cached_indices].copy()
            result["global_index"] = cached_indices
            return result.reset_index(drop=True)
    
    sampled = data_df.sample(n=sample_count, random_state=seed)
    sampled_indices = sampled.index.tolist()
    
    if cache is not None:
        cache.save("selected_global_indices", sampled_indices, expected_count=sample_count)
        print(f"  [Cache] Saved {sample_count} sampled indices")
    
    result = sampled.copy()
    result["global_index"] = sampled_indices
    return result.reset_index(drop=True)


def build_episode_lookup(meta_df: pd.DataFrame) -> dict:
    lookup = {}
    for _, row in meta_df.iterrows():
        episode_index = int(row["episode_index"])
        tasks = row["tasks"]
        if isinstance(tasks, (list, tuple, np.ndarray)) and len(tasks) > 0:
            text = str(tasks[0])
        else:
            text = str(tasks)
        lookup[episode_index] = {
            "task_text": text,
            "video_from_timestamp": float(row["video_from_timestamp"])
            if not np.isnan(row["video_from_timestamp"])
            else 0.0,
        }
    return lookup


def extract_clip_embeddings(
    rows: pd.DataFrame,
    episode_lookup: dict,
    video_path: Path,
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    fps: float,
    device: torch.device,
    batch_size: int = 32,
    cache: CacheManager = None,
):
    if cache is not None:
        cached_image = cache.load("image_embeddings", expected_count=20000)
        cached_text = cache.load("text_embeddings", expected_count=20000)
        if cached_image is not None and cached_text is not None:
            print("  [Cache] Using cached CLIP embeddings")
            global_indices = rows["global_index"].tolist() if "global_index" in rows.columns else rows.index.tolist()
            image_features = cached_image[global_indices]
            text_features = cached_text[global_indices]
            return image_features, text_features

    clip_model.eval()
    text_inputs = processor(
        text=[episode_lookup[int(ep)]["task_text"] for ep in rows["episode_index"]],
        return_tensors="pt",
        padding=True,
    ).to(device)
    with torch.no_grad():
        text_outputs = clip_model.get_text_features(**text_inputs)
        if hasattr(text_outputs, 'last_hidden_state'):
            text_features = text_outputs.last_hidden_state[:, 0, :]
        else:
            text_features = text_outputs

    frame_extractor = VideoFrameExtractor(video_path, fps)
    images = []
    for idx, row in rows.iterrows():
        episode_info = episode_lookup[int(row["episode_index"])]
        absolute_timestamp = episode_info["video_from_timestamp"] + float(row["timestamp"])
        # 注意：video_frame_index 是视频文件中的绝对帧索引（0-19999）
        # 与数据集中的 frame_index（episode 内的相对帧索引 0-399）不同
        video_frame_index = int(round(absolute_timestamp * fps))
        image = frame_extractor.get_frame(video_frame_index)
        images.append(image)
    frame_extractor.close()

    image_features = []
    for start in range(0, len(images), batch_size):
        batch_images = images[start : start + batch_size]
        inputs = processor(images=batch_images, return_tensors="pt").to(device)
        with torch.no_grad():
            batch_outputs = clip_model.get_image_features(**inputs)
            if hasattr(batch_outputs, 'last_hidden_state'):
                batch_features = batch_outputs.last_hidden_state[:, 0, :]
            else:
                batch_features = batch_outputs
        image_features.append(batch_features.cpu())
    image_features = torch.cat(image_features, dim=0)

    return image_features, text_features.cpu()


def prepare_targets(rows: pd.DataFrame, arm: str = "right") -> torch.Tensor:
    action_array = np.stack(rows["action"].apply(lambda value: np.asarray(value)).values)
    if arm == "right":
        target = action_array[:, 7:14]
    else:
        target = action_array[:, :7]
    return torch.from_numpy(target).float()


def train_mlp(
    features: torch.Tensor,
    targets: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
):
    train_size = int(len(features) * 0.8)
    test_size = len(features) - train_size
    dataset = torch.utils.data.TensorDataset(features.to(device), targets.to(device))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    model = MLPRegressor(input_dim=features.shape[1], output_dim=targets.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_test = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                preds = model(xb)
                loss = criterion(preds, yb)
                test_loss += loss.item() * xb.size(0)
        test_loss /= len(test_loader.dataset)

        best_test = min(best_test, test_loss)
        print(f"Epoch {epoch:02d}: train_mse={train_loss:.6f}, test_mse={test_loss:.6f}")

    print(f"Best test MSE: {best_test:.6f}")
    return model, best_test


def train_mlp_with_test(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
):
    train_dataset = torch.utils.data.TensorDataset(train_features.to(device), train_targets.to(device))
    test_dataset = torch.utils.data.TensorDataset(test_features.to(device), test_targets.to(device))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    model = MLPRegressor(input_dim=train_features.shape[1], output_dim=train_targets.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_test = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                preds = model(xb)
                loss = criterion(preds, yb)
                test_loss += loss.item() * xb.size(0)
        test_loss /= len(test_loader.dataset)

        best_test = min(best_test, test_loss)
        print(f"Epoch {epoch:02d}: train_mse={train_loss:.6f}, test_mse={test_loss:.6f}")

    print(f"Best test MSE: {best_test:.6f}")
    return model, best_test


def load_clip_model_and_processor(model_name: str, cache_dir: Path):
    clip_model = CLIPModel.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        local_files_only=True,
    )
    processor = CLIPProcessor.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        local_files_only=True,
    )
    print("Loaded CLIP model from local cache (offline mode)")
    return clip_model, processor


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline CLIP+MLP regression for LeRobot dataset")
    parser.add_argument("--root", type=str, default=".", help="Dataset root directory")
    parser.add_argument("--cache-dir", type=str, default="./clip_cache", help="Local cache directory for CLIP weights")
    parser.add_argument("--computation-cache-dir", type=str, default="./computation_cache", help="Directory for cached embeddings")
    parser.add_argument("--sample-ratio", type=float, default=0.15, help="Fraction of rows to sample")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-right-arm", action="store_true", help="Predict right arm only (7 DoF)")
    parser.add_argument("--use-left-arm", action="store_true", help="Predict left arm only (7 DoF)")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.root)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(info.get("fps", 50.0))

    print("Loading data frames...")
    data_df = load_data_frames(root)
    print(f"Total data rows: {len(data_df)}")

    print("Loading episode metadata...")
    meta_df = load_episode_meta(root)
    episode_lookup = build_episode_lookup(meta_df)

    computation_cache_dir = Path(args.computation_cache_dir)
    computation_cache = CacheManager(computation_cache_dir)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    clip_model, processor = load_clip_model_and_processor("openai/clip-vit-base-patch32", cache_dir)
    for param in clip_model.parameters():
        param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading CLIP embeddings from cache...")
    cached_data = computation_cache.load("image_embeddings")
    if cached_data is None:
        raise FileNotFoundError("image_embeddings.pt not found in cache")
    image_feats = cached_data
    
    cached_data = computation_cache.load("text_embeddings")
    if cached_data is None:
        raise FileNotFoundError("text_embeddings.pt not found in cache")
    text_feats = cached_data

    total_samples = len(image_feats)
    print(f"Total cached samples: {total_samples}")

    arm = "right" if args.use_right_arm or not args.use_left_arm else "left"

    print("Randomly selecting 2000 training samples...")
    train_indices = random.sample(range(total_samples), 2000)
    train_indices.sort()
    
    train_image_feats = image_feats[train_indices]
    train_text_feats = text_feats[train_indices]
    train_targets = prepare_targets(data_df.iloc[train_indices].copy(), arm=arm)

    print("Building feature matrix for training...")
    train_features = torch.cat([train_image_feats, train_text_feats], dim=1)

    print("Building feature matrix for testing (all samples)...")
    test_features = torch.cat([image_feats, text_feats], dim=1)
    test_targets = prepare_targets(data_df, arm=arm)

    print(f"Training MLP to predict {arm} arm 7-DoF action")
    model, best_test = train_mlp_with_test(
        train_features, train_targets, test_features, test_targets,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device
    )


if __name__ == "__main__":
    main()
