import argparse
import json
import math
import os
import random
from pathlib import Path

os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


class CacheManager:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_path(self, name: str) -> Path:
        return self.cache_dir / f"{name}.pt"
    
    def save(self, name: str, data: dict, expected_count: int = None):
        path = self._get_path(name)
        temp_path = path.with_suffix('.tmp')
        save_data = {
            'data': data,
            'expected_count': expected_count,
            'saved_count': len(data) if hasattr(data, '__len__') else None,
        }
        if isinstance(data, dict):
            save_data['saved_count'] = len(list(data.values())[0]) if data else 0
        torch.save(save_data, temp_path)
        if path.exists():
            path.unlink()
        temp_path.rename(path)
        print(f"  [Cache] Saved {name} to {path}")
    
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
    
    def exists(self, name: str, expected_count: int = None) -> bool:
        path = self._get_path(name)
        if not path.exists():
            return False
        try:
            saved_data = torch.load(path, weights_only=False)
            if expected_count is not None and saved_data.get('expected_count') != expected_count:
                return False
            return True
        except:
            return False


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


def extract_clip_embeddings_batch(
    rows: pd.DataFrame,
    episode_lookup: dict,
    video_path: Path,
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    fps: float,
    device: torch.device,
    cache: CacheManager = None,
    batch_size: int = 32,
):
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

    if cache is not None:
        cached = cache.load("image_embeddings", expected_count=len(rows))
        if cached is not None:
            print("  [Cache] Using cached image embeddings")
            return cached, text_features.cpu()

    frame_extractor = VideoFrameExtractor(video_path, fps)
    images = []
    for idx, row in tqdm(rows.iterrows(), total=len(rows), desc="Extracting video frames"):
        episode_info = episode_lookup[int(row["episode_index"])]
        absolute_timestamp = episode_info["video_from_timestamp"] + float(row["timestamp"])
        video_frame_index = int(round(absolute_timestamp * fps))
        image = frame_extractor.get_frame(video_frame_index)
        images.append(image)
    frame_extractor.close()

    image_features = []
    for start in tqdm(range(0, len(images), batch_size), desc="Computing CLIP image embeddings"):
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

    if cache is not None:
        cache.save("image_embeddings", image_features, expected_count=len(rows))

    return image_features, text_features.cpu()


def prepare_targets(rows: pd.DataFrame, arm: str = "right") -> torch.Tensor:
    action_array = np.stack(rows["action"].apply(lambda value: np.asarray(value)).values)
    if arm == "right":
        target = action_array[:, 7:14]
    else:
        target = action_array[:, :7]
    return torch.from_numpy(target).float()


def normalize_features_per_modality(action_array, state_array, image_features):
    """
    对每个模态的特征进行标准化，确保尺度一致
    
    注意:
    - CLIP embedding 是语义方向空间，应使用 L2 normalization 保留方向
    - action 和 state 是物理量，使用 z-score 标准化
    """
    action_mean = action_array.mean(axis=0)
    action_std = action_array.std(axis=0) + 1e-8
    action_normalized = (action_array - action_mean) / action_std
    
    state_mean = state_array.mean(axis=0)
    state_std = state_array.std(axis=0) + 1e-8
    state_normalized = (state_array - state_mean) / state_std
    
    image_normalized = image_features / (image_features.norm(dim=1, keepdim=True) + 1e-8)
    
    return action_normalized, state_normalized, image_normalized


def compute_cosine_similarity_matrix(
    image_features: torch.Tensor,
    state_features: torch.Tensor,
    action_features: torch.Tensor,
    lambda_f: float = 1.0,
    lambda_s: float = 0.5,
    lambda_a: float = 0.25,
) -> torch.Tensor:
    """
    计算综合余弦相似度矩阵
    sim(i,j) = lambda_f * sim_f(i,j) + lambda_s * sim_s(i,j) + lambda_a * sim_a(i,j)
    """
    def cosine_sim(x):
        norm_x = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        return norm_x @ norm_x.T
    
    sim_f = cosine_sim(image_features)
    sim_s = cosine_sim(state_features)
    sim_a = cosine_sim(action_features)
    
    sim_matrix = lambda_f * sim_f + lambda_s * sim_s + lambda_a * sim_a
    return sim_matrix


def facility_location_coreset_selection(
    sim_matrix: torch.Tensor,
    k_coreset: int = 2000,
) -> list:
    """
    阶段2: Facility Location coreset选择 (增量更新优化版)
    
    目标函数: F(S) = sum_i max_{j in S} sim(z_i, z_j)
    贪心算法: 每步选择使 F(S union {x}) - F(S) 最大化的帧 x*
    
    优化:
    - 使用布尔mask替代线性查找，O(1)判断
    - 增量更新current_max_sim，避免重复计算
    - 向量化gain计算，提升效率
    """
    n = sim_matrix.shape[0]
    selected = []
    selected_mask = torch.zeros(n, dtype=torch.bool)
    current_max_sim = torch.zeros(n)
    
    print(f"Starting Facility Location coreset selection: {n} -> {k_coreset}")
    
    for step in tqdm(range(k_coreset), desc="Selecting coreset"):
        if step % 100 == 0:
            print(f"  Step {step}/{k_coreset}, current F(S) = {current_max_sim.sum().item():.4f}")
        
        # 获取未选中的样本索引
        unselected_mask = ~selected_mask
        
        # 向量化计算所有未选中样本的gain
        # new_max_sim[i] = max(current_max_sim, sim_matrix[i])
        # gain[i] = sum(new_max_sim[i]) - sum(current_max_sim)
        new_max_sim = torch.maximum(current_max_sim.unsqueeze(0), sim_matrix[unselected_mask])
        gains = new_max_sim.sum(dim=1) - current_max_sim.sum()
        
        # 找到gain最大的样本
        best_local_idx = gains.argmax().item()
        best_global_idx = torch.where(unselected_mask)[0][best_local_idx].item()
        
        # 更新选中状态
        selected.append(best_global_idx)
        selected_mask[best_global_idx] = True
        
        # 增量更新current_max_sim
        current_max_sim = torch.maximum(current_max_sim, sim_matrix[best_global_idx])
    
    print(f"Facility Location selection complete. Selected {len(selected)} frames.")
    return selected


def direct_coreset_selection(
    data_df: pd.DataFrame,
    episode_lookup: dict,
    video_path: Path,
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    fps: float,
    device: torch.device,
    cache: CacheManager = None,
    lambda_f: float = 1.0,
    lambda_s: float = 0.5,
    lambda_a: float = 0.25,
    k_coreset: int = 2000,
    batch_size: int = 32,
) -> tuple:
    """
    直接对全量数据执行 Facility Location coreset 选择（跳过时间重要性评分）
    """
    total_frames = len(data_df)
    
    if cache is not None:
        cached_final_df = cache.load("final_df", expected_count=k_coreset)
        cached_final_image = cache.load("final_image_embeddings", expected_count=k_coreset)
        cached_final_text = cache.load("final_text_embeddings", expected_count=k_coreset)
        if cached_final_df is not None and cached_final_image is not None and cached_final_text is not None:
            print("  [Cache] Using cached final coreset selection results")
            return cached_final_df, cached_final_image, cached_final_text
    
    # 提取全量数据的 CLIP 嵌入
    print("Extracting CLIP embeddings for all frames...")
    image_features, text_features = extract_clip_embeddings_batch(
        data_df,
        episode_lookup,
        video_path,
        clip_model,
        processor,
        fps,
        device,
        cache=cache,
        batch_size=batch_size,
    )
    
    # 解析 action 和 state
    print("Parsing action and state vectors...")
    action_array = np.stack(data_df["action"].apply(lambda value: np.asarray(value)).values)
    state_array = np.stack(data_df["observation.state"].apply(lambda value: np.asarray(value)).values)
    
    # 特征归一化
    print("Normalizing features for Facility Location...")
    action_norm, state_norm, image_norm = normalize_features_per_modality(
        action_array, state_array, image_features
    )
    
    state_features = torch.from_numpy(state_norm).float()
    action_features = torch.from_numpy(action_norm).float()
    image_features_norm = image_norm
    
    # 计算相似度矩阵（可缓存）
    if cache is not None:
        cached_sim_matrix = cache.load("sim_matrix", expected_count=total_frames)
        if cached_sim_matrix is not None:
            print("  [Cache] Using cached similarity matrix")
            sim_matrix = cached_sim_matrix
        else:
            print("Computing similarity matrix...")
            sim_matrix = compute_cosine_similarity_matrix(
                image_features_norm,
                state_features,
                action_features,
                lambda_f=lambda_f,
                lambda_s=lambda_s,
                lambda_a=lambda_a,
            )
            cache.save("sim_matrix", sim_matrix, expected_count=total_frames)
    else:
        print("Computing similarity matrix...")
        sim_matrix = compute_cosine_similarity_matrix(
            image_features_norm,
            state_features,
            action_features,
            lambda_f=lambda_f,
            lambda_s=lambda_s,
            lambda_a=lambda_a,
        )
    
    # Facility Location 贪心选择
    selected_indices = facility_location_coreset_selection(sim_matrix, k_coreset=k_coreset)
    selected_indices_sorted = sorted(selected_indices)
    
    final_df = data_df.iloc[selected_indices_sorted].reset_index(drop=True)
    final_image_features = image_features[selected_indices_sorted]
    final_text_features = text_features[selected_indices_sorted]
    
    print(f"\nFinal coreset size: {len(final_df)} frames")
    
    if cache is not None:
        cache.save("final_df", final_df, expected_count=k_coreset)
        cache.save("final_image_embeddings", final_image_features, expected_count=k_coreset)
        cache.save("final_text_embeddings", final_text_features, expected_count=k_coreset)
        cache.save("selected_global_indices", selected_indices_sorted, expected_count=k_coreset)
    
    return final_df, final_image_features, final_text_features


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
    for epoch in tqdm(range(1, epochs + 1), desc="Training MLP"):
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
    clip_model = CLIPModel.from_pretrained(model_name, cache_dir=str(cache_dir), local_files_only=True)
    processor = CLIPProcessor.from_pretrained(model_name, cache_dir=str(cache_dir), local_files_only=True)
    print("Loaded CLIP model from local cache (offline mode)")
    return clip_model, processor


def parse_args():
    parser = argparse.ArgumentParser(description="Direct Facility Location coreset selection (skip temporal scoring)")
    parser.add_argument("--root", type=str, default=".", help="Dataset root directory")
    parser.add_argument("--cache-dir", type=str, default="./clip_cache", help="Local cache directory for CLIP weights")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-right-arm", action="store_true", help="Predict right arm only (7 DoF)")
    parser.add_argument("--use-left-arm", action="store_true", help="Predict left arm only (7 DoF)")
    parser.add_argument("--lambda-f", type=float, default=1.0, help="Weight for image similarity")
    parser.add_argument("--lambda-s", type=float, default=0.5, help="Weight for state similarity")
    parser.add_argument("--lambda-a", type=float, default=0.25, help="Weight for action similarity")
    parser.add_argument("--k-coreset", type=int, default=2000, help="Final coreset size after Facility Location")
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

    cache_dir = root / "computation_cache"
    cache = CacheManager(cache_dir)
    
    clip_cache_dir = Path(args.cache_dir)
    clip_cache_dir.mkdir(parents=True, exist_ok=True)
    clip_model, processor = load_clip_model_and_processor("openai/clip-vit-base-patch32", clip_cache_dir)
    for param in clip_model.parameters():
        param.requires_grad = False

    video_path = root / "videos" / "observation.images.top" / "chunk-000" / "file-000.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    clip_model = clip_model.to(device)

    print("\n" + "=" * 60)
    print("DIRECT FACILITY LOCATION CORESET SELECTION (Skip Temporal Scoring)")
    print("=" * 60)
    
    final_df, final_image_features, final_text_features = direct_coreset_selection(
        data_df,
        episode_lookup,
        video_path,
        clip_model,
        processor,
        fps,
        device,
        cache=cache,
        lambda_f=args.lambda_f,
        lambda_s=args.lambda_s,
        lambda_a=args.lambda_a,
        k_coreset=args.k_coreset,
        batch_size=args.batch_size,
    )

    print("\n" + "=" * 60)
    print("BUILDING FEATURE MATRIX")
    print("=" * 60)
    
    if final_image_features.shape[0] != final_text_features.shape[0]:
        raise RuntimeError("Image and text embedding counts do not match.")
    train_features = torch.cat([final_image_features, final_text_features], dim=1)

    arm = "right" if args.use_right_arm or not args.use_left_arm else "left"
    train_targets = prepare_targets(final_df, arm=arm)

    print("\nLoading CLIP embeddings for entire dataset (test set) from cache...")
    total_frames = len(data_df)
    all_image_features = cache.load("image_embeddings", expected_count=total_frames)
    all_text_features = cache.load("text_embeddings", expected_count=total_frames)
    
    if all_image_features is None or all_text_features is None:
        raise RuntimeError("Failed to load cached CLIP embeddings for test set. Run with cache enabled first.")
    
    if all_image_features.shape[0] != all_text_features.shape[0]:
        raise RuntimeError("Image and text embedding counts do not match for test set.")
    test_features = torch.cat([all_image_features, all_text_features], dim=1)
    test_targets = prepare_targets(data_df, arm=arm)

    print(f"\nTraining MLP to predict {arm} arm 7-DoF action")
    print(f"Training samples (coreset): {len(train_features)}")
    print(f"Test samples (full dataset): {len(test_features)}")
    print(f"Feature dimension: {train_features.shape[1]}")
    print(f"Target dimension: {train_targets.shape[1]}")
    
    model, best_test_loss = train_mlp_with_test(
        train_features, train_targets, test_features, test_targets,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device
    )

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Training samples (coreset): {len(train_features)}")
    print(f"Test samples (full dataset): {len(test_features)}")
    print(f"Best test MSE (on full dataset): {best_test_loss:.6f}")


if __name__ == "__main__":
    main()