from collections import defaultdict
import contextlib
import json
import time
import subprocess
import sys
import cv2
import torch
import os
import numpy as np
from tqdm import tqdm
import random
import torch.nn.functional as F
import warnings
import logging
import multiprocessing as mp
from transformers import AutoVideoProcessor, AutoModel

# do before: git clone git@github.com:facebookresearch/vjepa2.git
#sys.path.append("vjepa2")

# ===============================================================
# Notebook-style config
# Edit these values directly before running.
# ===============================================================
CONFIG = {
    # Paths
    "input_folder": "algonauts_2025.competitors/stimuli/",
    "output_folder": "embeddings/",

    # Only process these specific files (relative to input_folder/movies/).
    # Set to None to process all .mkv files found under input_folder.
    "video_filter": [
        "friends/s1/friends_s01e01a.mkv",
        "friends/s1/friends_s01e01b.mkv",
    ],

    # Feature extraction
    "tr": 1.49,
    "chunk_length": 8,
    "seconds_before_chunk": 6,
    "num_chunks": 1,
    "chunk_id": 0,

    # Model selection
    "use_vjepa": True,
    "use_ijepa": False,
    "vjepa_model_name": "facebook/vjepa2-vitl-fpc64-256",

    # Parallelism — set to False if you have only one GPU or are running on CPU
    "parallelize_across_gpus": False,
    "gpu_ids": [0],
}

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("torchvision").setLevel(logging.ERROR)


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def get_torch_device(gpu_id):
    if torch.cuda.is_available() and gpu_id != "cpu":
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def get_movie_info(movie_path):
    cap = cv2.VideoCapture(movie_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return fps, frame_count / fps


def split_movie_into_chunks(movie_path, tr, chunk_length, seconds_before_chunk):
    assert seconds_before_chunk < chunk_length, "seconds_before_chunk must be shorter than chunk_length"
    assert seconds_before_chunk + tr <= chunk_length, "chunk must be long enough to hold the whole TR window"

    _, video_duration = get_movie_info(movie_path)
    chunks, chunk_of_interests = [], []
    start_time = 0.0

    while start_time < video_duration:
        chunk_start = max(0, start_time - seconds_before_chunk)
        chunk_end = min(chunk_start + chunk_length, video_duration)

        rel_start = start_time - chunk_start
        rel_end = min(rel_start + tr, chunk_end - chunk_start)

        chunks.append((chunk_start, chunk_end))
        chunk_of_interests.append((rel_start, rel_end))
        start_time += tr

    return chunks, chunk_of_interests


def cut_clip(src, t0, t1):
    os.makedirs("clips", exist_ok=True)
    random_number = random.randint(1, 1_000_000_000)
    out = os.path.join(
        "clips",
        f"{os.environ.get('SLURM_ARRAY_TASK_ID', 0)}_{random_number}_chunk_{t0:.2f}_{t1:.2f}.mp4",
    )
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-y",
        "-ss", str(t0),
        "-to", str(t1),
        "-i", src,
        "-vf", "yadif",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        out,
    ]
    subprocess.run(cmd, check=True)

    for _ in range(10):
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            cap = cv2.VideoCapture(out)
            ret, _ = cap.read()
            cap.release()
            if ret:
                break
        time.sleep(0.2)

    return out


def register_hooks(mapping):
    store, handles = {}, []

    def save(name):
        def _hook(_, __, out):
            store[name] = out.detach()
        return _hook

    for k in mapping.keys():
        handles.append(mapping[k].register_forward_hook(save(k)))
    return store, handles


def crop_vjepa_by_time(tensor, rel_start, rel_end, clip_seconds):
    if tensor.dim() < 3:
        return tensor.squeeze()

    tensor = tensor.reshape(-1, 16, 16, tensor.shape[-1])  # T H W D
    features_pooled = F.adaptive_avg_pool2d(
        tensor.permute(0, 3, 1, 2),
        (3, 3),
    ).permute(0, 2, 3, 1)  # T H W D

    T = tensor.size(0)
    tok0 = min(T - 1, int(round(rel_start / clip_seconds * T)))
    tok1 = min(T, max(tok0 + 1, int(round(rel_end / clip_seconds * T))))
    return features_pooled[tok0:tok1].mean(0).flatten()


def get_video_from_path(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError(f"No frames could be read from {video_path}")

    return np.stack(frames, axis=0)


def build_vjepa_bundle(vjepa_model_name, device):
    with tqdm(total=2, desc="Loading V-JEPA model", leave=True) as pbar:
        processor = AutoVideoProcessor.from_pretrained(vjepa_model_name)
        pbar.update(1)
        model = AutoModel.from_pretrained(vjepa_model_name).to(device).eval()
        pbar.update(1)

    mapping = {
        "v-jepa2-vitl-enc-10layer-fc2": model.encoder.layer[10].mlp.fc2,
        "v-jepa2-vitl-enc-18layer-norm2": model.encoder.layer[18].norm2,
        "v-jepa2-vitl-enc-20layer-fc2": model.encoder.layer[20].mlp.fc2,
        "v-jepa2-vitl-enc-last-ln": model.encoder.layernorm,
        "v-jepa2-vitl-predictor-5-layer-norm": model.predictor.layer[5].norm1,
        "v-jepa2-vitl-pred-fc1": model.predictor.layer[11].mlp.fc1,
        "v-jepa2-vitl-final-features": model.predictor.proj,
    }

    store, handles = register_hooks(mapping)
    return {
        "processor": processor,
        "model": model,
        "mapping": mapping,
        "store": store,
        "handles": handles,
    }


@torch.no_grad()
def extract_video_features(
    video_path,
    video_file,
    output_folder,
    relative_path,
    tr,
    chunk_length,
    seconds_before_chunk,
    vjepa_bundle,
    device,
):
    chunks, chunk_of_interests = split_movie_into_chunks(
        video_path, tr, chunk_length, seconds_before_chunk
    )

    features = defaultdict(list)
    manifest_rows = []

    for tr_idx, ((c_start, c_end), (i_start, i_end)) in enumerate(
        tqdm(
            zip(chunks, chunk_of_interests),
            total=len(chunks),
            desc=f"Processing {os.path.basename(video_path)}",
        )
    ):
        clip_path = cut_clip(video_path, c_start, c_end)
        video_np = get_video_from_path(clip_path)
        video_t = torch.from_numpy(video_np).permute(0, 3, 1, 2)

        row = {
            "movie_file": video_file,
            "relative_path": relative_path,
            "video_path": video_path,
            "tr_index": int(tr_idx),
            "tr_start_sec": float(tr_idx * tr),
            "tr_end_sec": float((tr_idx * tr) + tr),
            "clip_start_sec": float(c_start),
            "clip_end_sec": float(c_end),
            "interest_start_in_clip_sec": float(i_start),
            "interest_end_in_clip_sec": float(i_end),
        }

        x_hf = vjepa_bundle["processor"](video_t, return_tensors="pt")["pixel_values_videos"].to(device)
        _ = vjepa_bundle["model"].get_vision_features(x_hf)

        current_vjepa_keys = []
        for k, tensors in vjepa_bundle["store"].items():
            slice_of_interest = crop_vjepa_by_time(tensors, i_start, i_end, c_end - c_start)
            features[k].append(slice_of_interest.cpu())
            current_vjepa_keys.append(k)

        row["vjepa_feature_keys"] = current_vjepa_keys
        manifest_rows.append(row)
        vjepa_bundle["store"].clear()
        os.remove(clip_path)

    saved_feature_paths = {}
    for k, v in features.items():
        features[k] = torch.stack(v, dim=0)

    for k, v in features.items():
        save_dir = os.path.join(output_folder, k, relative_path)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, video_file.rsplit(".", 1)[0] + ".pt")
        torch.save(v, save_path)
        saved_feature_paths[k] = save_path

    manifest_dir = os.path.join(output_folder, "manifests", relative_path)
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = os.path.join(manifest_dir, video_file.rsplit(".", 1)[0] + ".json")

    manifest = {
        "movie_file": video_file,
        "relative_path": relative_path,
        "video_path": video_path,
        "tr": float(tr),
        "chunk_length": float(chunk_length),
        "seconds_before_chunk": float(seconds_before_chunk),
        "num_tr_windows": int(len(manifest_rows)),
        "feature_paths": saved_feature_paths,
        "rows": manifest_rows,
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def list_video_files(input_folder, video_filter=None):
    """Walk input_folder for .mkv files.

    If video_filter is a list of paths (relative to input_folder/movies/),
    only those files are returned — no walking needed.
    """
    if video_filter is not None:
        video_files = []
        movies_folder = os.path.join(input_folder, "movies")
        for rel in video_filter:
            full = os.path.join(movies_folder, rel)
            if not os.path.isfile(full):
                raise FileNotFoundError(
                    f"Filtered video not found: {full}\n"
                    "Make sure you ran: datalad get stimuli/movies/<path>"
                )
            root = os.path.dirname(full)
            file = os.path.basename(full)
            video_files.append((root, file))
        return video_files

    video_files = []
    for root, _, files in os.walk(input_folder):
        if any(part.startswith(".") for part in root.split(os.sep)):
            continue
        for file in files:
            if file.endswith(".mkv"):
                video_files.append((root, file))
    random.seed(0)
    random.shuffle(video_files)
    return video_files


def shard_video_files(video_files, num_shards):
    return [list(x) for x in np.array_split(video_files, num_shards)]


def process_video_file_list(
    video_files,
    input_folder,
    output_folder,
    tr,
    chunk_length,
    seconds_before_chunk,
    vjepa_bundle,
    device,
):
    for root, file in tqdm(video_files, desc="Videos", unit="video"):
        relative_path = os.path.relpath(root, input_folder)
        video_path = os.path.join(root, file)
        stem = file.rsplit(".", 1)[0] + ".pt"

        requested_keys = list(vjepa_bundle["mapping"].keys())
        all_exist = True
        for key in requested_keys:
            if not os.path.isfile(os.path.join(output_folder, key, relative_path, stem)):
                all_exist = False
                break

        if all_exist:
            tqdm.write(f"  [SKIP] {file} — already embedded")
            continue

        tqdm.write(f"  [RUN]  {file}")
        extract_video_features(
            video_path=video_path,
            video_file=file,
            output_folder=output_folder,
            relative_path=relative_path,
            tr=tr,
            chunk_length=chunk_length,
            seconds_before_chunk=seconds_before_chunk,
            vjepa_bundle=vjepa_bundle,
            device=device,
        )


def run_worker(worker_rank, gpu_id, video_files, config, output_folder):
    log_dir = os.path.join(output_folder, "worker_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"worker_{worker_rank}_gpu{gpu_id}.log")

    try:
        local_device = get_torch_device(gpu_id if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)

        with open(log_path, "w") as logf:
            logf.write(f"[Worker {worker_rank}] Starting on {local_device} with {len(video_files)} videos\n")
            logf.flush()

        print(f"[Worker {worker_rank}] Starting on {local_device} with {len(video_files)} videos")

        vjepa_bundle = build_vjepa_bundle(
            vjepa_model_name=config["vjepa_model_name"],
            device=local_device,
        )

        process_video_file_list(
            video_files=video_files,
            input_folder=config["input_folder"],
            output_folder=output_folder,
            tr=config["tr"],
            chunk_length=config["chunk_length"],
            seconds_before_chunk=config["seconds_before_chunk"],
            vjepa_bundle=vjepa_bundle,
            device=local_device,
        )

        print(f"[Worker {worker_rank}] Finished on {local_device}")

    except Exception:
        import traceback
        tb = traceback.format_exc()
        print(f"[Worker {worker_rank}] FAILED on gpu {gpu_id}\n{tb}")
        with open(log_path, "a") as logf:
            logf.write("\n[EXCEPTION]\n")
            logf.write(tb)
            logf.flush()
        raise


def main():
    use_vjepa = CONFIG["use_vjepa"]
    use_ijepa = CONFIG["use_ijepa"]

    if not use_vjepa:
        raise ValueError("This script is V-JEPA-only. Set CONFIG['use_vjepa']=True.")
    if use_ijepa:
        raise ValueError("This script is V-JEPA-only. Set CONFIG['use_ijepa']=False.")

    output_folder = (
        f"{CONFIG['output_folder']}"
        f"_tr{CONFIG['tr']}_len{CONFIG['chunk_length']}_before{CONFIG['seconds_before_chunk']}"
        f"_vjepa1_ijepa0"
    )

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "worker_logs"), exist_ok=True)

    video_files = list_video_files(CONFIG["input_folder"], CONFIG.get("video_filter"))
    if CONFIG.get("num_chunks", 1) > 1:
        pre_chunks = np.array_split(video_files, CONFIG["num_chunks"])
        video_files = list(pre_chunks[CONFIG["chunk_id"]])

    parallelize_across_gpus = CONFIG.get("parallelize_across_gpus", False)
    gpu_ids = CONFIG.get("gpu_ids", [0])

    if parallelize_across_gpus and torch.cuda.is_available() and len(gpu_ids) > 1:
        num_available = torch.cuda.device_count()
        valid_gpu_ids = [g for g in gpu_ids if g < num_available]
        if not valid_gpu_ids:
            raise ValueError(
                f"None of the requested gpu_ids={gpu_ids} are available. Found {num_available} CUDA devices."
            )

        shards = shard_video_files(video_files, len(valid_gpu_ids))
        ctx = mp.get_context("spawn")
        procs = []

        for worker_rank, (gpu_id, shard) in enumerate(zip(valid_gpu_ids, shards)):
            p = ctx.Process(target=run_worker, args=(worker_rank, gpu_id, shard, CONFIG, output_folder))
            p.start()
            procs.append(p)

        exit_codes = []
        for p in procs:
            p.join()
            exit_codes.append(p.exitcode)

        if any(code != 0 for code in exit_codes):
            raise RuntimeError(f"One or more GPU workers failed. Exit codes: {exit_codes}")
    else:
        local_device = get_torch_device(0 if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

        vjepa_bundle = build_vjepa_bundle(
            vjepa_model_name=CONFIG["vjepa_model_name"],
            device=local_device,
        )

        process_video_file_list(
            video_files=video_files,
            input_folder=CONFIG["input_folder"],
            output_folder=output_folder,
            tr=CONFIG["tr"],
            chunk_length=CONFIG["chunk_length"],
            seconds_before_chunk=CONFIG["seconds_before_chunk"],
            vjepa_bundle=vjepa_bundle,
            device=local_device,
        )

    print(f"Done. Features saved under: {output_folder}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
