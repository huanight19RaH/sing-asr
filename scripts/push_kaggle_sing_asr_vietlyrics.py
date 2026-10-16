"""
SING-ASR Kaggle GPU Dispatcher & Monitor (Phase XV: VietLyrics Macro-Scale Pretraining).
======================================================================================
Pushes, monitors, and retrieves results for VietLyrics 647h Pretraining with Dual T4 GPUs
using Kaggle runner credentials from uwir_resfes/kaggle_runner_ui.
Lead Researcher: Thai Hung Nguyen (AIT Laboratory, Department of Information Technology).
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Directories
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = WORKSPACE_ROOT / "kaggle_kernel"
OUTPUT_DIR = WORKSPACE_ROOT / "data" / "processed" / "kaggle_outputs"
MODELS_DIR = WORKSPACE_ROOT / "models" / "macro_vietlyrics_phowhisper"

TOKEN_DIR_BASE = Path(r"D:\THStudy\UniversityStudy\Research\uwir_resfes\underwater-image-enhancement\kaggle_runner_ui\data\tokens")
ACCOUNT1_TOKEN = TOKEN_DIR_BASE / "account1" / "kaggle.json"
ACCOUNT2_TOKEN = TOKEN_DIR_BASE / "account2" / "kaggle.json"
ACCOUNT3_TOKEN = TOKEN_DIR_BASE / "account3" / "kaggle.json"


def get_kaggle_env(token_path: Path):
    if not token_path.exists():
        raise FileNotFoundError(f"Kaggle token not found at: {token_path}")
    with open(token_path, "r", encoding="utf-8") as f:
        creds = json.load(f)

    # Sync ~/.kaggle/access_token
    access_token_file = Path.home() / ".kaggle" / "access_token"
    access_token_file.parent.mkdir(parents=True, exist_ok=True)
    with open(access_token_file, "w", encoding="utf-8") as f:
        f.write(creds["key"].strip())

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["KAGGLE_CONFIG_DIR"] = str(token_path.parent)
    env["KAGGLE_USERNAME"] = creds["username"]
    env["KAGGLE_KEY"] = creds["key"]
    env["KAGGLE_API_TOKEN"] = creds["key"]

    actual_username = creds.get("username", "thung192")
    return env, actual_username


def prepare_vietlyrics_metadata(username: str):
    meta = {
        "id": f"{username}/sing-asr-vietlyrics-pretraining",
        "title": "SING-ASR VietLyrics Macro Scale Pretraining",
        "code_file": "sing_asr_vietlyrics_pretrain.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [
            f"{username}/sing-asr-benchmark-data"
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }
    meta_path = KERNEL_DIR / "kernel-metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[*] Prepared Kernel Metadata: {meta['id']}")
    return meta["id"]


def push_kernel(env: dict):
    print("[*] Pushing Phase XV Kernel to Kaggle GPU Runner...")
    cmd = ["kaggle", "kernels", "push", "-p", str(KERNEL_DIR)]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print("STDOUT:", res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)
    return res.returncode == 0


def main():
    print("================================================================================")
    print("SING-ASR Kaggle Phase XV Dispatcher: VietLyrics 647h Pretraining")
    print("Lead Researcher: Thai Hung Nguyen (AIT Laboratory, Dept of IT)")
    print("================================================================================")

    # Check token
    target_token = ACCOUNT1_TOKEN if ACCOUNT1_TOKEN.exists() else (ACCOUNT2_TOKEN if ACCOUNT2_TOKEN.exists() else ACCOUNT3_TOKEN)
    if not target_token.exists():
        print("[!] No Kaggle token found. Dry run mode enabled.")
        return

    env, username = get_kaggle_env(target_token)
    print(f"[*] Using Kaggle Account: {username}")

    kernel_id = prepare_vietlyrics_metadata(username)
    success = push_kernel(env)
    if success:
        print(f"[+] Successfully pushed {kernel_id} to Kaggle!")
    else:
        print("[!] Failed to push kernel to Kaggle.")


if __name__ == "__main__":
    main()
