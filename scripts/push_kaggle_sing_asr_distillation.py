"""
SING-ASR Kaggle GPU Dispatcher & Monitor (Phase IX: Knowledge Distillation).
=============================================================================
Pushes, monitors, and retrieves results for PhoWhisper Cross-Attention Guided Distillation
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
MODELS_DIR = WORKSPACE_ROOT / "models" / "distilled_adapter_phowhisper"

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


def prepare_distillation_metadata(username: str):
    meta = {
        "id": f"{username}/sing-asr-cross-attention-distillation",
        "title": "SING-ASR Cross Attention Distillation Benchmark",
        "code_file": "sing_asr_distillation_kaggle.py",
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
    meta_path = KERNEL_DIR / "distillation-kernel-metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta_path, meta["id"]


def push_kernel(account_token=ACCOUNT1_TOKEN):
    env, username = get_kaggle_env(account_token)
    print(f"[*] Dang su dung tai khoan Kaggle: {username} ({account_token.parent.name})")
    
    meta_path, kernel_slug = prepare_distillation_metadata(username)
    # Temporarily copy to kernel-metadata.json for kaggle CLI
    standard_meta = KERNEL_DIR / "kernel-metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        content = f.read()
    with open(standard_meta, "w", encoding="utf-8") as f:
        f.write(content)

    cmd = [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(KERNEL_DIR)]
    print(f"[*] Thuc thi: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, encoding="utf-8", errors="replace")

    print("\n--- KET QUA DAY KERNEL ---")
    print(f"Return code: {res.returncode}")
    if res.stdout:
        print(f"Stdout:\n{res.stdout.strip()}")
    if res.stderr:
        print(f"Stderr:\n{res.stderr.strip()}")

    if res.returncode == 0:
        print(f"\n[OK] Da day thanh cong kernel: https://www.kaggle.com/code/{kernel_slug}")
        return True, kernel_slug, env
    else:
        print("[!] Day kernel that bai. Vui long kiem tra lai thong bao loi.")
        return False, None, env


def get_status(kernel_slug: str, env: dict):
    cmd = [sys.executable, "-m", "kaggle", "kernels", "status", kernel_slug]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, encoding="utf-8", errors="replace")
    if res.returncode == 0:
        return res.stdout.strip()
    return f"error: {res.stderr.strip()}"


def pull_outputs(kernel_slug: str, env: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "-m", "kaggle", "kernels", "output", kernel_slug, "-p", str(OUTPUT_DIR)]
    print(f"[*] Dang tai output ve: {OUTPUT_DIR} ...")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, encoding="utf-8", errors="replace")
    if res.returncode == 0:
        print("[OK] Da tai toan bo ket qua tu Kaggle ve may thanh cong!")
        res_json = OUTPUT_DIR / "exp9_kaggle_distillation_results.json"
        if res_json.exists():
            target_json = WORKSPACE_ROOT / "data" / "processed" / "exp9_kaggle_distillation_results.json"
            import shutil
            shutil.copy(res_json, target_json)
            print(f"  - Ket qua da luu vao: {target_json}")
        # Copy adapter files if present
        distilled_dir = OUTPUT_DIR / "distilled_adapter_phowhisper"
        if distilled_dir.exists():
            import shutil
            for f in distilled_dir.glob("*"):
                shutil.copy(f, MODELS_DIR / f.name)
            print(f"  - Trong so da luu vao: {MODELS_DIR}")
        return True
    else:
        print(f"[!] Tai output that bai: {res.stderr.strip()}")
        return False


def monitor_and_download(kernel_slug: str, env: dict, poll_interval_sec: int = 20, max_wait_sec: int = 1800):
    print(f"\n[*] Bat dau theo doi kernel: {kernel_slug}")
    start_time = time.time()

    while time.time() - start_time < max_wait_sec:
        st = get_status(kernel_slug, env)
        print(f"  [{time.strftime('%H:%M:%S')}] Trang thai: {st}")

        if "complete" in st.lower():
            print("\n[OK] Kernel da hoan thanh thanh cong!")
            pull_outputs(kernel_slug, env)
            return True
        elif "error" in st.lower() or "fail" in st.lower():
            print(f"\n[!] Kernel bi loi: {st}")
            return False

        time.sleep(poll_interval_sec)

    print("\n[!] Timeout cho doi kernel.")
    return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SING-ASR Kaggle Phase IX Dispatcher")
    parser.add_argument("--account", choices=["account1", "account2", "account3"], default="account1")
    parser.add_argument("--push-only", action="store_true")
    args = parser.parse_args()

    token_map = {
        "account1": ACCOUNT1_TOKEN,
        "account2": ACCOUNT2_TOKEN,
        "account3": ACCOUNT3_TOKEN
    }
    selected_token = token_map[args.account]

    success, slug, env = push_kernel(selected_token)
    if success and not args.push_only:
        monitor_and_download(slug, env)
