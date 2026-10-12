"""
SING-ASR Kaggle GPU Dispatcher & Monitor (Phase VIII).
======================================================
Pushes, monitors, and retrieves results for PhoWhisper LoRA Fine-Tuning
using the existing Kaggle runner credentials from uwir_resfes/kaggle_runner_ui.
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
MODELS_DIR = WORKSPACE_ROOT / "models" / "lora_adapter_phowhisper"

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

    # Use actual username for kernel slug
    actual_username = creds.get("username", "thung192")
    return env, actual_username


def push_kernel(account_token=ACCOUNT1_TOKEN):
    env, username = get_kaggle_env(account_token)
    print(f"[*] Đang sử dụng tài khoản Kaggle: {username} ({account_token.parent.name})")
    print(f"[*] Thư mục kernel: {KERNEL_DIR}")

    # Read kernel metadata
    meta_file = KERNEL_DIR / "kernel-metadata.json"
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    kernel_slug = meta["id"]

    cmd = [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(KERNEL_DIR)]
    print(f"[*] Thực thi: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, encoding="utf-8", errors="replace")

    print("\n--- KẾT QUẢ ĐẨY KERNEL ---")
    print(f"Return code: {res.returncode}")
    if res.stdout:
        print(f"Stdout:\n{res.stdout.strip()}")
    if res.stderr:
        print(f"Stderr:\n{res.stderr.strip()}")

    if res.returncode == 0:
        print(f"\n[OK] Đã đẩy thành công kernel: https://www.kaggle.com/code/{kernel_slug}")
        return True, kernel_slug, env
    else:
        print("[!] Đẩy kernel thất bại. Vui lòng kiểm tra lại thông báo lỗi.")
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
    print(f"[*] Đang tải output về: {OUTPUT_DIR} ...")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, encoding="utf-8", errors="replace")
    if res.returncode == 0:
        print("[OK] Đã tải toàn bộ kết quả từ Kaggle về máy thành công!")
        # Copy results JSON to data/processed
        res_json = OUTPUT_DIR / "exp8_kaggle_lora_results.json"
        if res_json.exists():
            target_json = WORKSPACE_ROOT / "data" / "processed" / "exp8_kaggle_lora_results.json"
            import shutil
            shutil.copy(res_json, target_json)
            print(f"  - Kết quả đã lưu vào: {target_json}")
        return True
    else:
        print(f"[!] Tải output thất bại: {res.stderr.strip()}")
        return False


def monitor_and_download(kernel_slug: str, env: dict, poll_interval_sec: int = 20, max_wait_sec: int = 1800):
    print(f"\n[*] Bắt đầu theo dõi kernel: {kernel_slug}")
    start_time = time.time()

    while time.time() - start_time < max_wait_sec:
        st = get_status(kernel_slug, env)
        print(f"  [{time.strftime('%H:%M:%S')}] Trạng thái: {st}")

        if "complete" in st.lower():
            print("\n[OK] Kernel đã hoàn thành thành công!")
            pull_outputs(kernel_slug, env)
            return True
        elif "error" in st.lower() or "fail" in st.lower():
            print(f"\n[!] Kernel bị lỗi: {st}")
            return False

        time.sleep(poll_interval_sec)

    print("\n[!] Hết thời gian chờ (timeout). Bạn có thể chạy lại để tiếp tục theo dõi.")
    return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SING-ASR Kaggle LoRA Dispatcher")
    parser.add_argument("--push", action="store_true", help="Push kernel to Kaggle")
    parser.add_argument("--monitor", action="store_true", help="Monitor kernel until complete")
    parser.add_argument("--status", action="store_true", help="Check current status")
    parser.add_argument("--download", action="store_true", help="Download output artifacts")
    args = parser.parse_args()

    meta_file = KERNEL_DIR / "kernel-metadata.json"
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    slug = meta["id"]
    env, _ = get_kaggle_env(ACCOUNT1_TOKEN)

    if args.status:
        st = get_status(slug, env)
        print(f"Trạng thái: {st}")
    elif args.download:
        pull_outputs(slug, env)
    elif args.monitor and not args.push:
        monitor_and_download(slug, env)
    else:
        ok, kslug, kenv = push_kernel(ACCOUNT1_TOKEN)
        if ok and args.monitor:
            monitor_and_download(kslug, kenv)

