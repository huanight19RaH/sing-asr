#!/usr/bin/env python3
"""
Singing ASR Manuscript Overleaf Packager
=======================================
Tự động đóng gói toàn bộ thư mục docs/paper/ thành file sing_asr_manuscript_overleaf.zip
để người dùng có thể tải thẳng lên Overleaf bằng 1 click chuột.
Bỏ qua các file rác biên dịch (.aux, .log, .out, .bbl, .blg, v.v.).
"""

import os
import sys
import zipfile
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_DIR = WORKSPACE_ROOT / "docs" / "paper"
OUTPUT_ZIP = MANUSCRIPT_DIR / "sing_asr_manuscript_overleaf.zip"

EXCLUDE_EXTENSIONS = {'.aux', '.log', '.out', '.blg', '.bbl', '.synctex.gz', '.html'}
EXCLUDE_NAMES = {'sing_asr_manuscript_overleaf.zip', 'Thumbs.db', '.DS_Store'}

def create_overleaf_zip():
    print(f"[*] Dang dong goi du an LaTeX tu: {MANUSCRIPT_DIR}")
    
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
        
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(MANUSCRIPT_DIR):
            for file in files:
                file_path = Path(root) / file
                
                if file in EXCLUDE_NAMES or file_path.suffix in EXCLUDE_EXTENSIONS:
                    continue
                if file_path == OUTPUT_ZIP:
                    continue
                
                rel_path = file_path.relative_to(MANUSCRIPT_DIR)
                zipf.write(file_path, arcname=str(rel_path))
                
    print(f"\n[OK] Da tao thanh cong goi Overleaf: {OUTPUT_ZIP}")
    print(f"[*] Kich thuoc: {OUTPUT_ZIP.stat().st_size / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    create_overleaf_zip()
