# AURA Project Rule — Bắt Buộc Sử Dụng Dữ Liệu Thực Tế (Strict Real-World Data Policy)

## 1. Nguyên Tắc Cốt Lõi (Mandatory Ground Rule)
- **TUYỆT ĐỐI KHÔNG SỬ DỤNG DỮ LIỆU TỰ TẠO (SYNTHETIC / SELF-CREATED DATA)** để:
  1. Đánh giá độ chính xác, hiệu năng (evaluation benchmarks, metrics F1, accuracy, AUC).
  2. Báo cáo kết quả nghiên cứu khoa học hoặc rút ra kết luận khoa học cho dự án AURA.
  3. Làm cơ sở chứng minh hay biện giải cho các câu hỏi nghiên cứu (RQ0 — RQ4, AIT, SIB).

## 2. Lý Do Khoa Học & Tính Hợp Thức (Scientific Rationale)
- Dữ liệu tự tạo (do LLM sinh hoặc do thuật toán tạo tín hiệu nhân tạo) không phản ánh đúng phân phối sinh học, độ nhiễu thực tế, tính đa dạng ngữ âm, và sự biến thiên phức tạp của bệnh lý tâm thần hoặc hiện tượng tâm lý người thật.
- Đánh giá trên dữ liệu tự tạo tạo ra "ảo tưởng hiệu năng" (performance illusion), dẫn đến kết quả sai lệch và không thể công bố quốc tế.

## 3. Quy Định Thực Thi Bắt Buộc (Enforcement)
- **Mọi thực nghiệm chính thức (Official Experiments)** từ Milestone M4, M7, M8, M9, M10, M14 bắt buộc phải chạy trên các bộ dữ liệu thực tế, đã qua thẩm định hoặc công bố bình duyệt (peer-reviewed / public clinical corpora):
  - **Speech & Multimodal:** DAIC-WOZ (USC ICT), RAVDESS (Zenodo/Kaggle), CREMA-D, IEMOCAP, VLSP Audio Corpus.
  - **Text & Clinical Language:** UMD Reddit Suicidality Corpus (C-SSRS), SMHD (Schizophrenia & Mental Health), CLPsych benchmarks.
  - **EEG / Physiological:** WESAD, PhysioNet Sleep/EEG benchmarks, CAPS normative distributions.
- **Dữ liệu tổng hợp (Synthetic Data):**
  - Chỉ được coi là công cụ kiểm tra cú pháp mã nguồn (Code syntax unit tests) trong quá trình debug, **không bao giờ** được ghi vào bảng kết quả hay báo cáo khoa học.
  - Bộ dữ liệu `data/benchmark_narratives_v1.jsonl` được đánh dấu là `ARCHIVED_SYNTHETIC_UNIT_TEST_ONLY` và phải được thay thế hoàn toàn bằng dữ liệu thực tế khi thực nghiệm chính thức.

## 4. Cơ Chế Tính Toán (Compute Strategy)
- Do tài nguyên máy cục bộ hạn chế (RAM 8GB đang quá tải, GPU MX350 2GB không tương thích CUDA PyTorch hiện đại), toàn bộ quá trình xử lý tập dữ liệu lớn và huấn luyện mô hình sâu (WavLM, HuBERT, Whisper, PhoBERT) sẽ được đẩy lên **Kaggle GPU Cluster (T4 x2 / P100 16GB VRAM)** thông qua hệ thống `kaggle_runner_ui` tích hợp 3 tài khoản Kaggle của dự án.
