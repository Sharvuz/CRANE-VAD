# CRANE-VAD: Counterfactual Residual Alignment Network with Evidential Fusion

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle: Dual Tesla T4](https://img.shields.io/badge/Kaggle-Dual%20Tesla%20T4%20Ready-20BEFF.svg?logo=kaggle&logoColor=white)](https://kaggle.com)
[![Research: HCMCOU](https://img.shields.io/badge/Research-HCMCOU-green.svg)]()

> **100% Standalone Weakly-Supervised Video Anomaly Detection without Warm-Start Dependencies**  
> **Tác giả:** Huỳnh Quang Vũ (MSSV: 2351010247)  
> **Giảng viên hướng dẫn:** TS. Lê Viết Tuấn  
> **Khoa Công nghệ Thông tin — Trường Đại học Mở TP. Hồ Chí Minh (HCMCOU)**

---

## 📖 1. Giới thiệu & Động lực Khoa học (Executive Motivation)

Trong bài toán Phát hiện Bất thường Video Giám sát Yếu (WS-VAD), mô hình chỉ nhận nhãn mức video $Y \in \{0, 1\}$ mà không có mốc thời gian chi tiết của sự cố.

### Hạn chế cốt tử của các thế hệ trước (DSANet & RUPA-DSA):
1. **Bế tắc của Warm-Start (The Warm-Start Dilemma):** Trong phiên bản trước (RUPA-DSA), khi huấn luyện từ đầu (End-to-End from scratch), mô hình rơi vào hiện tượng sụp đổ đạo hàm (**Gradient Collapse**). Nguyên nhân do bộ phát hiện ban đầu xuất ra điểm số ngẫu nhiên, khiến thuật toán phân ngưỡng (Top-K / Otsu) chọn nhầm các khung hình tội phạm đưa vào ngân hàng nguyên mẫu $P$ (**Prototype Contamination**). Để đồ án hoạt động, tác giả phải dùng giải pháp tình thế là đóng băng mạng DSANet đã huấn luyện trước (`main_lr = 0.0`).
2. **Kiến trúc cồng kềnh phụ thuộc:** DSANet xếp chồng Attention cục bộ với mạng Đồ thị tích chập kép (Distance GCN + Similarity GCN), tạo ra độ phức tạp lớn $O(T^2)$ và làm nhiễu gradient khi tối ưu từ đầu.
3. **Nhiễu nền trong các mô hình VLM (VadCLIP, Alert-CLIP):** Hơn 90% thông tin trong video giám sát là phông nền tĩnh (đường sá, bờ tường). Khi chiếu trực tiếp đặc trưng thô $X$ vào không gian văn bản CLIP, mô hình rất dễ báo động giả vì nhận diện nhầm bối cảnh.

---

## 💡 2. Kiến trúc Đột phá CRANE-VAD

**CRANE-VAD** loại bỏ hoàn toàn mạng cơ sở DSANet, Dual GCN và sự phụ thuộc vào Warm-Start:

```
                     ┌─────────────────────────────────────────┐
                     │   Frozen CLIP ViT-B/16 Features X (512) │
                     └────────────────────┬────────────────────┘
                                          │
                     ┌────────────────────▼────────────────────┐
                     │   Bidirectional S4D Temporal Encoder    │
                     │    Linear Complexity O(T), State N=64   │
                     └────────────────────┬────────────────────┘
                                          │
                                 Temporal Features H
                                          │
             ┌────────────────────────────┼────────────────────────────┐
             │                            │                            │
             ▼                            ▼                            ▼
   ┌───────────────────┐        ┌───────────────────┐        ┌───────────────────┐
   │  Branch 1:        │        │  Branch 2 (LGAM): │        │  Branch 3:        │
   │  Discriminative   │        │  Label-Gated Asym.│        │  Residual-to-Text │
   │  MIL Head         │        │  Normal Memory    │        │  Counterfactual   │
   │                   │        │                   │        │  Alignment        │
   │  Output:          │        │  Normal Rec: F_rec│        │  Projection to    │
   │  S_det ∈ [0, 1]   │        │  Clean Res: R     │        │  CLIP Crime Space │
   │  Beta: α_d, β_d   │        │  S_rec ∈ [0, 1]   │        │  S_sem ∈ [0, 1]   │
   │  Uncert: σ²_d     │        │  Beta: α_r, β_r   │        │  Beta: α_s, β_s   │
   └─────────┬─────────┘        └─────────┬─────────┘        └─────────┬─────────┘
             │                            │                            │
             └────────────────────────────┼────────────────────────────┘
                                          ▼
                        ┌───────────────────────────────────┐
                        │    Beta-Evidential Dynamic Gate   │
                        │    Inverse-Variance Precision     │
                        │    w_i = (1/σ²_i) / Σ (1/σ²_j)    │
                        └─────────────────┬─────────────────┘
                                          ▼
                        ┌───────────────────────────────────┐
                        │      Final Score S_final ∈ [0, 1] │
                        │  S_final = w_d·S_d + w_r·S_r +    │
                        │            w_s·S_s                │
                        └───────────────────────────────────┘
```

### 4 Đột phá Cốt lõi (Scientific Novelties):
* **Label-Gated Asymmetric Memory (LGAM):** Trong WS-VAD, video bình thường ($Y=0$) chứa 100% khung hình sạch (Ground Truth). LGAM **chỉ cho phép video $Y=0$ cập nhật gradient vào Prototype $P$**; với video bất thường ($Y=1$), đường truyền gradient đến $P$ bị ngắt hoàn toàn qua phép tách `detach()`. Do đó, **nguy cơ nhiễm độc nguyên mẫu bằng 0 tuyệt đối từ Epoch 0**.
* **Bidirectional S4D Backbone ($O(T)$):** Mô hình hóa không gian trạng thái liên tục 2 chiều với khởi tạo HiPPO, độ phức tạp $O(T \log T)$ qua biến đổi FFT, nhẹ hơn 430 lần so với Self-Attention.
* **Counterfactual Residual-to-Text Alignment:** Khử sạch >90% bối cảnh tĩnh $R = H - F_{rec}$ trước khi chiếu vào không gian văn bản của 14 prompt tội phạm CLIP.
* **Beta-Evidential Dynamic Gate:** Đầu ra của mỗi nhánh là phân phối Beta $(\alpha, \beta)$, cho phép đo lường trực tiếp độ bất định Epistemic $\sigma^2$. Cổng tự động dồn trọng số cho nhánh tin cậy nhất mà không cần trick bias âm heuristic.

---

## 📊 3. Mục tiêu Benchmark

| Phương pháp | Backbone | Cần Warm-Start? | UCF-Crime ROC-AUC | XD-Violence AP |
| :--- | :---: | :---: | :---: | :---: |
| Sultani et al. (CVPR'18) | C3D | Không | 75.41% | 73.20% |
| RTFM (ICCV'21) | I3D | Không | 84.30% | 77.81% |
| DSANet Baseline (AAAI'26) | CLIP ViT-B | Không | 87.55% | 85.00% |
| RUPA-DSA cũ (Đồ án ngành) | CLIP + DSANet | **Bắt buộc (Đóng băng)** | 88.19% | 85.26% |
| **CRANE-VAD (Đề xuất)** | **CLIP + BiS4D** | **100% From Scratch** | **≥ 88.50%** | **≥ 86.50%** |

---

## 📂 4. Cấu trúc Mã nguồn (Project Structure)

```
CRANE-VAD/
├── README.md                      # Tài liệu tổng quan kiến trúc
├── requirements.txt               # Thư viện phụ thuộc
├── crane_kaggle.ipynb             # Notebook chạy toàn diện trên Kaggle Dual T4
├── list/                          # File danh sách CSV và Ground Truth
│   ├── ucf_CLIP_rgb.csv
│   ├── ucf_CLIP_rgbtest.csv
│   ├── gt_ucf.npy
│   ├── xd_CLIP_rgb.csv
│   ├── xd_CLIP_rgbtest.csv
│   └── gt.npy
├── scripts/
│   ├── prepare_text_embeddings.py # Tiền mã hóa Offline Text Embeddings
│   └── extract_clip_features.py   # Công cụ trích xuất CLIP từ file video MP4
└── src/
    ├── __init__.py
    ├── model.py                   # Kiến trúc BiS4D, LGAM, R2T và Evidential Gate
    ├── losses.py                  # Hệ thống hàm mất mát đa mục tiêu
    ├── dataset.py                 # Dataset & DataLoader tối ưu hóa bộ nhớ
    ├── config.py                  # Quản lý siêu tham số huấn luyện
    ├── text_prompts.py            # Prompt 14 lớp UCF và 7 lớp XD
    ├── train_ucf.py               # Huấn luyện UCF-Crime từ scratch
    ├── test_ucf.py                # Đánh giá ROC-AUC / AP trên UCF-Crime
    ├── train_xd.py                # Huấn luyện XD-Violence trên Dual T4
    ├── test_xd.py                 # Đánh giá AP trên XD-Violence
    └── utils/
        ├── tools.py               # Xử lý chuỗi thời gian, padding, splitting
        └── eval_utils.py          # Tính toán ROC-AUC và Average Precision
```

---

## 🚀 5. Hướng dẫn Chạy Thực nghiệm trên Kaggle (Dual Tesla T4)

### Bước 1: Mở Notebook trên Kaggle
Mở file [`crane_kaggle.ipynb`](crane_kaggle.ipynb) trong Kaggle Notebook. Chọn cấu hình:
- **Accelerator:** `GPU T4 x2`
- **Internet:** `On` (khi cài thư viện ban đầu)

### Bước 2: Huấn luyện trên UCF-Crime từ Scratch
```bash
python -u -m src.train_ucf \
    --dataset ucf \
    --batch-size 64 \
    --epochs 45 \
    --lr 1e-4 \
    --lr-proto 5e-4 \
    --n-prototypes 16 \
    --eval-freq 5 \
    --model-save-dir models/ucf
```

### Bước 3: Huấn luyện trên XD-Violence trên Dual T4 (Batch Size 96)
```bash
python -u -m src.train_xd \
    --dataset xd \
    --batch-size 96 \
    --epochs 25 \
    --lr 1e-4 \
    --lr-proto 5e-4 \
    --n-prototypes 16 \
    --eval-freq 5 \
    --model-save-dir models/xd
```

---

## 📜 6. Trích dẫn (Citation)

Nếu bạn sử dụng kiến trúc CRANE-VAD trong nghiên cứu của mình, vui lòng trích dẫn:

```bibtex
@article{huynh2026cranevad,
  title={CRANE-VAD: Counterfactual Residual Alignment Network with Evidential Fusion for Weakly-Supervised Video Anomaly Detection},
  author={Huynh, Quang-Vu and Le, Viet-Tuan},
  journal={Ho Chi Minh City Open University (HCMCOU)},
  year={2026}
}
```
