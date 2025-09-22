# Copilot Instructions — Solar PV Federated Anomaly Detection (VAE + SVDD + Federated Averaging)

> **Mục tiêu:** Sinh toàn bộ mã nguồn (module + CLI) để train mô hình phát hiện bất thường cho dữ liệu **solar PV** theo hướng **VAE → residual → SVDD (ngưỡng tự động) → Federated Averaging (FedAvg)**, kèm tiền xử lý đúng chuẩn từ *Users Manual TP-5200-61610.pdf*, khai thác **SandiaModelCoefficients.xlsx** (AOI & effective irradiance) và **CharacterDataForPVModels 9-26-14.xlsx** (hệ số nhiệt/IEC 61853-1), cùng phần **giải thích (XAI)** và **báo cáo**.

---

## 0) Yêu cầu chất lượng
- Viết code **Python 3.9+**, có **type hints**, **docstring** (Google style), chuẩn **PEP8**.
- Tách **module** và **CLI**, cấu hình qua **YAML**, logging rõ ràng.
- Không hard-code đường dẫn; mọi thứ đọc từ `configs/config.yaml`.
- Chịu lỗi thiếu cột (gợi ý map lại trong YAML). Có **fallback** khi thiếu hệ số Sandia/Characterization.
- Tránh **data leakage**: KHÔNG đưa các cột `Pm, Pm_calc, Imp, Vmp, Isc, Voc` vào tập đặc trưng dùng để huấn luyện VAE (chỉ dùng để trực quan/đánh giá).

---

## 1) Cấu trúc repo cần tạo
```
solar-pv-fed-anom/
  README.md
  pyproject.toml            # hoặc requirements.txt + setup.cfg
  configs/
    config.example.yaml
  solarpv/
    __init__.py
    io_schema.py            # mapping cột CSV theo YAML
    qc.py                   # QC/clean theo Users Manual
    features.py             # đặc trưng PV (Tcell, AOI, effective irradiance Sandia, ratios, time-features)
    preprocess.py           # pipeline đọc→QC→features→parquet
    vae.py                  # VAE (Keras mặc định) build/train/infer
    svdd.py                 # OneClassSVM wrapper + save/load
    fedavg.py               # FedAvg cho trọng số VAE
    explain.py              # SHAP (nếu có) + fallback residual variance
    report.py               # xuất HTML/PDF biểu đồ + summary
    utils.py                # seed, logging, timer, io helpers
  cli/
    spv_preprocess.py
    spv_train_local.py
    spv_federated.py
    spv_score_explain.py
    spv_report.py
  tests/
    test_features.py
    test_vae.py
  notebooks/ (optional)
```

---

## 2) File cấu hình (YAML)
Tạo `configs/config.example.yaml` (cho phép sao chép thành `configs/config.yaml`):

```yaml
data:
  sites:
    - {name: Cocoa,  path: data/raw/Cocoa}
    - {name: Eugene, path: data/raw/Eugene}
    - {name: Golden, path: data/raw/Golden}
  users_manual_pdf: "/mnt/data/Users Manual TP-5200-61610.pdf"
  sandia_coeffs_xlsx: "/mnt/data/SandiaModelCoefficients.xlsx"
  characterization_xlsx: "/mnt/data/CharacterDataForPVModels 9-26-14.xlsx"

preprocess:
  resample_rule: "1min"        # hoặc "5min"
  drop_maintenance_windows: true
  drop_when_precip_or_dew: true
  columns:
    datetime: "DateTime"
    site_id:  "Site"
    poa: "POA"
    ghi: "GHI"
    dhi: "DHI"
    dni: "DNI"
    t_back: "BackTemp"
    wind: "WindSpeed"
    imp: "Imp"
    vmp: "Vmp"
    pm:  "Pm"
    isc: "Isc"
    voc: "Voc"
    soiling_derate: "SoilingDerate"
    qa_residual:    "QA_Residual"
    maint_flag:     "Maintenance"
  out_dir: "data/processed"

model:
  backend: "keras"   # hoặc "torch" (nếu triển khai thêm)
  hidden_dim: 64
  latent_dim: 8
  epochs_local: 10
  batch_size: 256
  learning_rate: 1e-3

federated:
  rounds: 3
  frac_clients: 1.0

svdd:
  nu: 0.1
  gamma: "scale"

explain:
  shap: true
  shap_sample: 512
  top_k_features: 10

report:
  out_dir: "reports"
```

**Schema CSV mặc định** (có thể map lại trong YAML):  
`DateTime, Site, POA, GHI, DHI, DNI, BackTemp, WindSpeed, Imp, Vmp, Pm, Isc, Voc, SoilingDerate, QA_Residual, Maintenance`.

---

## 3) Tiền xử lý & QC (theo Users Manual)
Trong `solarpv/preprocess.py` và `solarpv/qc.py`:
- Đọc tất cả CSV trong `data/raw/<Site>/*.csv`, ép `DateTime` thành index (tz-naive OK).
- **QC tối thiểu**:
  - Loại khung **bảo trì** nếu `Maintenance == 1`.
  - Loại thời điểm **mưa/đọng sương** theo gợi ý Users Manual: có thể dựa vào **QA_Residual** (giữ trong ±3σ) và/hoặc cờ mưa nếu có.
  - Loại dòng có quá nhiều NaN (ví dụ >20% cột).
- **Resample** theo `resample_rule` (1min/5min). Cho phép forward-fill cho biến chậm (temp, wind), hạn chế fill cho irradiance/công suất.
- Lưu mỗi site thành Parquet: `data/processed/<Site>.parquet`.

---

## 4) Đặc trưng PV (features.py)
Triển khai các hàm:
- **Tcell**: xấp xỉ từ BackTemp & POA *hoặc* dùng **mô hình nhiệt Sandia** nếu có `a,b` trong `SandiaModelCoefficients.xlsx`.
- **AOI & Effective Irradiance (Sandia)**:
  - Đọc bộ hệ số AOI (A0..A4, B0..B5) theo **module identifier** (cho phép đặt trong YAML).
  - Tính **f(AOI)** và **Ee** theo Sandia. Nếu thiếu hệ số → fallback: bỏ AOI/Ee (đặt NaN an toàn).
- **Ratios**: DHI/GHI, DNI/GHI, POA/GHI (tránh chia cho 0).
- **Thời gian**: hour, day-of-week, sin/cos (giây trong ngày).
- **Pm_calc = Imp*Vmp** (chỉ để trực quan/đánh giá; KHÔNG dùng train).
- **Chuẩn hoá nhiệt** (tuỳ chọn): hàm đưa về STC dùng \(\alpha,\beta,\gamma\) đọc từ `CharacterData...xlsx` (bật qua YAML).

Kết quả: trả về ma trận numeric **X** đã chọn cột cho VAE (loại `Pm, Pm_calc, Imp, Vmp, Isc, Voc`).

---

## 5) VAE (vae.py)
- Backend mặc định **Keras (TF2)**:
  - Kiến trúc: `input_dim → Dense(hidden_dim, ReLU) → (z_mean, z_logvar)`; sampling; `Dense(hidden_dim, ReLU) → Dense(input_dim)`.
  - Loss: **MSE recon** + `1e-3 * KL`.
  - Optimizer: Adam(lr).
- API gợi ý:
  - `build_vae(input_dim, hidden_dim, latent_dim, lr) -> (vae_model, encoder_model)`
  - `train_vae(X_train, epochs, batch_size) -> weights`
  - `infer_reconstruction(X) -> X_recon`

---

## 6) SVDD (svdd.py)
- Dùng **One-Class SVM (RBF)** làm xấp xỉ (nu từ YAML, gamma=scale).  
- Huấn luyện trên **residual**: `resid = Xs - VAE(Xs)`.
- API:
  - `fit_svdd(residuals, nu, gamma) -> model`
  - `score_svdd(model, residuals) -> scores` (decision_function)
  - `predict_svdd(model, residuals) -> labels` (-1 anomaly, 1 normal)

---

## 7) Huấn luyện cục bộ per-site (cli/spv_train_local.py)
- Đối với từng site:
  1. Nạp `data/processed/<Site>.parquet` → chọn **numeric features** hợp lệ.
  2. Chuẩn hoá **StandardScaler** theo site.
  3. Train **VAE** (dữ liệu “normal” sau QC).
  4. Tính **residual** và train **SVDD**.
  5. Lưu artefact:
     - `models/<Site>_scaler.pkl`
     - `models/<Site>_vae_weights.pkl`
     - `models/<Site>_columns.json`
     - `models/<Site>_svdd.pkl`

Tham số đọc từ `model.*` và `svdd.*` trong YAML.

---

## 8) Federated Averaging (solarpv/fedavg.py, cli/spv_federated.py)
- Đọc tất cả `<Site>_vae_weights.pkl` → **FedAvg** theo từng tensor (trung bình).
- Ghi `models/global_vae_weights.pkl`.
- Hỗ trợ **multi-round**: lặp lại (global→fine-tune local vài epoch→gửi lại→avg).

---

## 9) Suy luận & Giải thích (cli/spv_score_explain.py, solarpv/explain.py)
- Nạp scaler/columns + dựng lại VAE đúng kiến trúc.
- Nếu có `global_vae_weights.pkl` → ưu tiên nạp; nếu không → dùng local weights.
- Tính **residual** cho toàn bộ timeline của site → **scores** bằng SVDD.
- **Gắn nhãn**: `labels = SVDD.predict(residual)`.
- **Giải thích**:
  - Nếu `explain.shap == true` và cài `shap`: chạy SHAP (Kernel/DeepExplainer phù hợp) trên **residual** hoặc **đầu vào** (chọn một, ghi rõ trong README).
  - Nếu không có SHAP: fallback **residual variance** theo feature → top-k.
- Xuất:
  - `scores/<Site>_scores.parquet` (timestamp, score, label)
  - `explain/<Site>_top_features.csv`

---

## 10) Báo cáo (solarpv/report.py, cli/spv_report.py)
- Sinh biểu đồ:
  - Timeline `POA, Pm, Pm_calc` overlay **anomaly flags**.
  - Histogram/QQ plot **scores**.
  - Bar chart **top features** theo site.
- Dùng **jinja2** tạo `reports/<YYYYMMDD>_summary.html` tổng hợp:
  - Số/tỷ lệ điểm bất thường theo ngày/giờ.
  - Tương quan với **Maintenance** (trước/sau bảo trì).
  - So sánh giữa các site.

---

## 11) README & ví dụ lệnh
Thêm hướng dẫn tối thiểu trong `README.md`:

```bash
# 0) Cài môi trường
pip install -r requirements.txt
# 1) Cấu hình
cp configs/config.example.yaml configs/config.yaml
# 2) Tiền xử lý & QC
python -m cli.spv_preprocess --config configs/config.yaml
# 3) Train local per-site
python -m cli.spv_train_local --config configs/config.yaml
# 4) Federated Averaging
python -m cli.spv_federated --config configs/config.yaml
# 5) Suy luận + Giải thích
python -m cli.spv_score_explain --config configs/config.yaml --site Cocoa --save-scores
# 6) Báo cáo
python -m cli.spv_report --config configs/config.yaml
```

---

## 12) Kiểm thử, reproducibility
- `utils.set_seed(42)` cho numpy/tf.
- `tests/test_features.py`: kiểm tra AOI/Ee không NaN khi có hệ số; ratios không chia 0.
- `tests/test_vae.py`: kiểm tra shapes encoder/decoder, loss giảm qua vài batch.
- GitHub Actions đơn giản: lint + chạy test nhanh.

---

## 13) (Tuỳ chọn) Docker & requirements
- `requirements.txt`:
  ```
  numpy
  pandas
  scipy
  scikit-learn
  pyyaml
  tensorflow>=2.10
  shap
  matplotlib
  plotly
  jinja2
  pyarrow
  ```
- Dockerfile CPU (python:3.11-slim), cài deps, entrypoint cho CLI.

---

## 14) Dữ liệu & thực hành tốt
- **Normal windows** để train VAE: chọn giờ phát điện ổn định, sau vệ sinh, không mưa/đọng sương.
- **Tuning SVDD**: `nu` ~ 0.05–0.15; có **hysteresis** (N điểm liên tiếp) khi cảnh báo.
- **Đánh giá**: nếu có ghi chú bảo trì/lau rửa, so sánh tỷ lệ cảnh báo **trước** mốc này (soiling/fault).

---

**Hãy tạo đầy đủ tất cả file/mã theo đặc tả trên, sẵn sàng chạy.**
