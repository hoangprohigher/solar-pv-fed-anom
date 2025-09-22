# Solar PV Federated Anomaly Detection (VAE + SVDD + FedAvg)

Pipeline phát hiện bất thường cho dữ liệu **solar PV** theo hướng: **VAE → residual → SVDD → Federated Averaging**,
kèm QC theo Users Manual, đặc trưng/nhãn từ Sandia & CharacterData, hỗ trợ **train theo Module**.

## Quick start
```bash
python -m pip install -r requirements.txt

cp configs/config.example.yaml configs/config.yaml

# Build module map từ hai file XLSX (Sandia + CharacterData)
python -m cli.spv_build_module_map --config configs/config.yaml --out configs/module_map.json

# Preprocess & QC (tự gắn Site/Module, gắn nhãn từ module_map.json nếu có)
python -m cli.spv_preprocess --config configs/config.yaml

# Train 1 model / Module
python -m cli.spv_train_by_module --config configs/config.yaml

# (Optional) Federated averaging các client weights
python -m cli.spv_federated --config configs/config.yaml

# Score + Explain một module cụ thể
python -m cli.spv_score_explain --config configs/config.yaml --site Cocoa --module mSi0166 --save-scores

# Báo cáo nhanh
python -m cli.spv_report --config configs/config.yaml
```

Cấu trúc dữ liệu mong đợi (ví dụ):
```
DATA FOR VALIDATING MODELS/
  Cocoa/
    Cocoa_mSi0166.csv
    Cocoa_CdTe75638.csv
    ...
  Eugene/
    Eugene_mSi0166.csv
    ...
  Golden/
    Golden_mSi0247.csv
    ...
CharacterDataForPVModels 9-26-14.xlsx
SandiaModelCoefficients.xlsx
Users Manual TP-5200-61610.pdf
```
