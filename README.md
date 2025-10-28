# 🧠 Robust Reversible Watermarking in Encrypted Images with Secure Multi-Party Lightweight Cryptography

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)]()
[![AES-GCM](https://img.shields.io/badge/Crypto-AES--GCM-green.svg)]()
[![RRW](https://img.shields.io/badge/Algorithm-Reversible%20Watermarking-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

> **Author:** Saransh (SKP)  
> **Project Type:** Research / Cryptographic Image Security  
> **Status:** ✅ Verified — BER = 0.0  |  PSNR ≈ 50 dB  |  Perfect SHA-256 Reversibility  

---

## 🧩 Abstract

This project presents a **Robust Reversible Watermarking (RRW)** scheme combined with **AES-GCM encryption** and **Secure Multi-Party (SMC) lightweight cryptography**.  
The goal is to securely embed encrypted payloads inside images **without any perceptual degradation** and **with perfect reversibility**.

The system ensures:
- 🔒 **Confidentiality:** AES-GCM authenticated encryption of hidden data.  
- ⚙️ **Reversibility:** Exact recovery of the original cover image (BER = 0).  
- 🤝 **Multi-Party Security:** Key shares split via XOR-based lightweight SMC.  
- 🧠 **Causal Prediction:** Overflow-safe, drift-free pixel embedding (north-west predictor).  
- 📊 **Integrity Validation:** SHA-256 verification of recovered cover.

---

## 🧱 System Architecture

[Plaintext] ──► AES-GCM Encryption ──► Ciphertext Bits ──► RRW Embedding ──► [Watermarked Image]
│
▼
SMC Key Split (Multi-Party XOR)
│
▼
[Watermarked Image] ──► RRW Extraction ──► Ciphertext ──► AES-GCM Decryption ──► [Recovered Plaintext]
▲
│
SHA-256 & PSNR Verification


---

## ⚙️ Features

| Module | Functionality |
|---------|----------------|
| **`crypto.py`** | AES-GCM encryption/decryption with Scrypt key derivation |
| **`smc.py`** | XOR-based Secure Multi-Party key split & combine |
| **`watermark_rrw.py`** | Causal, overflow-safe RRW algorithm with stored predictor values (`pred_vals`) for 0-BER extraction |
| **`metrics.py`** | PSNR, BER, and SSIM utilities |
| **`main.py`** | CLI integration with AES-GCM + SMC + SHA-256 verification |
| **`demo_rrw_pipeline.py`** | Stand-alone demo pipeline for quick testing |

---

## 🧪 Verification Results

| Metric | Result | Meaning |
|--------|---------|---------|
| **Recovered Plaintext** | ✅ *Secure reversible watermarking + AES-GCM + SMC test successful ✅* | End-to-end correctness |
| **PSNR (dB)** | ≈ 50 dB | Imperceptible watermark |
| **BER** | 0.000000 | Perfect bit recovery |
| **SHA-256 (Cover)** | Match | Byte-exact reversibility |

---

## 🧰 Installation

```bash
# Clone repo
git clone https://github.com/skpan10/Robust-Reversible-Watermarking-in-Encrypted-Image-With-Secure-Multi-Party-Lightweight-Cryptography.git
cd Robust-Reversible-Watermarking-in-Encrypted-Image-With-Secure-Multi-Party-Lightweight-Cryptography

# Create virtual environment (Windows)
python -m venv .venv & .\.venv\Scripts\activate
# (macOS/Linux: python -m venv .venv && source .venv/bin/activate)

# Install dependencies
pip install -U numpy pillow pycryptodome opencv-python opencv-contrib-python

🚀 Usage

Quick Test (Demo Mode) — auto-generates a gradient image:

python demo_rrw_pipeline.py ^
  --cover data/cover.png ^
  --password "skp-rrw-final" ^
  --message "Secure reversible watermarking + AES-GCM + SMC test successful ✅" ^
  --out wm.png ^
  --recover recovered.png

Full CLI (With SMC + Hash Verification)

# Embed
python main.py embed ^
  --cover data/cover.png ^
  --password "skp-rrw-final" ^
  --message "Top secret" ^
  --out wm.png

# Extract
python main.py extract ^
  --marked wm.png ^
  --meta wm.png.meta.json ^
  --password "skp-rrw-final" ^
  --recover recovered.png

Expected Output

Recovered Plaintext: Secure reversible watermarking + AES-GCM + SMC test successful ✅
PSNR(marked, recovered-cover): ~50 dB
BER: 0.0
✅ Perfect reversibility verified (bit-exact match)

📦 File Structure

Robust-Reversible-Watermarking/
├── crypto.py
├── smc.py
├── watermark_rrw.py
├── metrics.py
├── demo_rrw_pipeline.py
├── main.py
└── legacy_AES_LSB_version.py

## 📊 Example Output Images

| Watermarked Image | Recovered Image |
|--------------------|----------------|
| ![Watermarked](examples/watermarked.png) | ![Recovered](examples/recovered.png) |

<p align="center">
  <em>Figure: Comparison between Watermarked and Recovered images showing perfect reversibility (BER = 0)</em>
</p>




📈 Research Significance

This project demonstrates:

    A reversible watermarking method with zero information loss.

    Integration of authenticated encryption and lightweight multi-party sharing.

    Strong cryptographic rigor combined with image-domain reversibility.

    Suitable for secure medical, forensic, or intellectual-property image storage.


📜 License

This project is released under the MIT License — you are free to use, modify, and distribute with attribution.

✨ Citation

If you use or reference this work in academic research, please cite:

Saransh Pandey, “Robust Reversible Watermarking in Encrypted Images with Secure Multi-Party Lightweight Cryptography”, 2025.

🚀 Final Status:
✅ BER = 0.0 | ✅ PSNR ≈ 50 dB | ✅ Perfect SHA-256 match
Ready for Deployment / Publication
