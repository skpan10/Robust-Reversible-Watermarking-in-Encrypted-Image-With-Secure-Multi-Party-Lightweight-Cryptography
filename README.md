# 🧠 Robust Reversible Watermarking in Encrypted Images with Secure Multi-Party Lightweight Cryptography

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)]()
[![AES-GCM](https://img.shields.io/badge/Crypto-AES--GCM-green.svg)]()
[![RRW](https://img.shields.io/badge/Algorithm-Reversible%20Watermarking-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

> **Author:** Saransh Kumar Pandey
> **Project Type:** Research / Cryptographic Image Security  
> **Status:** ✅ Verified — BER = 0  |  PSNR ≈ 50 dB  |  Perfect SHA-256 Reversibility  

---

## 🧩 Abstract

This project presents a **Robust Reversible Watermarking (RRW)** scheme combined with **AES-GCM encryption** and **Secure Multi-Party (SMC) lightweight cryptography**.  
It securely embeds encrypted payloads inside images **without perceptual degradation** and with **bit-exact reversibility**.

Highlights:
- 🔒 **Confidentiality:** AES-GCM authenticated encryption of hidden data.  
- ⚙️ **Reversibility:** Exact recovery of original cover (BER = 0).  
- 🤝 **Multi-Party Security:** XOR-based key-share SMC.  
- 🧠 **Causal Prediction:** Overflow-safe, drift-free pixel embedding.  
- 📊 **Integrity Validation:** SHA-256 verification of recovered cover.

---

## 🧱 System Architecture

[Plaintext]
│
├──► AES-GCM Encryption ──► Ciphertext Bits
│
├──► RRW Embedding ──► [Watermarked Image]
│
└──► SMC Key Split (Multi-Party XOR)
│
▼
[Watermarked Image] ──► RRW Extraction ──► Ciphertext ──► AES-GCM Decryption ──► [Recovered Plaintext]
▲
│
SHA-256 & PSNR Verification


---

## ⚙️ Features

| Module | Functionality |
|---------|---------------|
| **`crypto.py`** | AES-GCM encryption/decryption + Scrypt key derivation |
| **`smc.py`** | XOR-based Secure Multi-Party key split / combine |
| **`watermark_rrw.py`** | Causal overflow-safe RRW with stored predictor context (`pred_vals`) |
| **`metrics.py`** | PSNR / BER / SSIM computation |
| **`main.py`** | CLI with AES-GCM + SMC + SHA-256 verification |
| **`demo_rrw_pipeline.py`** | Self-contained runnable demo pipeline |

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
# Clone
git clone https://github.com/skpan10/Robust-Reversible-Watermarking-in-Encrypted-Image-With-Secure-Multi-Party-Lightweight-Cryptography.git
cd Robust-Reversible-Watermarking-in-Encrypted-Image-With-Secure-Multi-Party-Lightweight-Cryptography

# Create environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```
# Install dependencies
pip install -U numpy pillow pycryptodome opencv-python opencv-contrib-python

## ⚙️ Usage Summary

| Mode | Command | Description |
|------|---------|-------------|
| **Quick Test (Demo Mode)** | `python demo_rrw_pipeline.py --cover data/cover.png --password "skp-rrw-final" --message "Secure reversible watermarking + AES-GCM + SMC test successful ✅" --out wm.png --recover recovered.png` | **Auto-generates a gradient image and runs the full pipeline** (AES-GCM encrypt → RRW embed → extract → decrypt) to produce `wm.png` and `recovered.png`. |
| **Full CLI Mode** | `python main.py embed --cover data/cover.png --password "skp-rrw-final" --message "Top secret" --out wm.png` | Embeds encrypted payload and writes **`wm.png`** plus **`wm.png.meta.json`**. |
| **Extraction Mode** | `python main.py extract --marked wm.png --meta wm.png.meta.json --password "skp-rrw-final" --recover recovered.png` | Extracts bits, decrypts plaintext, restores **`recovered.png`**, and verifies SHA-256. |

---

## 💾 Embed & Extraction Process

| Step | File Input | File Output | Description |
|------|------------|-------------|-------------|
| 1️⃣ **Embed Phase** | `cover.png` + plaintext + password | `watermarked.png`, `watermarked.png.meta.json` | AES-GCM encrypts plaintext; RRW embeds ciphertext bits; metadata saved for extraction. |
| 2️⃣ **Extract Phase** | `watermarked.png`, `watermarked.png.meta.json`, password | `recovered.png` + plaintext | RRW bit-exact extraction → AES-GCM decrypt → SHA-256 cover verification. |

---

## ✅ Expected Output

| Metric | Result | Description |
|--------|--------|-------------|
| **Recovered Plaintext** | `Secure reversible watermarking + AES-GCM + SMC test successful ✅` | Confirms end-to-end correctness. |
| **PSNR (dB)** | `≈ 50.1 dB` | Watermark is visually imperceptible. |
| **BER** | `0.0` | Perfect bit recovery. |
| **SHA-256 Match** | ✅ | Recovered cover is bit-exact to original. |
| **Final Status** | 🧠 *Project Verified* | BER = 0.0 • PSNR ≈ 50 dB • SHA-256 = Match. |


## 📦 File Structure

| Folder / File | Description |
|----------------|-------------|
| **`crypto.py`** | AES-GCM encryption & decryption with key derivation |
| **`smc.py`** | Secure Multi-Party XOR key split and merge |
| **`watermark_rrw.py`** | Core RRW algorithm (causal prediction + overflow-safe embedding) |
| **`metrics.py`** | PSNR, BER, and SSIM computation utilities |
| **`demo_rrw_pipeline.py`** | Stand-alone demo script for quick testing |
| **`main.py`** | CLI interface integrating AES-GCM + SMC + SHA-256 verification |
| **`legacy_AES_LSB_version.py`** | Archived hybrid AES+LSB version for reference |
| **`examples/watermarked.png`** | Output: Watermarked image |

## 📊 Example Output Images

<p align="center">
  <img src="examples/watermarked.png" alt="Watermarked Image" width="45%"/>
  <img src="examples/recovered.png" alt="Recovered Image" width="45%"/>
</p>

<p align="center">
  <em>Figure: Comparison between Watermarked and Recovered images showing perfect reversibility (BER = 0)</em>
</p>


📈 Research Significance

This project demonstrates:

    A reversible watermarking method with zero information loss.

    Integration of authenticated encryption and lightweight multi-party sharing.

    Strong cryptographic rigor with image-domain reversibility.

    Applicable to secure medical, forensic, and IP-safe image storage.

📜 License

Released under the MIT License — free to use, modify, and distribute with attribution.
✨ Citation

If you reference this work in academic research, please cite:

    Saransh Kumar Pandey,
    “Robust Reversible Watermarking in Encrypted Images with Secure Multi-Party Lightweight Cryptography”, 2025.

🚀 Final Verification

✅ BER = 0.000000
✅ PSNR ≈ 50 dB
✅ SHA-256 = Perfect Match
✅ Status — Ready for Deployment / Publication


