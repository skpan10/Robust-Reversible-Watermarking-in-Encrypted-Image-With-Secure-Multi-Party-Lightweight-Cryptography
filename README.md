# 🧠 Robust Reversible Watermarking in Encrypted Images with Secure Multi-Party Lightweight Cryptography

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)]()
[![AES-GCM](https://img.shields.io/badge/Crypto-AES--GCM-green.svg)]()
[![RRW](https://img.shields.io/badge/Algorithm-Reversible%20Watermarking-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

> **Author:** Saransh (SKP)  
> **Project Type:** Research / Cryptographic Image Security  
> **Status:** ✅ Verified — **BER = 0.0**  |  **PSNR ≈ 50 dB**  |  **Perfect SHA-256 Reversibility**

---

## 🧩 Abstract

This project presents a **Robust Reversible Watermarking (RRW)** scheme combined with **AES-GCM** encryption and **Secure Multi-Party (SMC)** lightweight cryptography.  
Goal: securely embed encrypted payloads inside images **without perceptual degradation** and with **bit-exact reversibility**.

**Highlights**
- 🔒 **Confidentiality:** AES-GCM authenticated encryption.
- ♻️ **Reversibility:** Exact cover recovery (BER = 0).
- 🤝 **Multi-Party:** XOR key split/merge (SMC).
- 🧠 **Causal RRW:** Overflow-safe, drift-free embedding (causal predictor).
- 🧾 **Integrity:** SHA-256 verification of recovered cover.

---

## 🧱 System Architecture

[Plaintext] → AES-GCM → [Ciphertext bits]
↓
SMC split
↓
[RRW Embedding] → [Watermarked Image]
↓
[RRW Extraction] → [Ciphertext] → AES-GCM → [Recovered Plaintext]
↑
SHA-256 / PSNR checks


---

## ⚙️ Features

| Module | Functionality |
|---|---|
| `crypto.py` | AES-GCM (PyCryptodome) + Scrypt key derivation |
| `smc.py` | XOR-based key split / combine (SMC) |
| `watermark_rrw.py` | Causal, overflow-safe RRW with stored predictor context (0-BER extraction) |
| `metrics.py` | PSNR / BER / SSIM utilities |
| `main.py` | CLI (AES-GCM + SMC + SHA-256 verification) |
| `demo_rrw_pipeline.py` | Minimal end-to-end demo runner |

---

## 🧪 Verification Results

| Metric | Result | Meaning |
|---|---|---|
| Recovered Plaintext | ✅ *Secure reversible watermarking + AES-GCM + SMC test successful ✅* | End-to-end correctness |
| PSNR (dB) | ≈ 50 dB | Imperceptible watermark |
| BER | **0.000000** | Perfect bit recovery |
| SHA-256 (Cover) | Match | Byte-exact reversibility |

---

## 🧰 Installation

```bash
# Clone
git clone https://github.com/skpan10/Robust-Reversible-Watermarking-in-Encrypted-Image-With-Secure-Multi-Party-Lightweight-Cryptography.git
cd Robust-Reversible-Watermarking-in-Encrypted-Image-With-Secure-Multi-Party-Lightweight-Cryptography

# Create venv
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# Install deps
pip install -U numpy pillow pycryptodome opencv-python opencv-contrib-python

🚀 Usage

Quick Demo (auto-generates a gradient cover)

python demo_rrw_pipeline.py ^
  --cover data/cover.png ^
  --password "skp-rrw-final" ^
  --message "Secure reversible watermarking + AES-GCM + SMC test successful ✅" ^
  --out watermarked.png ^
  --recover recovered.png

Full CLI (SMC + SHA-256 verification)

# Embed
python main.py embed ^
  --cover data/cover.png ^
  --password "skp-rrw-final" ^
  --message "Top secret" ^
  --out watermarked.png

# Extract
python main.py extract ^
  --marked watermarked.png ^
  --meta watermarked.png.meta.json ^
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
├── legacy_AES_LSB_version.py
└── examples/
    ├── watermarked.png
    └── recovered.png

## 📊 Example Output Images

| Watermarked Image | Recovered Image |
|---|---|
| ![Watermarked](https://raw.githubusercontent.com/skpan10/Robust-Reversible-Watermarking-in-Encrypted-Image-With-Secure-Multi-Party-Lightweight-Cryptography/main/examples/watermarked.png) | ![Recovered](https://raw.githubusercontent.com/skpan10/Robust-Reversible-Watermarking-in-Encrypted-Image-With-Secure-Multi-Party-Lightweight-Cryptography/main/examples/recovered.png) |

**Direct raw links (for download / copy):**

Watermarked: https://raw.githubusercontent.com/skpan10/Robust-Reversible-Watermarking-in-Encrypted-Image-With-Secure-Multi-Party-Lightweight-Cryptography/main/examples/watermarked.png

Recovered : https://raw.githubusercontent.com/skpan10/Robust-Reversible-Watermarking-in-Encrypted-Image-With-Secure-Multi-Party-Lightweight-Cryptography/main/examples/recovered.png

<p align="center">
  <em>Figure: Watermarked vs Recovered (BER = 0, bit-exact reversibility).</em>
</p>


📈 Research Significance

Reversible watermarking with zero information loss

Authenticated encryption + lightweight multi-party sharing

Strong cryptographic rigor with image-domain reversibility

Applicable to medical, forensics, and IP-sensitive archives

📜 License

Released under MIT License — free to use, modify, and distribute with attribution.

✨ Citation

Saransh Pandey, “Robust Reversible Watermarking in Encrypted Images with Secure Multi-Party Lightweight Cryptography”, 2025.
