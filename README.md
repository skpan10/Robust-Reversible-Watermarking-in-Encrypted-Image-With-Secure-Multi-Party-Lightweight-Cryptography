# 🔐 Robust Reversible Watermarking in Encrypted Images with Secure Multi-Party Lightweight Cryptography

A hybrid cryptographic framework that combines **Reversible Watermarking (RRW)** and **Secure Multi-Party Computation (SMC)** for **confidential, verifiable, and lossless image security**.  
This project ensures high fidelity (PSNR > 50 dB), robustness, and full reversibility of the original image post-decryption — making it ideal for secure media transmission, forensics, and blockchain-based digital ownership systems.

---

## 🧠 Project Overview
This implementation introduces an advanced **AES-GCM** encryption system fused with **Pixel Error Expansion (PEE)–based Reversible Watermarking**, enhanced by a **Secure Multi-Party XOR key-splitting model**.

### Core Features
- **AES-GCM Symmetric Encryption:** Authenticated, lossless, and tamper-detecting cryptography.  
- **RRW Algorithm:** Pixel-level reversible watermarking preserving perfect data recoverability.  
- **SMC-Inspired Key Sharing:** XOR-based multi-party key distribution for enhanced confidentiality.  
- **Integrity Verification:** SHA-256 hash validation to ensure no distortion during recovery.  
- **Quality Metrics:** Automatic PSNR, SSIM, and BER evaluation for benchmarking.  
- **Modular Design:** Fully Python-based, open-source, and easy to extend for research.

---

## 📁 Repository Structure
| File | Description |
|------|--------------|
| **main.py** | Entry point for RRW + SMC encryption and decryption pipeline. |
| **crypto.py** | AES-GCM encryption with Scrypt key derivation (KDF). |
| **smc.py** | Lightweight XOR key splitting & recombination for secure multi-party use. |
| **watermark_rrw.py** | Core reversible watermark embedding & extraction algorithm. |
| **metrics.py** | Evaluation metrics (PSNR, SSIM, BER). |
| **demo_rrw_pipeline.py** | Example standalone demo pipeline for testing. |
| **legacy_AES_LSB_version.py** | Older hybrid AES + LSB implementation for backward comparison. |

---

## ⚙️ Setup Instructions

### Prerequisites
- Python **3.8+**
- Install dependencies:
  ```bash
  pip install numpy opencv-python pillow pycryptodome

🚀 How to Run
1️⃣ Embed Watermark

Embed a secure encrypted watermark into an image.

python main.py embed --cover "input.png" --password "yourpass" \
  --message "Confidential Data" --out "watermarked.png"

This will generate:

    watermarked.png – Encrypted & watermark-embedded image.

    watermarked.png.meta.json – Metadata (location map, side-info).

2️⃣ Extract Watermark & Recover Original

python main.py extract --marked "watermarked.png" \
  --meta "watermarked.png.meta.json" \
  --password "yourpass" --recover "recovered.png"

Outputs:

    Recovered Message: Original plaintext extracted from encrypted watermark.

    Recovered Image: Pixel-perfect restoration validated via SHA-256.

    Quality Report: PSNR / SSIM values logged for verification.

📊 Example Output

[embed] used bits: 512000 | cover: (512,512,3)
[extract] recovered message: Confidential Data
[extract] PSNR(marked, recovered-cover): 58.74 dB
[verify] Perfect reversibility verified (bit-exact match).

🧩 Algorithm Flow

    AES-GCM Encryption: Encrypt plaintext → Ciphertext + Tag.

    SMC Key-Split: Split encryption key into XOR shares.

    Reversible Watermarking (PEE): Embed ciphertext bits → Image.

    Decryption & Extraction: Recombine keys → Recover exact plaintext.

    Integrity Validation: SHA-256 hash confirms bit-level restoration.

🧪 Performance Metrics
Metric	Description	Typical Result
PSNR	Peak Signal-to-Noise Ratio	55–60 dB
SSIM	Structural Similarity Index	≥ 0.99
BER	Bit Error Rate	≈ 0.0
🧰 Applications

    Digital watermarking in confidential systems.

    Secure reversible medical image transmission.

    Copyright protection with zero data loss.

    Blockchain-ready proof of image authenticity.

🧑‍💻 Author

Saransh Pandey (skpan10)
🔗 GitHub Profile

📧 Contact via LinkedIn
📝 License

This project is released under the MIT License — freely available for research and academic development.
