# demo_rrw_pipeline.py
# Minimal end-to-end demo: AES-GCM (aead) + PEE-based reversible watermarking.
# Usage:
#   python demo_rrw_pipeline.py --cover data/cover.png --password "my pass" \
#       --message "Top secret" --out watermarked.png --recover cover_recovered.png

import argparse, json, base64
import numpy as np
import cv2

from crypto import aead_encrypt, aead_decrypt
from watermark_rrw import pee_embed, pee_extract, bytes_to_bits, bits_to_bytes

# --- small helpers ---
def load_image_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_image_rgb(path: str, rgb: np.ndarray) -> None:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)

def b64(x: bytes) -> str:
    return base64.b64encode(x).decode("utf-8")

def ub64(s: str) -> bytes:
    return base64.b64decode(s.encode("utf-8"))

def psnr(x: np.ndarray, y: np.ndarray) -> float:
    # quick PSNR (dB)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    mse = np.mean((x - y) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((255.0 ** 2) / mse)

def main():
    ap = argparse.ArgumentParser(description="RRW + AES-GCM demo pipeline")
    ap.add_argument("--cover", required=True, help="path to cover image (png/jpg)")
    ap.add_argument("--password", required=True, help="passphrase for AEAD")
    ap.add_argument("--message", required=True, help="plaintext to embed")
    ap.add_argument("--out", default="watermarked.png", help="output watermarked image")
    ap.add_argument("--meta", default=None, help="side-info json path (optional)")
    ap.add_argument("--recover", default="cover_recovered.png", help="output recovered cover image")
    args = ap.parse_args()

    # 1) Load cover
    cover = load_image_rgb(args.cover)
    print(f"[+] Loaded cover: {args.cover}  shape={cover.shape}")

    # 2) Encrypt plaintext with AES-GCM
    salt, nonce, ct, tag = aead_encrypt(args.message.encode("utf-8"), args.password)

    # Build a compact ciphertext blob (JSON, base64 fields)
    ct_blob = json.dumps({
        "salt":  b64(salt),
        "nonce": b64(nonce),
        "tag":   b64(tag),
        "ct":    b64(ct)
    }).encode("utf-8")

    # 3) Convert blob to bits and embed via PEE
    bits = bytes_to_bits(ct_blob)
    wm_rgb, side_info, used = pee_embed(cover, bits)
    save_image_rgb(args.out, wm_rgb)

    # Save side-info (ordered coords + dims) so we can extract later
    meta_path = args.meta or (args.out + ".meta.json")
    meta = {
        "side_info": {
            "H": side_info["H"],
            "W": side_info["W"],
            "coords": b64(side_info["coords"]),  # <-- updated
            "used":   used                       # <-- store used explicitly (redundant but handy)
        },
        "used_bits": used,
        "cover_shape": list(cover.shape),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"[+] Embedded {used} bits → wrote {args.out} and {meta_path}")

    # 4) Extract back from the watermarked image using saved side-info
    wm_rgb2 = load_image_rgb(args.out)
    with open(meta_path, "r") as f:
        meta2 = json.load(f)

    side_info2 = {
        "H": meta2["side_info"]["H"],
        "W": meta2["side_info"]["W"],
        "coords": ub64(meta2["side_info"]["coords"]),  # <-- updated
        "used":   meta2["side_info"].get("used", meta2["used_bits"])
    }
    used2 = meta2["used_bits"]

    ext_bits, rec_cover = pee_extract(wm_rgb2, side_info2)
    ext_bits = ext_bits[:used2]  # truncate exactly to what we embedded
    blob_bytes = bits_to_bytes(ext_bits)

    # 5) Decrypt with original password
    obj = json.loads(blob_bytes.decode("utf-8"))
    salt2  = ub64(obj["salt"])
    nonce2 = ub64(obj["nonce"])
    tag2   = ub64(obj["tag"])
    ct2    = ub64(obj["ct"])

    pt = aead_decrypt(salt2, nonce2, ct2, tag2, args.password)
    plaintext = pt.decode("utf-8")

    # 6) Save recovered cover & print a quick PSNR check
    save_image_rgb(args.recover, rec_cover)
    ps = psnr(wm_rgb2, rec_cover)
    print(f"[+] Recovered plaintext: {plaintext}")
    print(f"[+] PSNR(marked, recovered-cover): {ps:.2f} dB")
    print(f"[+] Wrote recovered cover to: {args.recover}")

if __name__ == "__main__":
    main()

