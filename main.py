# main.py — Final Updated Version (RRW + SMC + AES-GCM + SHA256)
import argparse, json, base64, os, hashlib
import numpy as np
import cv2
from PIL import Image

from crypto import aead_encrypt, aead_decrypt, derive_key
from smc import split_key_xor, combine_key_xor
from watermark_rrw import pee_embed, pee_extract, bytes_to_bits, bits_to_bytes
from metrics import psnr, ber

# ======================================================
# === Helpers ==========================================
# ======================================================

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_image(path, rgb):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)

def b64(x: bytes) -> str:
    return base64.b64encode(x).decode('utf-8')

def ub64(s: str) -> bytes:
    return base64.b64decode(s.encode('utf-8'))

def sha256_digest(arr: np.ndarray) -> str:
    """Return SHA-256 hash of raw image bytes (for reversibility verification)."""
    return hashlib.sha256(arr.tobytes()).hexdigest()

# ======================================================
# === EMBED MODE =======================================
# ======================================================

def cmd_embed(args):
    cover = load_image(args.cover)
    cover_hash = sha256_digest(cover)

    # --- Secure Multi-Party Demo ---
    passphrase = args.password
    key, _ = derive_key(passphrase)
    k1, k2 = split_key_xor(key)
    _ = combine_key_xor(k1, k2)  # kept for demo symmetry

    # --- Encrypt message ---
    salt2, nonce, ct, tag = aead_encrypt(args.message.encode('utf-8'), passphrase)
    ct_blob = json.dumps({
        "salt":  b64(salt2),
        "nonce": b64(nonce),
        "tag":   b64(tag),
        "ct":    b64(ct)
    }).encode('utf-8')

    # --- Embed into image ---
    bits = bytes_to_bits(ct_blob)
    wm_rgb, side_info, used = pee_embed(cover, bits)
    save_image(args.out, wm_rgb)

    # --- Save metadata (coords + pred_vals + dims) ---
    meta = {
        "side_info": {
            "H": side_info["H"],
            "W": side_info["W"],
            "coords":    b64(side_info["coords"]),
            "pred_vals": b64(side_info["pred_vals"]),
            "used":      side_info.get("used", used),
        },
        "used_bits": used,
        "cover_shape": list(cover.shape),
        "cover_sha256": cover_hash
    }

    with open(args.out + ".meta.json", "w") as f:
        json.dump(meta, f)

    print(f"[embed] used bits: {used}  | cover: {cover.shape}")
    print(f"[embed] wrote: {args.out} & {args.out}.meta.json")
    print(f"[embed] cover hash: {cover_hash}")

# ======================================================
# === EXTRACT MODE =====================================
# ======================================================

def cmd_extract(args):
    wm_rgb = load_image(args.marked)
    with open(args.meta, "r") as f:
        meta = json.load(f)

    # Load coords + pred_vals + used
    side_info = {
        "H": meta["side_info"]["H"],
        "W": meta["side_info"]["W"],
        "coords":    ub64(meta["side_info"]["coords"]),
        "pred_vals": ub64(meta["side_info"]["pred_vals"]),
        "used":      meta["side_info"].get("used", meta["used_bits"]),
    }
    used = meta["used_bits"]

    bits, rec_rgb = pee_extract(wm_rgb, side_info)
    bits = bits[:used]
    blob = bits_to_bytes(bits)

    # --- Decrypt payload ---
    obj = json.loads(blob.decode('utf-8'))
    salt = ub64(obj["salt"])
    nonce = ub64(obj["nonce"])
    tag = ub64(obj["tag"])
    ct = ub64(obj["ct"])
    pt = aead_decrypt(salt, nonce, ct, tag, args.password)

    # --- Save and verify ---
    out_cover = args.recover if args.recover else "cover_recovered.png"
    save_image(out_cover, rec_rgb)

    rec_hash = sha256_digest(rec_rgb)
    original_hash = meta.get("cover_sha256", None)
    ps = psnr(wm_rgb, rec_rgb)

    print(f"[extract] message: {pt.decode('utf-8')}")
    print(f"[extract] PSNR(marked vs recovered): {ps:.2f} dB")

    if original_hash:
        print(f"[verify] cover hash (embed): {original_hash}")
        print(f"[verify] recovered hash (now): {rec_hash}")
        if rec_hash == original_hash:
            print("✅ Perfect reversibility verified (bit-exact match).")
        else:
            print("⚠️ Warning: hash mismatch (lossy step or rounding occurred).")

    print(f"[extract] wrote recovered cover: {out_cover}")

# ======================================================
# === ENTRY POINT ======================================
# ======================================================

def main():
    ap = argparse.ArgumentParser(description="RRW + SMC (with AES-GCM & SHA256 verification)")
    sub = ap.add_subparsers(dest="cmd")

    ap_e = sub.add_parser("embed")
    ap_e.add_argument("--cover", required=True, help="cover image (png/jpg)")
    ap_e.add_argument("--password", required=True, help="passphrase")
    ap_e.add_argument("--message", required=True, help="plaintext to embed")
    ap_e.add_argument("--out", default="watermarked.png", help="output watermarked")
    ap_e.set_defaults(func=cmd_embed)

    ap_x = sub.add_parser("extract")
    ap_x.add_argument("--marked", required=True, help="watermarked image")
    ap_x.add_argument("--meta", required=True, help="metadata JSON from embed phase")
    ap_x.add_argument("--password", required=True, help="passphrase")
    ap_x.add_argument("--recover", help="output recovered cover image")
    ap_x.set_defaults(func=cmd_extract)

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help()
        return
    args.func(args)

if __name__ == "__main__":
    main()
