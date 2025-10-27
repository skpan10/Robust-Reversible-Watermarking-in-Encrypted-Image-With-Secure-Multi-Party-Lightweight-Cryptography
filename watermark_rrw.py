import numpy as np
import zlib
from PIL import Image

# ======================================================
# === Helpers ==========================================
# ======================================================

def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert RGB → grayscale uint8"""
    if img.ndim == 3:
        return (0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2]).astype(np.uint8)
    return img

def clip255(x): 
    return np.uint8(np.clip(x, 0, 255))

# 4-neighbourhood mean predictor
def predict(I):
    Ip = np.pad(I, 1, mode='edge')
    pred = ((Ip[1:-1,2:] + Ip[1:-1,:-2] + Ip[2:,1:-1] + Ip[:-2,1:-1]) / 4.0).round()
    return pred.astype(np.int16)

# Bit conversions
def bits_to_bytes(bits: np.ndarray) -> bytes:
    pad = (8 - (len(bits) % 8)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits).tobytes()

def bytes_to_bits(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))

# ======================================================
# === Core RRW Embed ===================================
# ======================================================

def pee_embed(cover_rgb: np.ndarray, payload_bits: np.ndarray):
    """
    Robust reversible watermark embedding.
    cover_rgb : HxWx3 uint8
    payload_bits : 1D array of {0,1}
    returns: (watermarked_rgb, side_info_dict, used_bits)
    """
    I = to_gray(cover_rgb)
    H, W = I.shape
    pred = predict(I)
    err  = I.astype(np.int16) - pred

    # Select expandable bins (simple version: err == 0)
    loc = (err == 0)
    coords_all = list(zip(*np.where(loc)))          # deterministic row-major order
    used = min(len(coords_all), len(payload_bits))
    if used == 0:
        raise ValueError("No capacity to embed any bits with chosen bins.")

    coords_used = coords_all[:used]
    Iw = I.copy().astype(np.int16)

    for k, (r, c) in enumerate(coords_used):
        bit = int(payload_bits[k])
        Iw[r, c] = pred[r, c] + bit

    Iw = clip255(Iw)

    # Store only the used coordinates (ordered)
    coords_arr = np.array(coords_used, dtype=np.uint16)
    cmpr_coords = zlib.compress(coords_arr.tobytes(), level=9)
    side_info = {"H": H, "W": W, "coords": cmpr_coords, "used": used}

    # Merge into RGB for output
    wm_rgb = cover_rgb.copy()
    wm_rgb[..., 0] = Iw
    wm_rgb[..., 1] = Iw
    wm_rgb[..., 2] = Iw

    return wm_rgb, side_info, used

# ======================================================
# === Core RRW Extract =================================
# ======================================================

def pee_extract(marked_rgb: np.ndarray, side_info: dict):
    """
    Extract bits using the exact coordinate order saved at embed-time
    and progressively restore pixels so the local predictor sees the
    same context the embedder used.
    """
    Iw = to_gray(marked_rgb).astype(np.int16)
    H, W = int(side_info["H"]), int(side_info["W"])

    # Decompress ordered coordinate list
    coords_arr = np.frombuffer(
        zlib.decompress(side_info["coords"]), dtype=np.uint16
    ).reshape(-1, 2)
    used = int(side_info.get("used", len(coords_arr)))
    coords_used = [(int(r), int(c)) for r, c in coords_arr[:used]]

    # Progressive restoration with same predictor logic
    def pred_local(I, r, c):
        r_up = max(r - 1, 0)
        r_dn = min(r + 1, I.shape[0] - 1)
        c_lt = max(c - 1, 0)
        c_rt = min(c + 1, I.shape[1] - 1)
        return int(round((I[r_up, c] + I[r_dn, c] + I[r, c_rt] + I[r, c_lt]) / 4.0))

    Irec = Iw.copy()
    bits = np.zeros(used, dtype=np.uint8)

    for k, (r, c) in enumerate(coords_used):
        p = pred_local(Irec, r, c)
        errp = int(Irec[r, c] - p)
        bits[k] = 1 if errp >= 1 else 0
        Irec[r, c] = p  # progressively restore

    Irec = clip255(Irec)
    rec_rgb = marked_rgb.copy()
    rec_rgb[..., 0] = Irec
    rec_rgb[..., 1] = Irec
    rec_rgb[..., 2] = Irec

    return bits, rec_rgb


