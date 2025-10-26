# src/rrw.py
import numpy as np
import zlib
from PIL import Image

# --- Helpers ---
def to_gray(img: np.ndarray) -> np.ndarray:
    # img: HxWx3 uint8 -> return HxW uint8
    if img.ndim == 3:
        return (0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2]).astype(np.uint8)
    return img

def clip255(x): return np.uint8(np.clip(x, 0, 255))

# Simple predictor (4-neighborhood mean)
def predict(I):
    # pad edges with same values to avoid boundary issues
    Ip = np.pad(I, 1, mode='edge')
    # prediction: average of N,S,E,W
    pred = ((Ip[1:-1,2:] + Ip[1:-1,:-2] + Ip[2:,1:-1] + Ip[:-2,1:-1]) / 4.0).round()
    return pred.astype(np.int16)

# Convert bitstring <-> bytes
def bits_to_bytes(bits: np.ndarray) -> bytes:
    pad = (8 - (len(bits) % 8)) % 8
    if pad: bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    by = np.packbits(bits)
    return bytes(by)

def bytes_to_bits(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))

# --- Core PEE embedding ---
def pee_embed(cover_rgb: np.ndarray, payload_bits: np.ndarray):
    """
    cover_rgb: HxWx3 uint8
    payload_bits: 1D array of {0,1}
    Returns: watermarked_rgb, side_info (compressed loc map, params), used_bits
    """
    I = to_gray(cover_rgb)
    H, W = I.shape
    pred = predict(I)
    err = I.astype(np.int16) - pred  # prediction error

    # Choose expandable bins (e.g., error==0) for simplest demo
    # In real RRW you'd use a range (0 and -1) to balance distortion/capacity.
    loc = (err == 0)  # locations we can expand
    idx = np.argwhere(loc).reshape(-1)

    used = min(len(idx), len(payload_bits))
    if used == 0:
        raise ValueError("No capacity to embed any bits with chosen bins.")

    # We'll expand error 0 -> 0 or 1 to carry one bit per expandable pixel.
    Iw = I.copy().astype(np.int16)
    k = 0
    for (r, c) in zip(*np.where(loc)):
        if k >= used: break
        bit = int(payload_bits[k])
        # expansion: err=0, new pixel = pred + (0 + bit)
        Iw[r, c] = pred[r, c] + bit
        k += 1

    Iw = clip255(Iw)

    # Location map needs to be reproducible on extraction.
    # Here we store the positions we used and H,W as side info (compressed).
    used_map = np.zeros((H, W), dtype=np.uint8)
    used_map[loc] = 1
    used_map_bytes = used_map.tobytes()
    cmpr_map = zlib.compress(used_map_bytes, level=9)

    # Return a color image (merge gray channel back simply by replacing luminance)
    wm_rgb = cover_rgb.copy()
    # Put Iw into all channels equally (minimal demo). For production, use Y channel in YCbCr.
    wm_rgb[...,0] = Iw
    wm_rgb[...,1] = Iw
    wm_rgb[...,2] = Iw

    side_info = {
        "H": H, "W": W,
        "cmpr_map": cmpr_map
    }
    return wm_rgb, side_info, used

def pee_extract(marked_rgb: np.ndarray, side_info: dict):
    """
    Returns: recovered_bits (np.ndarray), recovered_cover_rgb (uint8)
    """
    Iw = to_gray(marked_rgb).astype(np.int16)
    H, W = side_info["H"], side_info["W"]
    used_map = np.frombuffer(zlib.decompress(side_info["cmpr_map"]), dtype=np.uint8).reshape(H, W).astype(bool)

    pred = predict(Iw.astype(np.uint8))
    # For places we used, err' = Iw - pred should be {0 or 1}; bit = err'
    errp = Iw - pred
    bits = np.zeros(int(used_map.sum()), dtype=np.uint8)

    k = 0
    Irec = Iw.copy()
    coords = np.argwhere(used_map)
    for r, c in coords:
        b = int(np.clip(errp[r, c], 0, 1))
        bits[k] = b
        # Reversible step: restore original pixel: pred + 0 (since original err was 0)
        Irec[r, c] = pred[r, c]
        k += 1

    Irec = clip255(Irec)
    rec_rgb = marked_rgb.copy()
    rec_rgb[...,0] = Irec
    rec_rgb[...,1] = Irec
    rec_rgb[...,2] = Irec
    return bits, rec_rgb
