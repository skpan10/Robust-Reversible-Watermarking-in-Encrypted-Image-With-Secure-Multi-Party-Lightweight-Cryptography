import numpy as np, zlib

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return (0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2]).astype(np.uint8)
    return img

def _clip255(x): return np.uint8(np.clip(x, 0, 255))

def _predict_causal(I_uint8: np.ndarray) -> np.ndarray:
    Ip = np.pad(I_uint8, 1, mode='edge')
    north = Ip[:-2, 1:-1]
    west  = Ip[1:-1, :-2]
    pred = ((north + west) / 2.0).round()
    return pred.astype(np.int16)

def pee_embed(cover_rgb: np.ndarray, payload_bits: np.ndarray):
    I = _to_gray(cover_rgb)
    H, W = I.shape

    pred = _predict_causal(I)              # predictor on the original cover
    err  = I.astype(np.int16) - pred

    # Expand only where err==0 AND pred<=254 (avoid +1 overflow)
    loc = (err == 0) & (pred <= 254)
    coords = np.argwhere(loc).astype(np.uint16)  # row-major (N,2)

    used = min(len(coords), len(payload_bits))
    if used == 0:
        raise ValueError("No capacity to embed any bits with chosen bins (safe).")

    Iw = I.copy().astype(np.int16)
    for k in range(used):
        r, c = map(int, coords[k])
        bit = int(payload_bits[k])         # 0 or 1
        Iw[r, c] = pred[r, c] + bit

    Iw = _clip255(Iw)

    # store coords in C-order
    cmpr_coords = zlib.compress(coords[:used].tobytes(order="C"), level=9)
    side_info = {"H": H, "W": W, "coords": cmpr_coords, "used": used}

    wm_rgb = cover_rgb.copy()
    wm_rgb[...,0] = Iw; wm_rgb[...,1] = Iw; wm_rgb[...,2] = Iw
    return wm_rgb, side_info, used

def pee_extract(marked_rgb: np.ndarray, side_info: dict):
    Iw = _to_gray(marked_rgb).astype(np.int16)

    coords = np.frombuffer(zlib.decompress(side_info["coords"]),
                           dtype=np.uint16).reshape(-1, 2, order="C")
    used = int(side_info.get("used", len(coords)))

    Irec = Iw.copy()
    bits = np.zeros(used, dtype=np.uint8)

    # Raster-scan; predictor uses only north & west from Irec (already restored)
    for k in range(used):
        r, c = map(int, coords[k])
        r_up = 0 if r == 0 else r-1
        c_lt = 0 if c == 0 else c-1
        p = int(round((Irec[r_up, c] + Irec[r, c_lt]) / 2.0))
        errp = int(Irec[r, c] - p)
        bits[k] = 1 if errp >= 1 else 0
        Irec[r, c] = p

    Irec = _clip255(Irec)
    rec_rgb = marked_rgb.copy()
    rec_rgb[...,0] = Irec; rec_rgb[...,1] = Irec; rec_rgb[...,2] = Irec
    return bits, rec_rgb



