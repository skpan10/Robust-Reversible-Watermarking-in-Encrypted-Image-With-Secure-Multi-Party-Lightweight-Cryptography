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

    pred = _predict_causal(I)                  # causal predictor on original cover
    err  = I.astype(np.int16) - pred

    # Expand only where err==0 AND pred<=254 (avoid +1 overflow)
    loc = (err == 0) & (pred <= 254)
    coords = np.argwhere(loc).astype(np.uint16)   # (N,2), row-major

    used = min(len(coords), len(payload_bits))
    if used == 0:
        raise ValueError("No capacity to embed any bits with chosen bins (safe).")

    coords_used = coords[:used]
    Iw = I.copy().astype(np.int16)

    # Embed using the *same* predictions
    for k in range(used):
        r, c = map(int, coords_used[k])
        bit = int(payload_bits[k])
        Iw[r, c] = pred[r, c] + bit

    Iw = _clip255(Iw)

    # Store coords and the predictor values at those coords (uint8), both compressed
    cmpr_coords = zlib.compress(coords_used.tobytes(order="C"), level=9)
    pred_vals   = pred[coords_used[:,0], coords_used[:,1]].astype(np.uint8)
    cmpr_pred   = zlib.compress(pred_vals.tobytes(), level=9)

    side_info = {"H": H, "W": W, "coords": cmpr_coords, "pred_vals": cmpr_pred, "used": used}

    wm_rgb = cover_rgb.copy()
    wm_rgb[...,0] = Iw; wm_rgb[...,1] = Iw; wm_rgb[...,2] = Iw
    return wm_rgb, side_info, used

def pee_extract(marked_rgb: np.ndarray, side_info: dict):
    Iw = _to_gray(marked_rgb).astype(np.int16)

    coords = np.frombuffer(zlib.decompress(side_info["coords"]),
                           dtype=np.uint16).reshape(-1, 2, order="C")
    used = int(side_info.get("used", len(coords)))
    coords = coords[:used]

    pred_vals = np.frombuffer(zlib.decompress(side_info["pred_vals"]), dtype=np.uint8)[:used]

    Irec = Iw.copy()
    bits = np.zeros(used, dtype=np.uint8)

    # Use the EXACT predictor values from embed-time (no drift)
    for k in range(used):
        r, c = map(int, coords[k])
        p = int(pred_vals[k])
        errp = int(Irec[r, c] - p)
        bits[k] = 1 if errp >= 1 else 0
        Irec[r, c] = p  # restore original pixel

    Irec = _clip255(Irec)
    rec_rgb = marked_rgb.copy()
    rec_rgb[...,0] = Irec; rec_rgb[...,1] = Irec; rec_rgb[...,2] = Irec
    return bits, rec_rgb



