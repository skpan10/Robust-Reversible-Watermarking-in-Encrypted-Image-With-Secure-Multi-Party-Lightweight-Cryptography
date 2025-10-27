# src/smc.py
import os

def split_key_xor(key_bytes: bytes):
    k1 = os.urandom(len(key_bytes))
    k2 = bytes([a ^ b for a, b in zip(key_bytes, k1)])
    return k1, k2

def combine_key_xor(k1: bytes, k2: bytes) -> bytes:
    return bytes([a ^ b for a, b in zip(k1, k2)])
