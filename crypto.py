# src/crypto.py
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import scrypt
from Crypto.Random import get_random_bytes

# KDF: derive a 256-bit key from a password (salted & hardened)
def derive_key(password: str, salt: bytes | None = None):
    if salt is None:
        salt = get_random_bytes(16)
    key = scrypt(password.encode('utf-8'), salt, key_len=32, N=2**15, r=8, p=1)
    return key, salt

# AEAD encrypt (AES-GCM) -> returns (salt, nonce, ct, tag)
def aead_encrypt(plaintext: bytes, password: str):
    key, salt = derive_key(password)
    nonce = get_random_bytes(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ct, tag = cipher.encrypt_and_digest(plaintext)
    return salt, nonce, ct, tag

# AEAD decrypt (AES-GCM)
def aead_decrypt(salt: bytes, nonce: bytes, ct: bytes, tag: bytes, password: str) -> bytes:
    key, _ = derive_key(password, salt)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    return cipher.decrypt_and_verify(ct, tag)
