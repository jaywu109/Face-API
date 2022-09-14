import json
from base64 import b64decode, b64encode
from os import path

from Crypto.Cipher import AES
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA
from Crypto.Signature.pkcs1_15 import PKCS115_SigScheme
from Crypto.Util.Padding import pad, unpad


secret_iv_prefix = "ceGWgZw6HSjlNnjQ+JRv"
secret_key_prefix = "RqgNv3l8ix6XjSGL+NyA"
secret_suffix = "6w=="
license_secret_pub_key = "LS0tLS1CRUdJTiBQVUJMSUMgS0VZLS0tLS0KTUlJQklqQU5CZ2txaGtpRzl3MEJBUUVGQUFPQ0FROEFNSUlCQ2dLQ0FRRUF0ZUFCUFl0dEpnTDA5aE5ORXNVRAozY3hUeE0wM3dTY2pQc1dWZ3BQZ2tlZVhXL3M5SVFEeTkvSDNlMjc4cUFRSmc5a3I3dkxlVlk1U2xvd2d1V2xLCldUZ1RMV0VsQ3Y2YTRPczVqeU80dlpwaEsxb0NabGsvSWNFWlQ2b2E2cjA0UVNmWUZQYzhSaFdybUZONllWemIKaGxxREhWU1BDZXE1NkpodWhTeDdjNkh5MDJJNEtvSUQ2cFA4Nzl2Z3RHNHRHTFgyRnN3SnhFZmhmQ1pPRDRuMgptTThjNFFhQ3I1OW5QTFpBU3IvRlM1MFhyQStBZVhpYzliemcxTThFbnFSRFN6QVpXTW9rRWY0Y3dhNmRzSGIxCnZneVdkSGV3SVB2UExNZW41cVpkTXVUREt4U0tmRHFnRTF4UWhLZnhOU2d2KzUzRGMyelZNWEFlNmVpVHBnS2EKeHdJREFRQUIKLS0tLS1FTkQgUFVCTElDIEtFWS0tLS0t"


def encrypt(msg: str) -> str:
    """return base-64"""
    data = msg.encode("utf-8")
    iv = b64decode(secret_iv_prefix + secret_suffix)
    key = b64decode(secret_key_prefix + secret_suffix)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    encrypted = cipher.encrypt(pad(data, AES.block_size))
    encrypted_msg = b64encode(encrypted).decode("utf-8")
    return encrypted_msg


def decrypt(encrypted_msg: str) -> str:
    data = b64decode(encrypted_msg)
    iv = b64decode(secret_iv_prefix + secret_suffix)
    key = b64decode(secret_key_prefix + secret_suffix)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted = unpad(cipher.decrypt(data), AES.block_size)
    decrypted_msg = decrypted.decode()
    return decrypted_msg


def read_license(license_file_path):
    if not path.isfile(license_file_path):
        return None
    with open(license_file_path) as f:
        data = json.load(f)
    return data


def verify_license_of_key(license, license_secret_sign, secret_pub_key):
    license_msg = json.dumps(license)
    license_data = license_msg.encode("utf-8")
    encrypted_pub_key = b64decode(secret_pub_key)
    pub_key = RSA.import_key(encrypted_pub_key)
    signature = b64decode(license_secret_sign)
    license_hash = SHA256.new(license_data)
    verifier = PKCS115_SigScheme(pub_key)
    try:
        verifier.verify(license_hash, signature)
        return True
    except:
        return False


def verify_license(license, license_secret_sign):
    return verify_license_of_key(license, license_secret_sign, license_secret_pub_key)
