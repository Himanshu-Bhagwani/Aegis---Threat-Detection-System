"""Hashing helpers"""

def sha1(text):
    import hashlib
    return hashlib.sha1(text.encode()).hexdigest()
