import ctypes
import numpy as np

def float2ascii(n, bchars):
    n = ctypes.c_uint32.from_buffer(ctypes.c_float(n)).value
    base = len(bchars)
    if n == 0: return bchars[0]
    res = ''
    while n > 0:
        digit = n % base
        res = bchars[digit] + res
        n = n // base
    return res

asciiCharSet = r"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"

reserved = r" \t\n\r\x0b\x0c"

customBase = asciiCharSet

for i in reserved:
    customBase.replace(i, "")

number = (np.random.normal(size=(1))[0]).astype(np.float32)
encoded = float2ascii(number, customBase)

print(number)
print(encoded)