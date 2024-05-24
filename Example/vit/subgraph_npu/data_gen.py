import numpy as np

print("-------gen vit_181--------")
t = np.ones([1, 197, 768], dtype=np.float32)

data = {"vit_181": t}

np.savez("vit_181.npz", **data)

# t = np.ones([1, 197, 768], dtype=np.float32)
# t2 = np.ones([1, 197, 1], dtype=np.float32)

# data = {"182": t, "183": t2}

# np.savez("test.npz", **data)




