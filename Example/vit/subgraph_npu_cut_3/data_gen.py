import numpy as np

print("-------gen vit_181--------")
t = np.ones([1, 197, 768], dtype=np.float32)

data = {"vit_181": t}

np.savez("vit_181.npz", **data)

# print("-------gen vit_1215--------")
# t1 = np.ones([1, 768], dtype=np.float32)

# data1 = {"vit_1215": t1}

# np.savez("vit_1215.npz", **data1)

# 读取二进制文件
data = np.fromfile('vit_1215.bin', dtype=np.float32)

# 将数据重新整形为所需的形状，这里假设数据形状为 (1, 768)
data = data.reshape((1, 768))

# 将数据保存为npz文件
np.savez('vit_1215.npz', vit_1215=data)


# t = np.ones([1, 197, 768], dtype=np.float32)
# t2 = np.ones([1, 197, 1], dtype=np.float32)

# data = {"182": t, "183": t2}

# np.savez("test.npz", **data)




