import numpy as np

# Define the path to your binary file
binary_file_path = 'sample.bin'

# Read the binary file into a NumPy array
data_array = np.fromfile(binary_file_path, dtype=np.float32)  # or another dtype if necessary

# Reshape the array to the desired shape
desired_shape = (1, 4, 32, 32)
data_array = data_array.reshape(desired_shape)

print(data_array)

# Create the dictionary with the specified key
data = {"sample": data_array}

# Save the dictionary to a .npz file
np.savez("sample.npz", **data)
