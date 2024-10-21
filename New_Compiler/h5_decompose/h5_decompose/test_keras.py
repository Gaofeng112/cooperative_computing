import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth for each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
from stable_diffusion_tf.diffusion_model import (
    BkTinyLCMSplitUNetModel,
    BkTinyLCMUNetModel,
    BkTinyLCMUNetModelAll,
)
custom_objects={'BkTinyLCMUNetModelAll':BkTinyLCMUNetModelAll}
path_to_model = "../../net/diffusion_model.hdf5"
diffusion = tf.keras.models.load_model(path_to_model,custom_objects = custom_objects)

# 打印模型概要
diffusion.summary()

# 获取模型配置
config = diffusion.get_config()

# 获取模型层
layers = diffusion.layers

# # 获取部分层
partial_diffusion = tf.keras.diffusions.diffusion(inputs=diffusion.input, 
                                   outputs=diffusion.get_layer('layer_name').output)

# # 或者使用函数式API创建子模型
# input_tensor = tf.keras.layers.Input(shape=[1, 32, 32, 4])
# x = diffusion.layers[1](input_tensor)
# x = diffusion.layers[2](x)
# output = diffusion.layers[3](x)
# sub_diffusion = tf.keras.diffusions.diffusion(inputs=input_tensor, outputs=output)