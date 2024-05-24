import onnx
from onnx import shape_inference

# 加载ONNX文件
model = onnx.load('unet/model.onnx')

# 设置输入形状
model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 4
model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 224
model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = 224

model.graph.input[1].type.tensor_type.shape.dim[0].dim_value = 1

model.graph.input[2].type.tensor_type.shape.dim[0].dim_value = 1
model.graph.input[2].type.tensor_type.shape.dim[1].dim_value = 1
model.graph.input[2].type.tensor_type.shape.dim[2].dim_value = 768

# 执行形状推理
inferred_model = shape_inference.infer_shapes(model)

from onnxsim import simplify

# 简化模型
model_simp, check = simplify(inferred_model)

# 检查简化后的模型是否有效
assert check, "Simplified ONNX model could not be validated"

# 保存简化后的模型
onnx.save(model_simp, './final_model.onnx')
