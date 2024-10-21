from abc import ABC
from typing import List, Dict, Set

class Device(ABC):
    def __init__(self):
        pass

    def get_type(self) -> str:
        """Returns the type of the device."""
        pass

    def get_cpu_structure(self) -> List[List[str]]:
        """Returns the CPU structure."""
        return []

    def get_npu_structure(self) -> List[List[str]]:
        """Returns the NPU structure."""
        return [
            ["FULLY_CONNECTED", "RESHAPE", "TRANSPOSE", "RESHAPE","FULLY_CONNECTED", "RESHAPE", "TRANSPOSE", "RESHAPE","FULLY_CONNECTED", "RESHAPE", "TRANSPOSE", "RESHAPE",
             "FULLY_CONNECTED", "RESHAPE", "TRANSPOSE", "RESHAPE","FULLY_CONNECTED", "RESHAPE", "TRANSPOSE", "RESHAPE","FULLY_CONNECTED", "RESHAPE", "TRANSPOSE", "RESHAPE",
             "FULLY_CONNECTED", "RESHAPE", "TRANSPOSE", "RESHAPE","FULLY_CONNECTED", "RESHAPE", "TRANSPOSE", "RESHAPE","FULLY_CONNECTED", "RESHAPE", "TRANSPOSE", "RESHAPE",
             "FULLY_CONNECTED", "RESHAPE", "TRANSPOSE", "RESHAPE","FULLY_CONNECTED", "RESHAPE", "TRANSPOSE", "RESHAPE","FULLY_CONNECTED", "RESHAPE", "TRANSPOSE", "RESHAPE",
             "FULLY_CONNECTED", "RESHAPE", "TRANSPOSE", "RESHAPE","FULLY_CONNECTED", "RESHAPE", "TRANSPOSE", "RESHAPE","FULLY_CONNECTED", "RESHAPE", "TRANSPOSE", "RESHAPE",
             "FULLY_CONNECTED", "RESHAPE", "TRANSPOSE", "RESHAPE","FULLY_CONNECTED", "RESHAPE", "TRANSPOSE", "RESHAPE","FULLY_CONNECTED", "RESHAPE", "TRANSPOSE", "RESHAPE",],
            ["ADD", "FULLY_CONNECTED", "LOGISTIC", "MUL", "FULLY_CONNECTED", "LOGISTIC", "MUL" ],
            ["RESHAPE", "LOGISTIC", "MUL", "PAD", "CONV_2D", "PAD", "CONV_2D", "ADD"],
            ["RESHAPE", "PAD", "CONV_2D", "ADD", "PAD", "CONV_2D"],
            ["RESHAPE", "LOGISTIC", "MUL", "PAD", "CONV_2D"],
            ["MUL","MUL","SUB","ADD","FULLY_CONNECTED"],
            ["FULLY_CONNECTED", "RESHAPE", "TRANSPOSE", "RESHAPE"],
            ["RESHAPE", "TRANSPOSE", "RESHAPE", "FULLY_CONNECTED"],
            ["FULLY_CONNECTED", "ADD","RESHAPE"],
            ["PAD","CONV_2D", "RESHAPE"],
            ["PAD", "CONV_2D"],
            ["CONV_2D"]
        ]

    def get_npu_support_op(self) -> List[str]:
        """Returns the list of supported NPU operations. add Transpose"""
        return ["CONV_2D", "RESHAPE", "ADD", "MUL", "LOGISTIC", "PAD","TRANSPOSE", "FULLY_CONNECTED","SUB"]

    def get_cpu_support_op(self) -> List[str]:
        """Returns the list of supported CPU operations."""
        return ["Sub", "Pow", "ReduceMean", "ADD", "Sqrt", "Div","Transpose", "Gather", "MatMul", "MUL", "Softmax", "Erf", "Gemm", "CONV_2D", "RESHAPE",
                "Sin", "Where", "ConstantOfShape", "Cast", "LOGISTIC", "Cos", "Expand", "Slice", "Unsqueeze"]

    def get_npu_prefer_op(self) -> List[str]:
        """Returns the preferred NPU operations."""
        return ["CONV_2D"]
