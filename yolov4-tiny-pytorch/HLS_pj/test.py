import torch
import torch.nn as nn
import torch.quantization
from yolo import YOLO


# 定义YOLOv4 Tiny模型结构

# 创建YOLOv4 Tiny模型实例
model = YOLO()

# 加载预训练权重
model.load_weights('/home/wma/yolov4-tiny-pytorch-master/logs/best_epoch_weights.pth')
# 定义一个评估方法，手动设置模型为评估模式
def evaluate_model(model):
    model.eval()

# 定义一个示例输入
example_input = torch.rand(1, 3, 416, 416)  # 假设输入尺寸与 YOLOv4 Tiny 模型的输入尺寸一致

# 将模型的权重量化为 8 位整数
quantized_model = torch.quantization.quantize_dynamic(
    model,  # 需要量化的模型
    {torch.nn.Conv2d, torch.nn.Linear},  # 需要量化的模块类型
    dtype=torch.qint8  # 目标数据类型
)

# 评估量化后的模型
evaluate_model(quantized_model)

# 评估量化后的模型
with torch.no_grad():
    output = quantized_model(example_input)

# 保存量化后的模型
torch.save(quantized_model.state_dict(), 'quantized_yolov4_tiny.pth')