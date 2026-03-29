import sys
import os

# 将当前目录添加到 Python 路径，这样就能找到 src 包了
sys.path.append(os.getcwd())

from src.utils.data import bbox_str_to_token_list

# 模拟模型输出的字符串 (基于你的 JSON 结果 [356, 356...])
fake_model_output = "bbox-356 bbox-356 bbox-356 bbox-356"

print(f"模拟模型输出: {fake_model_output}")

# 调用解析函数
try:
    decoded = bbox_str_to_token_list(fake_model_output)
    print(f"解析结果: {decoded}")
except Exception as e:
    print(f"解析出错: {e}")

# 测试正常情况
normal_output = "bbox-10 bbox-20 bbox-100 bbox-200"
print(f"\n模拟正常输出: {normal_output}")
decoded_normal = bbox_str_to_token_list(normal_output)
print(f"解析结果: {decoded_normal}")