import json
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import patches


def rescale_bbox(bbox, src, tgt):
    ratio = [tgt[0] / src[0], tgt[1] / src[1]] * 2
    return [
        [int(round(i * j)) for i, j in zip(entry, ratio)]
        for entry in bbox
    ]


# ====== 手动指定三个路径 ======
import json
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import patches


def rescale_bbox(bbox, src, tgt):
    ratio = [tgt[0] / src[0], tgt[1] / src[1]] * 2
    return [
        [int(round(i * j)) for i, j in zip(entry, ratio)]
        for entry in bbox
    ]


# ====== 修改成你需要的路径 ======
# image_path = Path("root/autodl-tmp/TSR/my_test_img")
image = input("输入：")
image_path = Path(f"/root/autodl-tmp/TSR/data/pubtabnet/val/{image}")
# json_path = Path("/root/autodl-tmp/unitable/experiments/pubtabnet/ssp_2m_mini_bbox_base/final.json")
json_path = Path("/root/autodl-tmp/TSR/experiments/pub_bbox_large_final/test_results/pub_bbox_large_final/html_table_result_0.json")


# 输出文件名（一定要以 .png 结尾）
output_path = Path("/root/autodl-tmp/TSR/vis_results/PMC535543_007_01.png")
# ==================================


# 加载 JSON
with open(json_path, "r") as f:
    data = json.load(f)

# 找图片名
image_name = image_path.name

# 取出该图片的 bbox 信息
if image_name not in data:
    raise KeyError(f"JSON 中没有找到图像 {image_name} 的记录，你的 final.json 里可能是别的名字")

entry = data[image_name]
pred_bboxes = entry.get("pred", [])
gt_bboxes = entry.get("gt", [])

# 读取图片
img = Image.open(image_path).convert("RGB")

# 缩放 bbox 到原图尺寸
pred_bboxes = rescale_bbox(pred_bboxes, src=(448, 448), tgt=img.size)
gt_bboxes = rescale_bbox(gt_bboxes, src=(448, 448), tgt=img.size)

# 画图
fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(img)
ax.set_axis_off()

# Pred: 红色
for box in pred_bboxes:
    rect = patches.Rectangle(
        (box[0], box[1]),
        box[2] - box[0],
        box[3] - box[1],
        linewidth=2,
        edgecolor="red",
        facecolor="none"
    )
    ax.add_patch(rect)

# GT: 蓝色
for box in gt_bboxes:
    rect = patches.Rectangle(
        (box[0], box[1]),
        box[2] - box[0],
        box[3] - box[1],
        linewidth=2,
        edgecolor="blue",
        facecolor="none"
    )
    ax.add_patch(rect)

# 保存
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
plt.close(fig)

print(f"Saved to: {output_path}")

# ==================================

# 加载 JSON
with open(json_path, "r") as f:
    data = json.load(f)

# 取文件名（带后缀）
image_name = image_path.name

# 取出该图片的 pred / gt
if image_name not in data:
    raise KeyError(f"JSON 中没有找到图像 {image_name} 的记录")

entry = data[image_name]
pred_bboxes = entry.get("pred", [])
gt_bboxes = entry.get("gt", [])

# 读取图片
img = Image.open(image_path).convert("RGB")
img_w, img_h = img.size

# ====== 核心修复逻辑 ======

# 1. 打印第一条数据来看看真面目 (Debug)
if len(pred_bboxes) > 0:
    sample_box = pred_bboxes[0]
    print(f"🔍 Debug - 原始数据示例: {sample_box}")
    
    # 智能判断缩放: 
    # 如果坐标都很小 (<1)，说明是归一化坐标 -> src=(1,1)
    # 如果坐标接近原图尺寸 -> 不需要缩放
    is_normalized = all(x <= 1.5 for x in sample_box)
    
    if is_normalized:
        print("检测到归一化坐标 (0-1)，正在缩放...")
        pred_bboxes = rescale_bbox(pred_bboxes, src=(1, 1), tgt=img.size)
    else:
        # 如果最大坐标都已经很大了（比如超过 448 的一半），可能不需要缩放，或者 src 不对
        # 建议先尝试：不缩放（假设模型输出的就是原图坐标）
        print("检测到绝对坐标，尝试直接使用 (跳过 448 缩放)...")
        # pred_bboxes = rescale_bbox(pred_bboxes, src=(448, 448), tgt=img.size) # <--- 先注释掉这一行！
else:
    print("⚠️ 没有预测框数据")

# 缩放到原图大小
pred_bboxes = rescale_bbox(pred_bboxes, src=(448, 448), tgt=img.size)
gt_bboxes = rescale_bbox(gt_bboxes, src=(448, 448), tgt=img.size)

# 画框
fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(img)
ax.set_axis_off()

# 预测框 = 红色
for box in pred_bboxes:
    # x, y, w, h = box[0], box[1], box[2], box[3]
    # rect = patches.Rectangle(
    #     (x, y), 
    #     w, h,
    #     linewidth=2,
    #     edgecolor="red",
    #     facecolor="none",
    #     label="Pred"
    # )

    rect = patches.Rectangle(
        (box[0], box[1]),
        box[2] - box[0],
        box[3] - box[1],
        linewidth=2,
        edgecolor="red",
        facecolor="none"
    )
    ax.add_patch(rect)

# GT 框 = 蓝色
for box in gt_bboxes:
    rect = patches.Rectangle(
        (box[0], box[1]),
        box[2] - box[0],
        box[3] - box[1],
        linewidth=2,
        edgecolor="blue",
        facecolor="none"
    )
    ax.add_patch(rect)

# 保存
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
plt.close(fig)

print(f"Saved to: {output_path}")