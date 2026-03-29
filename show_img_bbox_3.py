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
# image_path = Path(f"/root/autodl-tmp/TSR/vis_gt/{image}")
# image_path = Path("autodl-tmp/TSR/vis_gt/PMC515302_003_00.png")
# json_path = Path("/root/autodl-tmp/unitable/experiments/pubtabnet/ssp_2m_mini_bbox_base/final.json")
json_path = Path("/root/autodl-tmp/TSR/experiments/pub_bbox_large_final/test_results/pub_bbox_large_final/html_table_result_0.json")


# 输出文件名（一定要以 .png 结尾）
output_path = Path(f"/root/autodl-tmp/TSR/vis_mini/{image}")
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
        edgecolor="green",
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

# 缩放到原图大小
pred_bboxes = rescale_bbox(pred_bboxes, src=(448, 448), tgt=img.size)
gt_bboxes = rescale_bbox(gt_bboxes, src=(448, 448), tgt=img.size)

# 画框
fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(img)
ax.set_axis_off()

# # 预测框 = 红色
# for box in pred_bboxes:
#     rect = patches.Rectangle(
#         (box[0], box[1]),
#         box[2] - box[0],
#         box[3] - box[1],
#         linewidth=2,
#         edgecolor="red",
#         facecolor="none"
#     )
#     ax.add_patch(rect)

# GT 框 = 蓝色
for box in gt_bboxes:
    rect = patches.Rectangle(
        (box[0], box[1]),
        box[2] - box[0],
        box[3] - box[1],
        linewidth=2,
        edgecolor="green",
        facecolor="none"
    )
    ax.add_patch(rect)

# 保存
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
plt.close(fig)

print(f"Saved to: {output_path}")