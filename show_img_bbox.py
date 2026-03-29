import json
from pathlib import Path
from PIL import Image, ImageDraw
from typing import List, Tuple, Sequence
from matplotlib import pyplot as plt
from matplotlib import patches

# 直接通过jsonl文件画出bbox的位置
# 定义缩放边框的函数

def rescale_bbox(
    bbox: Sequence[Sequence[float]], 
    src: Tuple[int, int], 
    tgt: Tuple[int, int]
) -> Sequence[Sequence[float]]:
    assert len(src) == len(tgt) == 2
    ratio = [tgt[0] / src[0], tgt[1] / src[1]] * 2
    bbox = [[int(round(i * j)) for i, j in zip(entry, ratio)] for entry in bbox]
    return bbox

# 设定路径
image_folder = Path("/root/autodl-tmp/TSR/my_test_img")  # 图像文件夹路径
json_file_path = Path("/root/autodl-tmp/TSR/experiments/pub_bbox_large_final/test_results/pub_bbox_large_final/html_table_result_0.json")   # 单个 JSON 文件路径
# json_file_path = Path("./../../bbox_final_mamba_nvm_com_ld.json")   # 单个 JSON 文件路径
output_folder = Path("/root/autodl-tmp/TSR/vis_results")  # 输出文件夹路径
output_folder.mkdir(parents=True, exist_ok=True)

# 读取 JSON 文件
with open(json_file_path, 'r') as f:
    data = json.load(f)

# 获取所有图像文件
image_files = [f for f in image_folder.glob('*') if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']]  # Add more extensions if necessary

# 处理每个图像文件
for image_file in image_files:
    image_name = image_file.name

    # 去掉前缀 "j2_j1" 来匹配 JSON 数据
    image_name_without_prefix = image_name.replace("j2_j1_", "")

    if image_name_without_prefix not in data:
        print(f"未找到图像 {image_name_without_prefix} 对应的 JSON 数据，跳过。")
        continue

    # 获取图像路径
    image_path = image_file

    # 加载图像
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # 获取预测框和真实框
    image_data = data[image_name_without_prefix]
    pred_bboxes = image_data.get("pred", [])
    gt_bboxes = image_data.get("gt", [])

    print(f"[{image_name_without_prefix}] Sample Pred Raw: {pred_bboxes[0] if pred_bboxes else 'Empty'}")
    print(f"[{image_name_without_prefix}] GT Raw: {gt_bboxes[:1]}")

    # unitable和我们的模型需要缩放边框，TableMaster和SLANet不需要缩放
    pred_bboxes = rescale_bbox(pred_bboxes, src=(448, 448), tgt=image.size)
    gt_bboxes = rescale_bbox(gt_bboxes, src=(448, 448), tgt=image.size)

    # 用 Matplotlib 绘制框
    fig, ax = plt.subplots(figsize=(12, 10))
    for i in pred_bboxes:
        rect = patches.Rectangle(i[:2], i[2] - i[0], i[3] - i[1], linewidth=1, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    ax.set_axis_off()
    ax.imshow(image)

    # 保存输出图像，去掉扩展名
    output_name = image_name_without_prefix.rsplit(".", 1)[0]  # 去掉可能存在的最后一个扩展名
    output_path = output_folder / f"{output_name}_uni_ft.png"  # Add '.png' extension here
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Saved annotated image to: {output_path}")


# # 加载模型推理图片并画出bbox位置
# from typing import Tuple, List, Sequence, Optional, Union
# from pathlib import Path
# import re
# import torch
# import tokenizers as tk
# from PIL import Image
# from matplotlib import pyplot as plt
# from matplotlib import patches
# from torchvision import transforms
# from torch import nn, Tensor
# from functools import partial
# from bs4 import BeautifulSoup as bs
# import warnings
# import json
# import os
# import sys

# # 获取当前脚本所在目录的上一级目录
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# from src.model import EncoderDecoder, ImgLinearBackbone, Encoder, Decoder
# from src.utils import subsequent_mask, pred_token_within_range, greedy_sampling, bbox_str_to_token_list, cell_str_to_token_list, html_str_to_token_list, build_table_from_html_and_cell, html_table_template
# from src.trainer.utils import VALID_HTML_TOKEN, VALID_BBOX_TOKEN, INVALID_CELL_TOKEN

# warnings.filterwarnings('ignore')
# device = torch.device("cuda:0")

# image_folder = "/root/autodl-tmp/TSR/my_test_img"
# folder = Path(image_folder)
# image_files = [f for f in folder.glob('*') if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']]  # Add more extensions if necessary
# # image_name = "test1.png"
# # image_name = "test2.png"
# # image_path = f"./{image_name}"
# # output_file = "./output.jsonl"

# # output_folder = Path("/home/hj/data/zpl/uni_show/exact_bbox_val/bbox_show")
# output_folder = Path("/root/autodl-tmp/TSR/vis_results")
# output_folder.mkdir(parents=True, exist_ok=True)

# # 假设已经加载了 HTML 结构和 BBOX 检测模型
# # html_model, html_vocab = None, "/home/hj/data/zpl/uni_show/unitable_show_our/src/vocab/vocab_html.json"  # 需要实际的模型加载代码
# # bbox_model, bbox_vocab = "./best_mamba_nvm_com_ld.pt", "/home/hj/data/zpl/uni_show/unitable_show_our/vocab/vocab_bbox.json"  # 需要实际的模型加载代码
# bbox_model = "/root/autodl-tmp/TSR/experiments/pub_bbox_large_final/model/best.pt"
# bbox_vocab = "/root/autodl-tmp/TSR/vocab/vocab_bbox.json"

# for image_file in image_files:
#     print(f"Processing image: {image_file.name}")

#     image = Image.open(image_file).convert("RGB")
#     image_size = image.size

#     # fig, ax = plt.subplots(figsize=(12, 10))
#     # ax.imshow(image)

#     # model
#     d_model = 768
#     patch_size = 16
#     nhead = 12
#     dropout = 0.2

#     backbone = ImgLinearBackbone(d_model=d_model, patch_size=patch_size)
#     encoder = Encoder(
#         d_model=d_model,
#         nhead=nhead,
#         dropout = dropout,
#         activation="gelu",
#         norm_first=True,
#         nlayer=12,
#         ff_ratio=4,
#     )
#     decoder = Decoder(
#         d_model=d_model,
#         nhead=nhead,
#         dropout = dropout,
#         activation="gelu",
#         norm_first=True,
#         nlayer=4,
#         ff_ratio=4,
#     )

#     def autoregressive_decode(
#         model: EncoderDecoder,
#         image: Tensor,
#         prefix: Sequence[int],
#         max_decode_len: int,
#         eos_id: int,
#         token_whitelist: Optional[Sequence[int]] = None,
#         token_blacklist: Optional[Sequence[int]] = None,
#     ) -> Tensor:
#         model.eval()
#         with torch.no_grad():
#             memory = model.encode(image)
#             context = torch.tensor(prefix, dtype=torch.int32).repeat(image.shape[0], 1).to(device)

#         for _ in range(max_decode_len):
#             eos_flag = [eos_id in k for k in context]
#             if all(eos_flag):
#                 break

#             with torch.no_grad():
#                 # causal_mask = local_attention_mask(context.shape[1]).to(device)
#                 causal_mask = subsequent_mask(context.shape[1]).to(device)
#                 logits = model.decode(
#                     memory, context, tgt_mask=causal_mask, tgt_padding_mask=None
#                 )
#                 logits = model.generator(logits)[:, -1, :]

#             logits = pred_token_within_range(
#                 logits.detach(),
#                 white_list=token_whitelist,
#                 black_list=token_blacklist,
#             )

#             next_probs, next_tokens = greedy_sampling(logits)
#             context = torch.cat([context, next_tokens], dim=1)
#         return context

#     def load_vocab_and_model(
#         vocab_path: Union[str, Path],
#         max_seq_len: int,
#         model_weights: Union[str, Path],
#     ) -> Tuple[tk.Tokenizer, EncoderDecoder]:
#         vocab = tk.Tokenizer.from_file(vocab_path)
#         model = EncoderDecoder(
#             backbone=backbone,
#             encoder=encoder,
#             decoder=decoder,
#             vocab_size=vocab.get_vocab_size(),
#             d_model=d_model,
#             padding_idx=vocab.token_to_id("<pad>"),
#             max_seq_len=max_seq_len,
#             dropout=dropout,
#             norm_layer=partial(nn.LayerNorm, eps=1e-6)
#         )

#         model.load_state_dict(torch.load(model_weights, map_location="cpu"))
#         model = model.to(device)
#         return vocab, model

#     def image_to_tensor(image: Image, size: Tuple[int, int]) -> Tensor:
#         T = transforms.Compose([
#             transforms.Resize(size),
#             transforms.ToTensor(),
#             # transforms.Normalize(mean=[0.86597056,0.88463002,0.87491087], std = [0.20686628,0.18201602,0.18485524])
#             # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#         image_tensor = T(image)
#         image_tensor = image_tensor.to(device).unsqueeze(0)

#         return image_tensor

#     def rescale_bbox(
#         bbox: Sequence[Sequence[float]],
#         src: Tuple[int, int],
#         tgt: Tuple[int, int]
#     ) -> Sequence[Sequence[float]]:
#         assert len(src) == len(tgt) == 2
#         ratio = [tgt[0] / src[0], tgt[1] / src[1]] * 2
#         bbox = [[int(round(i * j)) for i, j in zip(entry, ratio)] for entry in bbox]
#         return bbox

#     # # Table structure extraction
#     # vocab, model = load_vocab_and_model(
#     #     vocab_path=html_vocab,
#     #     max_seq_len=784,
#     #     model_weights=html_model,
#     # )

#     # # Image transformation
#     # image_tensor = image_to_tensor(image, size=(448, 448))

#     # # Inference
#     # pred_html = autoregressive_decode(
#     #     model=model,
#     #     image=image_tensor,
#     #     prefix=[vocab.token_to_id("[html]")],
#     #     max_decode_len=512,
#     #     eos_id=vocab.token_to_id("<eos>"),
#     #     token_whitelist=[vocab.token_to_id(i) for i in VALID_HTML_TOKEN],
#     #     token_blacklist = None
#     # )

#     # # Convert token id to token text
#     # pred_html = pred_html.detach().cpu().numpy()[0]
#     # pred_html = vocab.decode(pred_html, skip_special_tokens=False)
#     # pred_html = html_str_to_token_list(pred_html)

#     # print(pred_html)

#     # Table cell bbox detection
#     vocab, model = load_vocab_and_model(
#         vocab_path=bbox_vocab,
#         max_seq_len=1024,
#         model_weights=bbox_model,
#     )

#     # Image transformation
#     image_tensor = image_to_tensor(image, size=(448, 448))

#     # Inference
#     pred_bbox = autoregressive_decode(
#         model=model,
#         image=image_tensor,
#         prefix=[vocab.token_to_id("[bbox]")],
#         max_decode_len=1024,
#         eos_id=vocab.token_to_id("<eos>"),
#         token_whitelist=[vocab.token_to_id(i) for i in VALID_BBOX_TOKEN[: 449]],
#         token_blacklist = None
#     )

#     # Convert token id to token text
#     pred_bbox = pred_bbox.detach().cpu().numpy()[0]
#     pred_bbox = vocab.decode(pred_bbox, skip_special_tokens=False)

#     # print(pred_bbox)

#     # Visualize detected bbox
#     pred_bbox = bbox_str_to_token_list(pred_bbox)
#     pred_bbox = rescale_bbox(pred_bbox, src=(448, 448), tgt=image_size)

#     fig, ax = plt.subplots(figsize=(12, 10))
#     for i in pred_bbox:
#         rect = patches.Rectangle(i[:2], i[2] - i[0], i[3] - i[1], linewidth=1, edgecolor='r', facecolor='none')
#         ax.add_patch(rect)
#     ax.set_axis_off()
#     ax.imshow(image)

#     output_path = output_folder / f"ourld_{image_file.name}"
#     fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
#     plt.close(fig)
#     print(f"Saved annotated image to: {output_path}")


