import os
import json
from PIL import Image, ImageDraw


# #修改“split”的值
# # 输入的JSONL文件路径
# input_file = 'train.jsonl'
# # 输出的JSONL文件路径
# output_file = 'train2.jsonl '

# # 读取原文件并修改数据
# with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
#     for line in infile:
#         # 解析JSONL中的每一行
#         data = json.loads(line)

#         # 修改 'split' 值
#         data['split'] = 'train'

#         # 将修改后的数据写入输出文件
#         json.dump(data, outfile, ensure_ascii=False)
#         outfile.write('\n')  # 每行末尾换行
# print("Processing complete")


# 读取原始 jsonl 文件并修改html结构
def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # 加载每行的 JSON 数据
            data = json.loads(line)

            # 处理 "structure" 中的 "tokens" 部分
            tokens = data.get("html", {}).get("structure", {}).get("tokens", [])
            if tokens:
                # 替换目标字符串
                for i, token in enumerate(tokens):
                    if token == "<td>[":
                        tokens[i] = "<td>"
                    elif token == "]</td>":
                        tokens[i] = "</td>"
                    elif token == ">[":
                        tokens[i] = ">"

            # 将修改后的数据写回输出文件
            json.dump(data, outfile, ensure_ascii=False)
            outfile.write("\n")

# # 输入输出文件路径
# input_file = 'val(unitable).jsonl'  # 你可以替换为你的输入文件路径
# output_file = 'output_val.jsonl'  # 你可以替换为你的输出文件路径

# # 调用函数处理文件
# process_jsonl(input_file, output_file)
# print("Processing complete")


# 根据txt修改jsonl里的bbox信息
def update_jsonl_with_txt(jsonl_file, txt_file):
    # 读取 JSONL 文件内容
    with open(jsonl_file, 'r', encoding='utf-8') as jsonl:
        jsonl_data = [json.loads(line) for line in jsonl.readlines()]

    # 读取 TXT 文件内容，并解析每个条目
    txt_data = {}
    with open(txt_file, 'r', encoding='utf-8') as txt:
        for line in txt.readlines():
            filename, annotations = line.strip().split('\t')
            annotations = json.loads(annotations)
            # 去掉前缀，只保留文件名
            filename = filename.split('/')[-1]
            txt_data[filename] = annotations

    # 遍历 JSONL 文件数据，更新相关信息
    for entry in jsonl_data:
        filename = entry["filename"]

        if filename in txt_data:  # 如果 JSONL 中有对应的条目
            annotations = txt_data[filename]

            # 更新 html.cells 部分，首先清除原有的 cells
            entry["html"]["cells"] = []  # 清除原有的 cells 数据

            # 为每个 annotations 中的 points 数据生成新的 bbox
            for ann in annotations:
                points = ann["points"]
                new_bbox = [points[0][0], points[0][1], points[2][0], points[2][1]]

                # 在 html.cells 下生成新的数据
                entry["html"]["cells"].append({"tokens": [], "bbox": new_bbox})

        # 如果没有找到对应的 TXT 数据，保留原有的 JSONL 数据
        else:
            print(f"Warning: No matching entry found for {filename} in txt data.")

    # 保存更新后的 JSONL 文件
    with open('updated_' + jsonl_file, 'w', encoding='utf-8') as jsonl:
        for entry in jsonl_data:
            jsonl.write(json.dumps(entry, ensure_ascii=False) + '\n')

# 使用示例
# update_jsonl_with_txt('train.jsonl' 'Label_train.txt')
# print("Processing complete"),
'''

'''
# show
# 输入输出文件夹
# input_jsonl_file = 'D:\实验\表格实验\\unitable_2\表格数据集制作\FTN-25_modify_2(clean_add)\\add_data\\val.jsonl'  # JSONL文件路径
input_jsonl_file = '/root/autodl-tmp/TSR/data/pubtabnet/PubTabNet_2.0.0.jsonl'
# input_image_folder = 'D:\实验\表格实验\\unitable_2\表格数据集制作\FTN-25_modify_2(clean_add)\\add_data\\val'  # 存放图片的文件夹路径
input_image_folder = '/root/autodl-tmp/TSR/data/pubtabnet/val'
# input_image_folder = 'autodl-tmp/TSR/my_test_img'
# output_folder = 'D:\实验\表格实验\\unitable_2\表格数据集制作\FTN-25_modify_2(clean_add)\\add_data\\show_val_beforemod'  # 输出保存画框图片的文件夹路径
output_folder = '/root/autodl-tmp/TSR/vis_111'

# 如果输出文件夹不存在，创建它
os.makedirs(output_folder, exist_ok=True)

# 处理jsonl文件
with open(input_jsonl_file, 'r') as f:
    for line in f:
        # 加载每行json数据
        data = json.loads(line.strip())

        # imgid = data.get('imgid', '')
        # PubTabNet 里的字段叫 filename
        filename = data.get('filename', '')
        # 顺便加个判断，只画验证集 (val)
        if data.get('split') != 'val': continue
        # bbox_list = [cell['bbox'] for cell in data['html']['cells']]
        bbox_list = [cell['bbox'] for cell in data['html']['cells'] if 'bbox' in cell]

        # 加载对应的图片
        # image_path = os.path.join(input_image_folder, f'{imgid}.jpg')
        # filename 里面已经包含了 .png 后缀，不用自己拼了
        image_path = os.path.join(input_image_folder, filename) 
        if not os.path.exists(image_path):
            print(f"Image {filename} not found!")
            continue

        # 打开图片 
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)

        # 在图片上绘制每个框
        for bbox in bbox_list:
            left, top, right, bottom = bbox

            # 检查并交换left/right和top/bottom的值，确保它们是有效的矩形坐标
            if left > right or top > bottom:
                print(f"Invalid bbox in image {filename}: {bbox}")

            left, right = min(left, right), max(left, right)
            top, bottom = min(top, bottom), max(top, bottom)

            # 绘制矩形
            draw.rectangle([left, top, right, bottom], outline="red", width=2)

        # 保存绘制框后的图片
        # output_image_path = os.path.join(output_folder, f'{imgid}_bbox.jpg')
        output_image_path = os.path.join(output_folder, f'GT_{filename}')
        img.save(output_image_path)
        print(f"Saved {output_image_path}")

print("Processing complete.")
