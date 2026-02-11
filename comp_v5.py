import os
import sys
from PIL import Image

args = sys.argv

normal_comp = '../../' + args[1] + '/margin_added'
HQ_comp = normal_comp + '/figs'

# dir1 = '../' + args[3] + '/odd/padded/resized'
# dir2 = '../' + args[3] + '/even/padded/resized'
# dir3 = '../' + args[3] + '/odd/table/padded/resized'
# dir4 = '../' + args[3] + '/even/table/padded/resized'

def compress_images(input_folder):
    # "comp"フォルダを作成
#    output_folder = os.path.join(input_folder, 'comp')
    q_value = int(args[2])
    output_folder = '../../' + args[1] + '/comp'
    os.makedirs(output_folder, exist_ok=True)

    # フォルダ内の全ての.jpgファイルを取得
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg'):
            file_path = os.path.join(input_folder, filename)
            img = Image.open(file_path)
            
            # 圧縮して保存
            output_path = os.path.join(output_folder, filename)
            img.save(output_path, 'JPEG', quality=q_value)
            print(f"Compressed and saved {filename} to {output_folder}")
        

def compress_images_ex(input_folder):
    # 入力フォルダが存在しない場合はスキップ
    if not os.path.exists(input_folder):
        print(f"Skipping {input_folder} - folder does not exist")
        return
    
    # "comp"フォルダを作成
#    output_folder = os.path.join(input_folder, 'comp')
    q_value = int(args[3])
    output_folder = '../../' + args[1] + '/comp'
    os.makedirs(output_folder, exist_ok=True)

    # フォルダ内の全ての.jpgファイルを取得
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg'):
            file_path = os.path.join(input_folder, filename)
            img = Image.open(file_path)
            
            # 圧縮して保存
            output_path = os.path.join(output_folder, filename)
            img.save(output_path, 'JPEG', quality=q_value)
            print(f"Compressed and saved {filename} to {output_folder}")

# 使用例

# dir_list_odd = [
    # f for f in os.listdir(parent_dir_odd) if os.path.isdir(os.path.join(parent_dir_odd, f))
# ]

# dir_list_even = [
    # f for f in os.listdir(parent_dir_even) if os.path.isdir(os.path.join(parent_dir_even, f))
# ]


compress_images(normal_comp)
compress_images_ex(HQ_comp)
