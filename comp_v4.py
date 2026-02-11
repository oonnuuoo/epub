import os
import sys
from PIL import Image

args = sys.argv

parent_dir_odd = '../../' + args[3] + '/odd'
parent_dir_even = '../../' + args[3] + '/even'

# dir1 = '../' + args[3] + '/odd/padded/resized'
# dir2 = '../' + args[3] + '/even/padded/resized'
# dir3 = '../' + args[3] + '/odd/table/padded/resized'
# dir4 = '../' + args[3] + '/even/table/padded/resized'

def compress_images(input_folder):
    # "comp"フォルダを作成
#    output_folder = os.path.join(input_folder, 'comp')
    q_value = int(args[1])
    output_folder = '../../' + args[3] + '/comp'
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
    q_value = int(args[2])
    output_folder = '../../' + args[3] + '/comp'
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

dir_list_odd = [
    f for f in os.listdir(parent_dir_odd) if os.path.isdir(os.path.join(parent_dir_odd, f))
]

dir_list_even = [
    f for f in os.listdir(parent_dir_even) if os.path.isdir(os.path.join(parent_dir_even, f))
]

# for i in range(len(dir_list_odd)):
compress_images(parent_dir_odd + '/cropped/margin_added')
compress_images(parent_dir_odd + '/cropped/ex')
compress_images_ex(parent_dir_odd + '/ex/resized')

# for i in range(len(dir_list_even)):
compress_images(parent_dir_even + '/cropped/margin_added')
compress_images(parent_dir_even + '/cropped/ex')
compress_images_ex(parent_dir_even + '/ex/resized')

# compress_images(dir1)
# compress_images(dir2)
# compress_images(dir3)
# compress_images(dir4)
# compress_images_ex(dir1 + '/ex')
# compress_images_ex(dir2 + '/ex')
# compress_images_ex(dir3 + '/ex')
# compress_images_ex(dir4 + '/ex')