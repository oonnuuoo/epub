from PIL import Image
from pathlib import Path
import os


def calculate_ratio(folder_path):
    """
    calculate average ratio of images in folder
    
    Args:
        folder_path: path to folder containing images
    
    Returns:
        average ratio of images in folder
    """
 
    files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    aspect_ratios = []
    for file in files:
        img = Image.open(os.path.join(folder_path, file))
        width, height = img.size
        ratio = height / width
        aspect_ratios.append(ratio)
    
    
    # omit minimum and maximum quartiles
    aspect_ratios.sort()
    length = len(aspect_ratios)
    aspect_ratios = aspect_ratios[length//4:length-length//4]

    # calculate average ratio
    average_ratio = sum(aspect_ratios) / len(aspect_ratios)
    return average_ratio


def add_horizontal_padding(img, target_ratio, margin_direction):
    """
    add white padding to the bottom or top

    Args:
        img: PIL Image Object
        target_ratio: objective height/width ratio
        margin_direction: 1 for top-aligned (padding at bottom), 3 for bottom-aligned (padding at top)

    Returns:
        PIL Image with padding added
    """

    width, height = img.size
    current_ratio = height / width

    if current_ratio >= target_ratio:
        # if ratio exceeds target_ratio, return as it is
        return img

    new_height = int(width * target_ratio)
    padding_height = new_height - height

    # create new image with white background
    new_img = Image.new('RGB', (width, new_height), (255, 255, 255))

    if margin_direction == 1:
        # place original image to the top
        new_img.paste(img, (0, 0))
    
    else:
        # place original image to the bottom
        new_img.paste(img, (0, padding_height))
    
    return new_img

def add_vertical_padding(img, target_ratio_inv, margin_direction):
    """
    add white padding to the left or right

    Args:
        img: PIL Image Object
        target_ratio_inv: objective width/height ratio
        margin_direction: 2 for right-aligned (padding at left), 4 for left-aligned (padding at right)

    Returns:
        PIL Image with padding added
    """

    width, height = img.size
    current_ratio_inv = width / height

    if current_ratio_inv >= target_ratio_inv:
        # if ratio exceeds target_ratio_inv, return as it is
        return img

    new_width = int(height * target_ratio_inv)
    padding_width = new_width - width

    # create new image with white background
    new_img = Image.new('RGB', (new_width, height), (255, 255, 255))

    if margin_direction == 2:
        # place original image to the right (padding on left)
        new_img.paste(img, (padding_width, 0))

    else:
        # place original image to the left (padding on right)
        new_img.paste(img, (0, 0))
    
    return new_img

def padding(folder_path, margin_direction, supported_formats=('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
    folder = Path(folder_path)

    if not folder.exists():
        print(f"ERROR: folder '{folder_path}' does not exist.")
        return

    # create folder "margin_added"
    output_folder = folder / "margin_added"
    output_folder.mkdir(exist_ok=True)

    # count images processed
    processed_count = 0

    # list temporally saved
    processed_images = []

    average_ratio = calculate_ratio(folder_path)
    print(f"平均アスペクト比（第一四分位-第三四分位）: {average_ratio:.3f}")

    # フォルダ直下の画像ファイルを読み込み
    print("画像ファイルを読み込んでいます...")
    for img_file in folder.iterdir():
        if img_file.is_file() and img_file.suffix.lower() in supported_formats:
            try:
                img = Image.open(img_file)
                print(f"読み込み: {img_file.name}")

                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                processed_images.append((img_file, img))

            except Exception as e:
                print(f" ERROR  :  {img_file.name} - {str(e)}")

    if not processed_images:
        print("処理対象の画像が見つかりません")
        return

    print(f"\n{len(processed_images)}個の画像を読み込みました\n")

    # Convert margin_direction to int for comparison
    margin_direction = int(margin_direction)

    if margin_direction == 1 or margin_direction == 3:
        # 最大のheight/width比を計算
        print("最大のheight/width比を計算")
        max_ratio = 0
        max_ratio_file = ""
        for img_file, img in processed_images:
            width, height = img.size
            ratio = height / width
            if ratio > max_ratio:
                max_ratio = ratio
                max_ratio_file = img_file.name

        print(f"最大アスペクト比: {max_ratio:.3f} (基準画像: {max_ratio_file})")
        print()

        # 各画像にパディングを追加して保存
        for img_file, img in processed_images:
            try:
                width, height = img.size
                current_ratio = height / width

                if current_ratio < max_ratio - 0.001:  # 浮動小数点の誤差を考慮
                    # パディングが必要
                    new_height = int(width * max_ratio)
                    padding = new_height - height
                    print(f"処理中: {img_file.name}")
                    print(f"  現在のサイズ: {width}x{height} (比率: {current_ratio:.3f})")
                    print(f"  パディングを追加: {padding}px")
                    img = add_horizontal_padding(img, max_ratio, margin_direction)
                    print(f"  新しいサイズ: {img.size[0]}x{img.size[1]} (比率: {max_ratio:.3f})")
                else:
                    print(f"処理中: {img_file.name} - パディング不要")

                # 保存
                output_path = output_folder / img_file.name
                save_kwargs = {}
                if img_file.suffix.lower() in ('.jpg', '.jpeg'):
                    save_kwargs['quality'] = 95
                    save_kwargs['optimize'] = True
                img.save(output_path, **save_kwargs)
                processed_count += 1
                print(f"  → 保存完了: {output_path}")
                print()

            except Exception as e:
                print(f"  → エラー: {img_file.name} - {str(e)}")
                print()

    else:
        # 最大のwidth/height比を計算
        min_ratio_inv = 0
        min_ratio_file = ""
        print("最大のwidth/height比を計算")
        for img_file, img in processed_images:
            width, height = img.size
            ratio = width / height
            if ratio > min_ratio_inv:
                min_ratio_inv = ratio
                min_ratio_file = img_file.name

        print(f"最大アスペクト比の逆数: {min_ratio_inv:.3f} (基準画像: {min_ratio_file})")
        print()

        # 各画像にパディングを追加して保存
        for img_file, img in processed_images:
            try:
                width, height = img.size
                current_ratio = width / height

                if current_ratio < min_ratio_inv - 0.001:  # 浮動小数点の誤差を考慮
                    # パディングが必要
                    new_width = int(height * min_ratio_inv)
                    padding = new_width - width
                    print(f"処理中: {img_file.name}")
                    print(f"  現在のサイズ: {width}x{height} (比率の逆数: {current_ratio:.3f})")
                    print(f"  パディングを追加: {padding}px")
                    img = add_vertical_padding(img, min_ratio_inv, margin_direction)
                    print(f"  新しいサイズ: {img.size[0]}x{img.size[1]} (比率の逆数: {min_ratio_inv:.3f})")
                else:
                    print(f"処理中: {img_file.name} - パディング不要")

                # 保存
                output_path = output_folder / img_file.name
                save_kwargs = {}
                if img_file.suffix.lower() in ('.jpg', '.jpeg'):
                    save_kwargs['quality'] = 95
                    save_kwargs['optimize'] = True
                img.save(output_path, **save_kwargs)
                processed_count += 1
                print(f"  → 保存完了: {output_path}")
                print()

            except Exception as e:
                print(f"  → エラー: {img_file.name} - {str(e)}")
                print()

    print(f"\n処理完了: {processed_count}個の画像を処理しました")
        

def get_padding_direction_config():
    """
     the direction in which padding is added

    Returns:
        value margin_dirction from 1 to 4
    """

    print("\n　余白を追加する位置を選択")
    print("\n cf.) 上方寄せ(下方に余白を追加)  :  1")
    print("\n      右方寄せ(左方に余白を追加)  :  2")
    print("\n      下方寄せ(上方に余白を追加)  :  3")
    print("\n      左方寄せ(右方に余白を追加)  :  4")
    value = input("\n input value between 1 and 4  :  ").strip().lower()

    return(value)

def main():
    """メイン関数"""
    print("画像に余白を追加して位置揃え")

    #　処理対象のフォルダパスを入力
    folder_path = input("\n add folder path in which images exist  :  ").strip()

    # remove quote characters
    folder_path = folder_path.strip('"\'')

    #　choose in which direction padding is added
    margin_direction = get_padding_direction_config()
    
    padding(folder_path, margin_direction)  

if __name__ == "__main__":
    main()