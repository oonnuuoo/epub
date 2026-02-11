from PIL import Image
from pathlib import Path
import os
from datetime import datetime


def write_log(log_file_path, message):
    """
    Write a message to the log file
    
    Args:
        log_file_path: path to the log file
        message: message to write
    """
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def initialize_log(parent_folder_path, proportion, margin_pattern):
    """
    Initialize log file with processing parameters
    
    Args:
        parent_folder_path: parent folder path
        proportion: proportion value for ratio calculation
        margin_pattern: margin pattern choice (1-3)
    
    Returns:
        path to the log file
    """
    # Create log file in the parent folder
    log_file_path = os.path.join(parent_folder_path, "log.txt")
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Write header and parameters to log
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write("=" * 80 + '\n')
        f.write(f"Processing Started: {timestamp}\n")
        f.write("=" * 80 + '\n\n')
        f.write("USER INPUT PARAMETERS:\n")
        f.write(f"  Parent Folder Path: {parent_folder_path}\n")
        f.write(f"  Proportion for Ratio Calculation: {proportion}%\n")
        
        # Decode margin pattern
        pattern_description = {
            "1": "全ページ上方寄せ（下方に余白を追加)",
            "2": "奇数ページ右方寄せ, 偶数ページ左寄せ(見開き左側に奇数ページ)",
            "3": "奇数ページ左方寄せ, 偶数ページ右寄せ(見開き左側に奇数ページ)"
        }
        f.write(f"  Margin Pattern: {margin_pattern} - {pattern_description.get(margin_pattern, 'Unknown')}\n")
        f.write('\n')
    
    return log_file_path


def calculate_ratio(parent_folder_path, proportion, log_file_path=None):
    """
    calculate average ratio of images in folder
    
    Args:
        folder_path: path to folder containing images
        proportion: proportion of images to use for calculation
        log_file_path: path to log file (optional)
    
    Returns:
        average ratio of images in folder
    """
 
    folder_path_odd = parent_folder_path + "/odd/cropped"
    folder_path_even = parent_folder_path + "/even/cropped"

    folder_odd = Path(folder_path_odd)
    files_odd = [f for f in os.listdir(folder_path_odd) if f.endswith('.jpg')]

    folder_even = Path(folder_path_even)
    files_even = [f for f in os.listdir(folder_path_even) if f.endswith('.jpg')]

    aspect_ratios_odd = []
    aspect_ratios_even = []
    for file in files_odd:
        img = Image.open(os.path.join(folder_path_odd, file))
        width, height = img.size
        ratio = height / width
        aspect_ratios_odd.append(ratio)
    
    for file in files_even:
        img = Image.open(os.path.join(folder_path_even, file))
        width, height = img.size
        ratio = height / width
        aspect_ratios_even.append(ratio)

    aspect_ratios = aspect_ratios_odd + aspect_ratios_even
    print(aspect_ratios)


    # omit minimum and maximum quartiles
    aspect_ratios.sort()
    length = len(aspect_ratios)
    print(length)
    start = int(length*(100-proportion)/100/2)
    end = int(length-length*(100-proportion)/100/2)
    print(start, end)

    aspect_ratios = aspect_ratios[start : end]

    # calculate average ratio
    target_ratio = sum(aspect_ratios) / len(aspect_ratios)
    
    # Log ratio calculation results
    if log_file_path:
        write_log(log_file_path, "RATIO CALCULATION:")
        write_log(log_file_path, f"  Total images found: {length}")
        write_log(log_file_path, f"  Odd page images: {len(files_odd)}")
        write_log(log_file_path, f"  Even page images: {len(files_even)}")
        write_log(log_file_path, f"  Images used for calculation (after filtering): {len(aspect_ratios)}")
        write_log(log_file_path, f"  Min ratio used: {min(aspect_ratios):.4f}")
        write_log(log_file_path, f"  Max ratio used: {max(aspect_ratios):.4f}")
        write_log(log_file_path, f"  Target Aspect Ratio (height/width): {target_ratio:.4f}\n")
    
    return target_ratio


def add_horizontal_padding_side(img, target_ratio, margin_direction):
    """
    add white padding on top(margin_direction = 1) or bottom side(margin_direction = 3)

    Args:
        img: PIL Image Object
        target_ratio: objective height/width ratio
        margin_direction: 1 for top-aligned (padding on bottom), 3 for bottom-aligned (padding on top)
    
    Returns:
        PIL Image with padding added
    """
    
    
    width, height = img.size
    if margin_direction == 1:
        # 上方寄せ
        new_height = int(width * target_ratio) # ratio = h / w
        padding_height = new_height - height
        new_img = Image.new('RGB', (width, new_height), (255, 255, 255))
        new_img.paste(img, (0, 0))
    else:
        # 下方寄せ
        new_height = int(width * target_ratio)
        padding_height = new_height - height
        new_img = Image.new('RGB', (width, new_height), (255, 255, 255))
        new_img.paste(img, (0, padding_height))
    return new_img


def add_horizontal_padding_on_top_add_bottom(img, target_ratio):
    """
    add white padding on both top and bottom side
    縦書き文章内の横長の画像を想定

    Args:
        img: PIL Image Object
        target_ratio: objective height/width ratio

    Returns:
        PIL Image with padding added
    """
    width, height = img.size
    new_height = int(width * target_ratio)
    padding_height_top = int((new_height - height) / 2)
    new_img = Image.new('RGB', (width, new_height), (255, 255, 255))
    new_img.paste(img, (0, padding_height_top))

    return new_img


def add_vertical_padding_side(img, target_ratio, margin_direction):
    """
    add white padding on left(margin_direction = 2) or right side(margin_direction = 4)

    Args:
        img: PIL Image Object
        target_ratio: objective width/height ratio
        margin_direction: 2 for left-aligned (padding on right), 4 for right-aligned (padding on left)

    Returns:
        PIL Image with padding 
    """

    width, height = img.size
    if margin_direction == 2:
        # 右方寄せ
        new_width = int(height / target_ratio) # ratio=height/width
        padding_width = new_width - width
        new_img = Image.new('RGB', (new_width, height), (255, 255, 255))
        new_img.paste(img, (padding_width, 0))
    else:
        # 左方寄せ
        new_width = int(height / target_ratio)
        new_img = Image.new('RGB', (new_width, height), (255, 255, 255))
        new_img.paste(img, (0, 0))

    return new_img

def add_vertical_padding_on_left_add_right(img, target_ratio):
    """
    add white padding on both left and right side
    横書き文章内の縦長の画像を想定

    Args:
        img: PIL Image Object
        target_ratio: objective width/height ratio

    Returns:
        PIL Image with padding added
    """
    
    width, height = img.size
    new_width = int(height / target_ratio)
    padding_width_left = (new_width - width) / 2
    new_img = Image.new('RGB', (new_width, height), (255, 255, 255))
    new_img.paste(img, (padding_width_left, 0))

    return new_img


def padding(parent_folder_path, folder_path, margin_direction, target_ratio, log_file_path=None):
    """
    add padding to images in folder and save to new folder
    """

    folder = Path(folder_path)
    parent_folder = Path(parent_folder_path)
    supported_formats = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tiff')
    output_folder = parent_folder / "margin_added"
    output_folder.mkdir(exist_ok=True)

    # target_ratio = calculate_ratio(folder_path)
    print(f"平均アスペクト比（第一四分位-第三四分位）: {target_ratio:.3f}")

    # count images processed
    processed_count = 0
    error_count = 0

    # list temporally saved
    processed_images = []

    target_ratio_plus = target_ratio + 0.001 # なんか誤差？で上下が欠けるので縦に少し余裕を

    # フォルダ直下の画像ファイルを読み込み
    print("画像ファイルを読み込んでいます...")
    if log_file_path:
        write_log(log_file_path, f"PROCESSING FOLDER: {folder_path}")
        write_log(log_file_path, f"  Margin Direction: {margin_direction}")
    
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
                error_count += 1
                if log_file_path:
                    write_log(log_file_path, f"  ERROR loading: {img_file.name} - {str(e)}")

    if not processed_images:
        print("処理対象の画像が見つかりません")
        if log_file_path:
            write_log(log_file_path, "  No images found to process\n")
        return

    print(f"\n{len(processed_images)}個の画像を読み込みました\n")
    if log_file_path:
        write_log(log_file_path, f"  Images loaded successfully: {len(processed_images)}")

    # Convert margin_direction to int for comparison
    # margin_direction = int(margin_direction)

    if margin_direction == 1 or margin_direction == 3:   # 横書き
        # 各画像にパディングを追加して保存
        for img_file, img in processed_images:
            try:
                width, height = img.size
                current_ratio = height / width

                # if current_ratio < target_ratio - 0.001:  # 浮動小数点の誤差を考慮 ratio = h / w
                if height < target_ratio * width:
                    # 目標アスペクト比よりも横長の場合
                    img = add_horizontal_padding_side(img, target_ratio, margin_direction)
                    # img = add_horizontal_padding_side(img, target_ratio_plus, margin_direction)
                    print(f"  新しいサイズ: {img.size[0]}x{img.size[1]} (比率: {target_ratio:.3f})")
                else:
                    # 目標アスペクト比よりも縦長の場合
                    img = add_vertical_padding_on_left_add_right(img, target_ratio)
                    # img = add_vertical_padding_on_left_add_right(img, target_ratio_plus)
                    print(f"  新しいサイズ: {img.size[0]}x{img.size[1]} (比率: {target_ratio:.3f})")

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
                error_count += 1
                if log_file_path:
                    write_log(log_file_path, f"  ERROR processing: {img_file.name} - {str(e)}")
                print()

    else:   # 縦書き
        # 各画像にパディングを追加して保存
        for img_file, img in processed_images:
            try:
                width, height = img.size
                current_ratio = height / width

                if current_ratio < target_ratio - 0.001:  # 浮動小数点の誤差を考慮
                    # 縦書きにおいて横長の画像
                    
                    # img = add_horizontal_padding_on_top_add_bottom(img, target_ratio)
                    # img = add_horizontal_padding_on_top_add_bottom(img, target_ratio_plus)
                    img = add_horizontal_padding_side(img, target_ratio, 1)  # 縦書きにおいて上部寄せ(目次ページなど想定)
                    print(f"  新しいサイズ: {img.size[0]}x{img.size[1]} (比率: {target_ratio:.3f})")
                else:
                    # 縦書きにおいて縦長の画像
                    
                    img = add_vertical_padding_side(img, target_ratio, margin_direction)
                    # img = add_vertical_padding_side(img, target_ratio_plus, margin_direction)
                    print(f"  新しいサイズ: {img.size[0]}x{img.size[1]} (比率: {target_ratio:.3f})")
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
                error_count += 1
                if log_file_path:
                    write_log(log_file_path, f"  ERROR processing: {img_file.name} - {str(e)}")
                print()

    print(f"\n処理完了: {processed_count}個の画像を処理しました")
    if log_file_path:
        write_log(log_file_path, f"  Successfully processed: {processed_count} images")
        write_log(log_file_path, f"  Errors encountered: {error_count}\n")
        

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

def get_margin_pattern_config():
    
    print("\n　余白を追加するパターンを選択")
    print("\n cf.) 全ページ上方寄せ（下方に余白を追加)                         :  1")
    print("\n      奇数ページ右方寄せ, 偶数ページ左寄せ(見開き左側に奇数ページ)  :  2")
    print("\n      奇数ページ左方寄せ, 偶数ページ右寄せ(見開き左側に奇数ページ)  :  3")
    value = input("\n input value between 1 and 3  :  ").strip().lower()

    return(value)

def padding_branch(parent_folder_path, margin_pattern, target_ratio, log_file_path=None):

    folder_path_odd = parent_folder_path + "/odd/cropped"
    folder_path_even = parent_folder_path + "/even/cropped"

    # folder_path_odd = parent_folder_path + "/odd/cropped/resized/cropped"
    # folder_path_even = parent_folder_path + "/even/cropped/resized/cropped"

    folder_path_odd_sub = parent_folder_path + "/odd/ex/resized"
    folder_path_even_sub = parent_folder_path + "/even/ex/resized"

    if margin_pattern == "1":
        # 横書き, 全ページ上方寄せ
        padding(parent_folder_path, folder_path_odd, "1", target_ratio, log_file_path)
        padding(parent_folder_path, folder_path_even, "1", target_ratio, log_file_path)
        padding(parent_folder_path, folder_path_odd_sub, "1", target_ratio, log_file_path)
        padding(parent_folder_path, folder_path_even_sub, "1", target_ratio, log_file_path)

    elif margin_pattern == "2":
        # 縦書き, 奇数ページ右方寄せ, 偶数ページ左寄せ 
        # [3|2] ←こんなイメージ
        padding(parent_folder_path, folder_path_odd, "2", target_ratio, log_file_path)
        padding(parent_folder_path, folder_path_even, "4", target_ratio, log_file_path)
        padding(parent_folder_path, folder_path_odd_sub, "2", target_ratio, log_file_path)
        padding(parent_folder_path, folder_path_even_sub, "4", target_ratio, log_file_path)

    elif margin_pattern == "3":
        # 縦書き, 奇数ページ左方寄せ, 偶数ページ右寄せ
        # [2|1] ←こんなイメージ
        padding(parent_folder_path, folder_path_odd, "4", target_ratio, log_file_path)
        padding(parent_folder_path, folder_path_even, "2", target_ratio, log_file_path)
        padding(parent_folder_path, folder_path_odd_sub, "4", target_ratio, log_file_path)
        padding(parent_folder_path, folder_path_even_sub, "2", target_ratio, log_file_path)


    else:
        print("Invalid margin pattern")
        if log_file_path:
            write_log(log_file_path, "ERROR: Invalid margin pattern selected")
        return

def main():
    """メイン関数"""
    print("画像に余白を追加して位置揃え")

    #　処理対象のフォルダの親ディレクトリを入力
    parent_folder_path = input("\n add parent folder path in which images exist  :  ").strip()

    # input the proportion of the images for calculating the target_ratio
    proportion = int(input("\n add proportion of the images for calculating the target_ratio(1-100)  :  ").strip())

    # remove quote characters
    parent_folder_path = parent_folder_path.strip('"\'')

    #　choose in which direction padding is added

    # margin_direction = get_padding_direction_config()
    margin_pattern = get_margin_pattern_config()

    # Initialize log file
    log_file_path = initialize_log(parent_folder_path, proportion, margin_pattern)
    print(f"\nLog file created at: {log_file_path}\n")

    # Calculate target ratio with logging
    target_ratio = calculate_ratio(parent_folder_path, proportion, log_file_path)
    
    # Process images with logging
    padding_branch(parent_folder_path, margin_pattern, target_ratio, log_file_path)
    
    # Write completion timestamp to log
    end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_log(log_file_path, "=" * 80)
    write_log(log_file_path, f"Processing Completed: {end_timestamp}")
    write_log(log_file_path, "=" * 80)
    
    print(f"\nProcessing complete. Check log file at: {log_file_path}")

if __name__ == "__main__":
    main()