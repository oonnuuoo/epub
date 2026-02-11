import os
import numpy as np
from PIL import Image
from pathlib import Path
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


def initialize_log(folder_path, crop_margins, page_num_config, unify_aspect_ratio):
    """
    Initialize log file with processing parameters
    
    Args:
        folder_path: folder path containing images
        crop_margins: tuple of (top, bottom, left, right) crop percentages
        page_num_config: page number removal configuration
        unify_aspect_ratio: whether to unify aspect ratios
    
    Returns:
        path to the log file
    """
    # Create log file in the processed folder
    parent_folder = Path(folder_path).parent
    log_file_path = parent_folder / "image_cropper_log.txt"
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Write header and parameters to log
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write("=" * 80 + '\n')
        f.write(f"Image Cropping Process Started: {timestamp}\n")
        f.write("=" * 80 + '\n\n')
        f.write("USER INPUT PARAMETERS:\n")
        f.write(f"  Source Folder: {folder_path}\n")
        f.write(f"  Output Folder: {folder_path}/cropped\n\n")
        
        # Crop margins
        f.write("INITIAL CROP SETTINGS:\n")
        top, bottom, left, right = crop_margins
        f.write(f"  Top: {top}%\n")
        f.write(f"  Bottom: {bottom}%\n")
        f.write(f"  Left: {left}%\n")
        f.write(f"  Right: {right}%\n\n")
        
        # Page number removal config
        f.write("PAGE NUMBER REMOVAL SETTINGS:\n")
        if page_num_config['remove']:
            f.write(f"  Enabled: Yes\n")
            position_map = {
                'top': '上部（ヘッダー）',
                'bottom': '下部（フッター）',
                'both': '上下部（ヘッダー+フッター）'
            }
            f.write(f"  Position: {page_num_config['position']} - {position_map.get(page_num_config['position'], 'Unknown')}\n")
            f.write(f"  Crop Percentage: {page_num_config['crop_percentage']}%\n")
        else:
            f.write(f"  Enabled: No\n")
        f.write('\n')
        
        # Aspect ratio unification
        f.write("ASPECT RATIO UNIFICATION:\n")
        f.write(f"  Enabled: {'Yes' if unify_aspect_ratio else 'No'}\n")
        if unify_aspect_ratio:
            f.write("  Method: Add white padding to bottom to match tallest image\n")
        f.write('\n')
        
        # Processing steps
        f.write("PROCESSING STEPS:\n")
        f.write(f"  1. Initial crop (Top:{top}% Bottom:{bottom}% Left:{left}% Right:{right}%)\n")
        f.write("  2. Remove white borders (threshold: 200, noise skip: 5px)\n")
        if page_num_config['remove']:
            f.write(f"  3. Crop {page_num_config['position']} by {page_num_config['crop_percentage']}% to remove page numbers\n")
            f.write(f"  4. Remove white margins from {page_num_config['position']} edge\n")
            f.write("  5. Add 5px white margin on all sides\n")
            if unify_aspect_ratio:
                f.write("  6. Unify aspect ratios (add bottom padding)\n")
        else:
            f.write("  3. Add 5px white margin on all sides\n")
            if unify_aspect_ratio:
                f.write("  4. Unify aspect ratios (add bottom padding)\n")
        f.write('\n')
    
    return log_file_path


def crop_with_custom_margins(img, top=5, bottom=5, left=5, right=5):
    """
    画像の上下左右から指定パーセンテージをトリミング
    
    Args:
        img: PIL Image object
        top: 上部からトリミングするパーセンテージ（デフォルト5%）
        bottom: 下部からトリミングするパーセンテージ（デフォルト5%）
        left: 左側からトリミングするパーセンテージ（デフォルト5%）
        right: 右側からトリミングするパーセンテージ（デフォルト5%）
    
    Returns:
        トリミング後のPIL Image
    """
    width, height = img.size
    
    # トリミングする量を計算
    crop_top = int(height * top / 100)
    crop_bottom = int(height * bottom / 100)
    crop_left = int(width * left / 100)
    crop_right = int(width * right / 100)
    
    if crop_left - 20 <= 0:
        crop_left = 20
    if crop_top - 10 <= 0:
        crop_top = 10
    if width - crop_right + 20 > width:
        crop_right = 20
    if height - crop_bottom + 10 > height:
        crop_bottom = 10
    # トリミング領域を定義 (left, top, right, bottom)

    crop_box = (
        crop_left - 20,
        crop_top - 10,
        width - crop_right + 20,
        height - crop_bottom + 10
    )
    
    return img.crop(crop_box)

def crop_top(img, percentage=5):
    """
    画像の上部から指定パーセンテージをトリミング
    
    Args:
        img: PIL Image object
        percentage: 上部からトリミングするパーセンテージ（デフォルト5%）
    
    Returns:
        トリミング後のPIL Image
    """
    width, height = img.size
    
    # 上部からトリミングする量を計算
    crop_height = int(height * percentage / 100)
    
    # トリミング領域を定義 (left, top, right, bottom)
    crop_box = (
        0,
        crop_height,
        width,
        height
    )
    
    return img.crop(crop_box)

def crop_bottom(img, percentage=5):
    """
    画像の下部から指定パーセンテージをトリミング
    
    Args:
        img: PIL Image object
        percentage: 下部からトリミングするパーセンテージ（デフォルト5%）
    
    Returns:
        トリミング後のPIL Image
    """
    width, height = img.size
    
    # 下部からトリミングする量を計算
    crop_height = int(height * percentage / 100)
    
    # トリミング領域を定義 (left, top, right, bottom)
    crop_box = (
        0,
        0,
        width,
        height - crop_height
    )
    
    return img.crop(crop_box)

def remove_white_borders(img, threshold=200, noise_threshold=5):
    """
    輝度値を基準に白い余白を除去（ノイズスキップ機能付き）
    
    Args:
        img: PIL Image object
        threshold: 白と判定する輝度値の閾値（デフォルト240）
        noise_threshold: ノイズと判定するピクセル数の閾値（デフォルト5）
    
    Returns:
        余白除去後のPIL Image
    """
    # グレースケールに変換
    gray = img.convert('L')
    
    # NumPy配列に変換
    img_array = np.array(gray)
    height, width = img_array.shape
    
    # 白でない部分を検出（閾値より小さい部分）
    non_white_pixels = img_array < threshold
    
    # 上端の境界を見つける（ノイズスキップ付き）
    row_min = 0
    for y in range(height):
        # この行の非白色ピクセル数をカウント
        non_white_count = np.sum(non_white_pixels[y, :])
        if non_white_count > noise_threshold:
            row_min = y
            break
        elif non_white_count > 0 and non_white_count <= noise_threshold:
            # ノイズの場合、5pxスキップ
            skip_to = min(y + 5, height - 1)
            for check_y in range(y + 1, skip_to + 1):
                if check_y < height and np.sum(non_white_pixels[check_y, :]) > noise_threshold:
                    row_min = check_y
                    break
            else:
                continue
            break
    
    # 下端の境界を見つける（ノイズスキップ付き）
    row_max = height - 1
    for y in range(height - 1, -1, -1):
        # この行の非白色ピクセル数をカウント
        non_white_count = np.sum(non_white_pixels[y, :])
        if non_white_count > noise_threshold:
            row_max = y
            break
        elif non_white_count > 0 and non_white_count <= noise_threshold:
            # ノイズの場合、5pxスキップ
            skip_to = max(y - 5, 0)
            for check_y in range(y - 1, skip_to - 1, -1):
                if check_y >= 0 and np.sum(non_white_pixels[check_y, :]) > noise_threshold:
                    row_max = check_y
                    break
            else:
                continue
            break
    
    # 左端の境界を見つける（ノイズスキップ付き）
    col_min = 0
    for x in range(width):
        # この列の非白色ピクセル数をカウント
        non_white_count = np.sum(non_white_pixels[:, x])
        if non_white_count > noise_threshold:
            col_min = x
            break
        elif non_white_count > 0 and non_white_count <= noise_threshold:
            # ノイズの場合、5pxスキップ
            skip_to = min(x + 5, width - 1)
            for check_x in range(x + 1, skip_to + 1):
                if check_x < width and np.sum(non_white_pixels[:, check_x]) > noise_threshold:
                    col_min = check_x
                    break
            else:
                continue
            break
    
    # 右端の境界を見つける（ノイズスキップ付き）
    col_max = width - 1
    for x in range(width - 1, -1, -1):
        # この列の非白色ピクセル数をカウント
        non_white_count = np.sum(non_white_pixels[:, x])
        if non_white_count > noise_threshold:
            col_max = x
            break
        elif non_white_count > 0 and non_white_count <= noise_threshold:
            # ノイズの場合、5pxスキップ
            skip_to = max(x - 5, 0)
            for check_x in range(x - 1, skip_to - 1, -1):
                if check_x >= 0 and np.sum(non_white_pixels[:, check_x]) > noise_threshold:
                    col_max = check_x
                    break
            else:
                continue
            break
    
    # 有効な境界が見つかった場合のみトリミング
    if row_min <= row_max and col_min <= col_max:
        return img.crop((col_min, row_min, col_max + 1, row_max + 1))
    else:
        # 全体が白い場合は元の画像を返す
        return img

def remove_top_white_margin(img, threshold=200, noise_threshold=5):
    """
    画像上端の白いマージンのみを除去（ノイズスキップ機能付き）
    
    Args:
        img: PIL Image object
        threshold: 白と判定する輝度値の閾値（デフォルト240）
        noise_threshold: ノイズと判定するピクセル数の閾値（デフォルト5）
    
    Returns:
        上端の白いマージンを除去したPIL Image
    """
    # グレースケールに変換
    gray = img.convert('L')
    
    # NumPy配列に変換
    img_array = np.array(gray)
    
    height, width = img_array.shape
    
    # 上から下に向かって、白でない行を探す（ノイズスキップ付き）
    for row in range(height):
        # この行の非白色ピクセル数をカウント
        non_white_count = np.sum(img_array[row] < threshold)
        
        if non_white_count > noise_threshold:
            # 十分な非白色ピクセルがある場合
            if row > 0:
                return img.crop((0, row, width, height))
            else:
                return img
        elif non_white_count > 0 and non_white_count <= noise_threshold:
            # ノイズの場合、5pxスキップして次の有効な行を探す
            skip_to = min(row + 5, height - 1)
            for check_row in range(row + 1, skip_to + 1):
                if check_row < height:
                    check_count = np.sum(img_array[check_row] < threshold)
                    if check_count > noise_threshold:
                        return img.crop((0, check_row, width, height))
    
    # すべて白の場合は元の画像を返す
    return img

def remove_bottom_white_margin(img, threshold=200, noise_threshold=5):
    """
    画像下端の白いマージンのみを除去（ノイズスキップ機能付き）
    
    Args:
        img: PIL Image object
        threshold: 白と判定する輝度値の閾値（デフォルト240）
        noise_threshold: ノイズと判定するピクセル数の閾値（デフォルト5）
    
    Returns:
        下端の白いマージンを除去したPIL Image
    """
    # グレースケールに変換
    gray = img.convert('L')
    
    # NumPy配列に変換
    img_array = np.array(gray)
    
    height, width = img_array.shape
    
    # 下から上に向かって、白でない行を探す（ノイズスキップ付き）
    for row in range(height - 1, -1, -1):
        # この行の非白色ピクセル数をカウント
        non_white_count = np.sum(img_array[row] < threshold)
        
        if non_white_count > noise_threshold:
            # 十分な非白色ピクセルがある場合
            if row < height - 1:
                return img.crop((0, 0, width, row + 1))
            else:
                return img
        elif non_white_count > 0 and non_white_count <= noise_threshold:
            # ノイズの場合、5pxスキップして次の有効な行を探す
            skip_to = max(row - 5, 0)
            for check_row in range(row - 1, skip_to - 1, -1):
                if check_row >= 0:
                    check_count = np.sum(img_array[check_row] < threshold)
                    if check_count > noise_threshold:
                        return img.crop((0, 0, width, check_row + 1))
    
    # すべて白の場合は元の画像を返す
    return img

def add_white_margin(img, margin_px=20):
    """
    画像の周囲に白い余白を追加
    
    Args:
        img: PIL Image object
        margin_px: 追加する余白のピクセル数（デフォルト20px）
    
    Returns:
        余白追加後のPIL Image
    """
    width, height = img.size
    
    # 新しい画像サイズ
    new_width = width + 2 * margin_px
    new_height = height + 2 * 5  # 上下には5pxの余白を追加
    
    # 白背景の新しい画像を作成
    new_img = Image.new('RGB', (new_width, new_height), (255, 255, 255))
    
    # 元の画像を中央に配置
    new_img.paste(img, (margin_px, 5))
    
    return new_img

def add_bottom_padding(img, target_ratio):
    """
    画像の下部に白いパディングを追加してアスペクト比を統一
    
    Args:
        img: PIL Image object
        target_ratio: 目標のアスペクト比（height/width）
    
    Returns:
        パディング追加後のPIL Image
    """
    width, height = img.size
    
    # 目標の高さを計算
    new_height = int(width * target_ratio)
    
    # 新しい画像を作成（白背景）
    new_img = Image.new('RGB', (width, new_height), (255, 255, 255))
    
    # 元の画像を上部に配置
    new_img.paste(img, (0, 0))
    
    return new_img

def process_images(folder_path, crop_margins, page_num_config, unify_aspect_ratio, log_file_path=None):
    """
    フォルダ内の画像を一括処理
    
    Args:
        folder_path: 処理対象の画像が含まれるフォルダのパス
        crop_margins: (top, bottom, left, right) のタプル
        page_num_config: ページ番号除去の設定ディクショナリ
        unify_aspect_ratio: アスペクト比を統一するかどうか
        log_file_path: path to log file (optional)
    """
    folder = Path(folder_path)
    
    # 対応する画像フォーマット
    supported_formats = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tiff')
    
    # 出力フォルダの作成
    output_folder = folder / "cropped"
    output_folder.mkdir(exist_ok=True)
    
    # カウンター
    processed_count = 0
    error_count = 0
    
    # 処理済み画像のリスト（アスペクト比統一用）
    processed_images = []
    
    # トリミング設定の展開
    top, bottom, left, right = crop_margins
    
    print("=" * 60)
    print(f"処理対象フォルダ: {folder}")
    print(f"出力先: {output_folder}")
    print("=" * 60)
    
    if log_file_path:
        write_log(log_file_path, "IMAGE PROCESSING STARTED\n")
        write_log(log_file_path, f"Source folder: {folder}")
        write_log(log_file_path, f"Output folder: {output_folder}\n")
    
    # 画像の読み込みと初期処理
    print("\n画像ファイルを処理中...\n")
    
    for img_file in folder.iterdir():
        if img_file.is_file() and img_file.suffix.lower() in supported_formats:
            try:
                # 画像を読み込み
                img = Image.open(img_file)
                original_size = img.size
                print(f"処理中: {img_file.name}")
                print(f"  元のサイズ: {img.size[0]}x{img.size[1]}")
                
                # 1. カスタムマージンでトリミング
                img = crop_with_custom_margins(img, top, bottom, left, right)
                print(f"  トリミング後: {img.size[0]}x{img.size[1]}")
                
                # 2. 白い余白を除去
                img = remove_white_borders(img)
                print(f"  余白除去後: {img.size[0]}x{img.size[1]}")
                
                # 3. ページ番号除去処理（オプション）
                if page_num_config['remove']:
                    if page_num_config['position'] == 'bottom':
                        # 下部をトリミング
                        img = crop_bottom(img, page_num_config['crop_percentage'])
                        # 下端の白いマージンを除去
                        img = remove_bottom_white_margin(img)
                        print(f"  ページ番号除去後: {img.size[0]}x{img.size[1]}")
                    elif page_num_config['position'] == 'top':
                        # 上部をトリミング
                        img = crop_top(img, page_num_config['crop_percentage'])
                        # 上端の白いマージンを除去
                        img = remove_top_white_margin(img)
                        print(f"  ページ番号除去後: {img.size[0]}x{img.size[1]}")
                    else:
                        # 上下部をトリミング
                        img = crop_bottom(img, page_num_config['crop_percentage'])
                        img = crop_top(img, page_num_config['crop_percentage'])
                        # 上下端の白いマージンを除去
                        img = remove_bottom_white_margin(img)
                        img = remove_top_white_margin(img)
                        print(f"  ページ番号除去後: {img.size[0]}x{img.size[1]}")
                
                # 4. 20pxの白い余白を追加
                img = add_white_margin(img, 20)
                print(f"  余白追加後: {img.size[0]}x{img.size[1]}")
                
                # アスペクト比統一が有効な場合は一時保存
                if unify_aspect_ratio:
                    processed_images.append((img_file, img))
                else:
                    # 保存
                    output_path = output_folder / img_file.name
                    save_kwargs = {}
                    if img_file.suffix.lower() in ('.jpg', '.jpeg'):
                        save_kwargs['quality'] = 95
                        save_kwargs['optimize'] = True
                    img.save(output_path, **save_kwargs)
                    processed_count += 1
                    print(f"  → 保存完了: {output_path}")
                
                if log_file_path:
                    write_log(log_file_path, f"Successfully processed: {img_file.name}")
                    write_log(log_file_path, f"  Original size: {original_size[0]}x{original_size[1]}")
                    write_log(log_file_path, f"  Final size: {img.size[0]}x{img.size[1]}")
                
                print()
                
            except Exception as e:
                print(f"  → エラー: {img_file.name} - {str(e)}")
                error_count += 1
                if log_file_path:
                    write_log(log_file_path, f"ERROR processing {img_file.name}: {str(e)}")
                print()
    
    # アスペクト比の統一処理
    if unify_aspect_ratio and processed_images:
        print("=" * 60)
        print("アスペクト比統一処理を開始...")
        
        if log_file_path:
            write_log(log_file_path, "\nASPECT RATIO UNIFICATION:")
        
        # 最大のheight/width比を計算
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
        
        if log_file_path:
            write_log(log_file_path, f"  Maximum aspect ratio: {max_ratio:.4f}")
            write_log(log_file_path, f"  Reference image: {max_ratio_file}\n")
        
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
                    print(f"  下部に{padding}pxのパディングを追加")
                    img = add_bottom_padding(img, max_ratio)
                    print(f"  新しいサイズ: {img.size[0]}x{img.size[1]} (比率: {max_ratio:.3f})")
                    
                    if log_file_path:
                        write_log(log_file_path, f"  Added padding to {img_file.name}: {padding}px")
                else:
                    print(f"処理中: {img_file.name} - パディング不要")
                    if log_file_path:
                        write_log(log_file_path, f"  No padding needed for {img_file.name}")
                
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
                    write_log(log_file_path, f"  ERROR during aspect ratio unification for {img_file.name}: {str(e)}")
                print()
    
    print("=" * 60)
    print(f"処理完了: {processed_count}個の画像を処理しました。")
    if error_count > 0:
        print(f"エラー: {error_count}個の画像でエラーが発生しました。")
    print(f"出力先: {output_folder}")
    
    if log_file_path:
        write_log(log_file_path, f"\nPROCESSING SUMMARY:")
        write_log(log_file_path, f"  Successfully processed: {processed_count} images")
        write_log(log_file_path, f"  Errors encountered: {error_count} images")
        write_log(log_file_path, f"  Output folder: {output_folder}\n")

def get_float_input(prompt, default_value, min_val=0, max_val=100):
    """
    浮動小数点数の入力を取得
    
    Args:
        prompt: 入力プロンプト
        default_value: デフォルト値
        min_val: 最小値
        max_val: 最大値
    
    Returns:
        入力された値またはデフォルト値
    """
    while True:
        user_input = input(prompt).strip()
        if user_input == '':
            return default_value
        try:
            value = float(user_input)
            if min_val <= value <= max_val:
                return value
            else:
                print(f"{min_val}から{max_val}の間の数値を入力してください。")
        except ValueError:
            print("有効な数値を入力してください。")

def get_crop_margins():
    """
    トリミングマージンを対話的に取得
    
    Returns:
        (top, bottom, left, right) のタプル
    """
    print("\n【初期トリミング設定】")
    print("画像の各辺からトリミングする割合を設定します。")
    print("（Enterキーでデフォルト値を使用）")
    
    # 簡単設定オプション
    print("\n簡単設定:")
    print("  1. すべて同じ値（一括設定）")
    print("  2. 個別に設定")
    
    choice = input("選択してください (1/2, デフォルト: 1): ").strip()
    
    if choice != '2':
        # 一括設定
        all_value = get_float_input(
            "すべての辺のトリミング率を入力（%, デフォルト: 5.0）: ", 5.0
        )
        return (all_value, all_value, all_value, all_value)
    else:
        # 個別設定
        print("\n各辺のトリミング率を個別に設定します:")
        top = get_float_input("  上部のトリミング率（%, デフォルト: 5.0）: ", 5.0)
        bottom = get_float_input("  下部のトリミング率（%, デフォルト: 5.0）: ", 5.0)
        left = get_float_input("  左側のトリミング率（%, デフォルト: 5.0）: ", 5.0)
        right = get_float_input("  右側のトリミング率（%, デフォルト: 5.0）: ", 5.0)
        
        return (top, bottom, left, right)

def get_page_number_config():
    """
    ページ番号除去の設定を取得
    
    Returns:
        設定のディクショナリ
    """
    config = {
        'remove': False,
        'position': 'bottom',
        'crop_percentage': 5.0
    }
    
    print("\n【ページ番号除去設定】")
    print("処理内容: 余白除去後、指定位置をトリミングしてから")
    print("         端の白いマージンを再度除去します")
    
    remove_input = input("\nページ番号除去処理を行いますか？ (y/n, デフォルト: y): ").strip().lower()
    
    if remove_input == 'n':
        print("→ ページ番号除去: 無効")
        return config
    
    config['remove'] = True
    print("→ ページ番号除去: 有効")
    
    # ページ番号の位置を選択
    print("\nページ番号の位置を選択してください:")
    print("  1. 下部（フッター）")
    print("  2. 上部（ヘッダー）")
    print("  3. 上下部 (ヘッダー+フッター)")
    
    position_input = input("選択してください (1/2/3, デフォルト: 1): ").strip()
    
    if position_input == '2':
        config['position'] = 'top'
        print("→ ページ番号位置: 上部")
        
        # 上部トリミング率を設定
        print("\n上部からトリミングする割合を設定します。")
        print("（ページ番号の位置に応じて調整してください）")
        print("  - 3%: ページ番号が非常に上端に近い場合")
        print("  - 5%: 標準的な位置の場合（デフォルト）")
        print("  - 7-10%: ページ番号が下めに配置されている場合")
        
        config['crop_percentage'] = get_float_input(
            "上部トリミング率を入力してください（%, デフォルト: 5.0）: ", 5.0
        )
    elif position_input == '1':
        config['position'] = 'bottom'
        print("→ ページ番号位置: 下部")
        
        # 下部トリミング率を設定
        print("\n下部からトリミングする割合を設定します。")
        print("（ページ番号の位置に応じて調整してください）")
        print("  - 3%: ページ番号が非常に下端に近い場合")
        print("  - 5%: 標準的な位置の場合（デフォルト）")
        print("  - 7-10%: ページ番号が上めに配置されている場合")
        
        config['crop_percentage'] = get_float_input(
            "下部トリミング率を入力してください（%, デフォルト: 5.0）: ", 5.0
        )

    else:
        config['position'] = 'both'
        print("→ ページ番号位置: 上下部")
        
        # 下部トリミング率を設定
        print("\n下部からトリミングする割合を設定します。")
        print("（ページ番号の位置に応じて調整してください）")
        print("  - 3%: ページ番号が非常に下端に近い場合")
        print("  - 5%: 標準的な位置の場合（デフォルト）")
        print("  - 7-10%: ページ番号が上めに配置されている場合")
        
        config['crop_percentage'] = get_float_input(
            "上下部トリミング率を入力してください（%, デフォルト: 5.0）: ", 5.0
        )

    
    print(f"→ トリミング率: {config['crop_percentage']}%")
    
    return config

def get_aspect_ratio_config():
    """
    アスペクト比統一の設定を取得
    
    Returns:
        統一するかどうかのbool値
    """
    print("\n【アスペクト比統一設定】")
    print("処理内容: すべての画像のアスペクト比（高さ/幅）を統一します。")
    print("         最も縦長の画像に合わせて、他の画像の下部に白いパディングを追加します。")
    
    unify_input = input("\nアスペクト比を統一しますか？ (y/n, デフォルト: n): ").strip().lower()
    
    if unify_input == 'y':
        print("→ アスペクト比統一: 有効")
        return True
    else:
        print("→ アスペクト比統一: 無効")
        return False

def main():
    """メイン関数"""
    print("=" * 60)
    print("画像トリミング・余白除去ツール")
    print("（ノイズスキップ機能・アスペクト比統一機能付き）")
    print("=" * 60)
    
    # 処理対象のフォルダパスを入力
    folder_path = input("\n画像が含まれるフォルダのパスを入力してください: ").strip()
    
    # クォートを除去（ドラッグ&ドロップ時に付く場合がある）
    folder_path = folder_path.strip('"\'')
    
    # トリミングマージンを取得
    crop_margins = get_crop_margins()
    top, bottom, left, right = crop_margins
    print(f"\n→ トリミング設定: 上:{top}% 下:{bottom}% 左:{left}% 右:{right}%")
    
    # ページ番号除去の設定を取得
    page_num_config = get_page_number_config()
    
    # アスペクト比統一の設定を取得
    unify_aspect_ratio = get_aspect_ratio_config()
    
    # Initialize log file
    log_file_path = initialize_log(folder_path, crop_margins, page_num_config, unify_aspect_ratio)
    print(f"\nLog file created at: {log_file_path}\n")
    
    print("\n" + "=" * 60)
    print("処理を開始します...")
    print("処理順序:")
    print(f"1. トリミング (上:{top}% 下:{bottom}% 左:{left}% 右:{right}%)")
    print("2. 白い余白を除去（5px以下のノイズはスキップ）")
    
    step_num = 3
    if page_num_config['remove']:
        print(f"3. {page_num_config['position']}{page_num_config['crop_percentage']}%トリミング")
        print(f"4. {page_num_config['position']}端の白いマージンを除去（5px以下のノイズはスキップ）")
        print("5. 上下左右に5pxの白い余白を追加")
        step_num = 6
    else:
        print("3. 上下左右に5pxの白い余白を追加")
        step_num = 4
    
    if unify_aspect_ratio:
        print(f"{step_num}. すべての画像のアスペクト比を統一（下部にパディング追加）")
    
    print("=" * 60 + "\n")
    
    # 画像処理を実行
    process_images(folder_path, crop_margins, page_num_config, unify_aspect_ratio, log_file_path)
    
    # Write completion timestamp to log
    end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_log(log_file_path, "=" * 80)
    write_log(log_file_path, f"Processing Completed: {end_timestamp}")
    write_log(log_file_path, "=" * 80)
    
    print(f"\nProcessing complete. Check log file at: {log_file_path}")

if __name__ == "__main__":
    main()