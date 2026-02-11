import os
import numpy as np
from PIL import Image
from pathlib import Path

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
    
    # トリミング領域を定義 (left, top, right, bottom)
    crop_box = (
        crop_left - 5,
        crop_top - 5,
        width - crop_right + 5,
        height - crop_bottom + 5
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

def remove_white_borders(img, threshold=200):
    """
    輝度値を基準に白い余白を除去
    
    Args:
        img: PIL Image object
        threshold: 白と判定する輝度値の閾値（デフォルト240）
    
    Returns:
        余白除去後のPIL Image
    """
    # グレースケールに変換
    gray = img.convert('L')
    
    # NumPy配列に変換
    img_array = np.array(gray)
    
    # 白でない部分を検出（閾値より小さい部分）
    non_white_pixels = img_array < threshold
    
    # 白でない部分の境界を見つける
    rows = np.any(non_white_pixels, axis=1)
    cols = np.any(non_white_pixels, axis=0)
    
    # 境界インデックスを取得
    if rows.any() and cols.any():
        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]
        
        # 元の画像（カラー）をトリミング
        return img.crop((col_min, row_min, col_max + 1, row_max + 1))
    else:
        # 全体が白い場合は元の画像を返す
        return img

def remove_top_white_margin(img, threshold=200):
    """
    画像上端の白いマージンのみを除去
    
    Args:
        img: PIL Image object
        threshold: 白と判定する輝度値の閾値（デフォルト240）
    
    Returns:
        上端の白いマージンを除去したPIL Image
    """
    # グレースケールに変換
    gray = img.convert('L')
    
    # NumPy配列に変換
    img_array = np.array(gray)
    
    height, width = img_array.shape
    
    # 上から下に向かって、白でない行を探す
    for row in range(height):
        # この行に白でないピクセルがあるか確認
        if np.any(img_array[row] < threshold):
            # 白でない行が見つかったら、ここからをトリミング
            if row > 0:
                return img.crop((0, row, width, height))
            else:
                return img
    
    # すべて白の場合は元の画像を返す
    return img

def remove_bottom_white_margin(img, threshold=200):
    """
    画像下端の白いマージンのみを除去
    
    Args:
        img: PIL Image object
        threshold: 白と判定する輝度値の閾値（デフォルト240）
    
    Returns:
        下端の白いマージンを除去したPIL Image
    """
    # グレースケールに変換
    gray = img.convert('L')
    
    # NumPy配列に変換
    img_array = np.array(gray)
    
    height, width = img_array.shape
    
    # 下から上に向かって、白でない行を探す
    for row in range(height - 1, -1, -1):
        # この行に白でないピクセルがあるか確認
        if np.any(img_array[row] < threshold):
            # 白でない行が見つかったら、ここまでをトリミング
            if row < height - 1:
                return img.crop((0, 0, width, row + 1))
            else:
                return img
    
    # すべて白の場合は元の画像を返す
    return img

def add_white_border(img, border_width=5):
    """
    画像の上下左右に白い余白を追加
    
    Args:
        img: PIL Image object
        border_width: 追加する余白の幅（ピクセル）（デフォルト5）
    
    Returns:
        余白を追加したPIL Image
    """
    width, height = img.size
    
    # 新しい画像サイズを計算
    new_width = width + (border_width * 2)
    new_height = height + (border_width * 2)
    
    # 白い背景の新しい画像を作成
    new_img = Image.new('RGB', (new_width, new_height), (255, 255, 255))
    
    # 元の画像を中央に配置
    new_img.paste(img, (border_width, border_width))
    
    return new_img

def process_images(folder_path, crop_margins, page_num_config,
                  supported_formats=('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
    """
    指定フォルダ内の画像を処理
    
    Args:
        folder_path: 処理対象のフォルダパス
        crop_margins: (top, bottom, left, right) のタプル - トリミング率
        page_num_config: ページ番号設定のディクショナリ
        supported_formats: サポートする画像形式のタプル
    """
    # パスをPathオブジェクトに変換
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"エラー: フォルダ '{folder_path}' が存在しません。")
        return
    
    # croppedフォルダを作成
    output_folder = folder / "cropped"
    output_folder.mkdir(exist_ok=True)
    
    # 処理した画像数をカウント
    processed_count = 0
    
    top, bottom, left, right = crop_margins
    remove_page_num = page_num_config['remove']
    page_position = page_num_config['position']
    page_crop_percentage = page_num_config['crop_percentage']
    
    # フォルダ直下の画像ファイルを処理
    for img_file in folder.iterdir():
        # ファイルかつサポートされている画像形式かチェック
        if img_file.is_file() and img_file.suffix.lower() in supported_formats:
            try:
                print(f"処理中: {img_file.name}")
                
                # 画像を開く
                img = Image.open(img_file)
                
                # RGBAの場合はRGBに変換（透明部分を白で埋める）
                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 1. 指定された値でトリミング
                print(f"  ステップ1: トリミング (上:{top}% 下:{bottom}% 左:{left}% 右:{right}%)")
                img = crop_with_custom_margins(img, top=top, bottom=bottom, left=left, right=right)
                
                # 2. 白い余白を除去
                print(f"  ステップ2: 白い余白を除去")
                img = remove_white_borders(img, threshold=240)
                
                # 3. ページ番号除去処理（オプション）
                if remove_page_num:
                    if page_position == 'top':
                        # 3-1. 上部から指定%をトリミング
                        print(f"  ステップ3: 上部{page_crop_percentage}%トリミング（ページ番号除去）")
                        img = crop_top(img, percentage=page_crop_percentage)
                        
                        # 3-2. 上端の白いマージンを除去
                        print(f"  ステップ4: 上端の白いマージンを除去")
                        img = remove_top_white_margin(img, threshold=240)
                    else:  # bottom
                        # 3-1. 下部から指定%をトリミング
                        print(f"  ステップ3: 下部{page_crop_percentage}%トリミング（ページ番号除去）")
                        img = crop_bottom(img, percentage=page_crop_percentage)
                        
                        # 3-2. 下端の白いマージンを除去
                        print(f"  ステップ4: 下端の白いマージンを除去")
                        img = remove_bottom_white_margin(img, threshold=240)
                
                # 4. 上下左右に5pxの白い余白を追加
                print(f"  ステップ{5 if remove_page_num else 3}: 上下左右に5pxの白い余白を追加")
                img = add_white_border(img, border_width=5)
                
                # croppedフォルダに保存
                output_path = output_folder / img_file.name
                
                # 保存時の品質設定
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
    
    print("=" * 60)
    print(f"処理完了: {processed_count}個の画像を処理しました。")
    print(f"出力先: {output_folder}")

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
    
    position_input = input("選択してください (1/2, デフォルト: 1): ").strip()
    
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
    else:
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
    
    print(f"→ トリミング率: {config['crop_percentage']}%")
    
    return config

def main():
    """メイン関数"""
    print("=" * 60)
    print("画像トリミング・余白除去ツール")
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
    
    print("\n" + "=" * 60)
    print("処理を開始します...")
    print("処理順序:")
    print(f"1. トリミング (上:{top}% 下:{bottom}% 左:{left}% 右:{right}%)")
    print("2. 白い余白を除去")
    
    if page_num_config['remove']:
        position_text = "上部" if page_num_config['position'] == 'top' else "下部"
        print(f"3. {position_text}{page_num_config['crop_percentage']}%トリミング")
        print(f"4. {position_text[0]}端の白いマージンを除去")
        print("5. 上下左右に5pxの白い余白を追加")
    else:
        print("3. 上下左右に5pxの白い余白を追加")
    
    print("=" * 60 + "\n")
    
    # 画像処理を実行
    process_images(folder_path, crop_margins, page_num_config)

if __name__ == "__main__":
    main()