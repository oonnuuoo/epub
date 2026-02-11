import numpy as np
from PIL import Image

def remove_top_white_margin_with_skip(img, threshold=200):
    """
    画像上端の白いマージンのみを除去（5px以下の非白色部分をスキップ）
    
    Args:
        img: PIL Image object
        threshold: 白と判定する輝度値の閾値（デフォルト200）
    
    Returns:
        上端の白いマージンを除去したPIL Image
    """
    # グレースケールに変換
    gray = img.convert('L')
    
    # NumPy配列に変換
    img_array = np.array(gray)
    
    height, width = img_array.shape
    skip_threshold = 5  # スキップする非白色部分の最大高さ
    
    # 上から下に向かって境界を探す
    row = 0
    while row < height:
        # この行に白でないピクセルがあるか確認
        if np.any(img_array[row] < threshold):
            # 非白色ピクセルを発見、各x座標での連続長を調べる
            has_significant_content = False
            
            for col in range(width):
                if img_array[row, col] < threshold:
                    # この列での連続する非白色ピクセル数をカウント
                    consecutive_length = 0
                    for check_row in range(row, height):
                        if img_array[check_row, col] < threshold:
                            consecutive_length += 1
                        else:
                            break
                    
                    # 連続長が閾値より大きい場合、有効なコンテンツとみなす
                    if consecutive_length > skip_threshold:
                        has_significant_content = True
                        break
            
            if has_significant_content:
                # 有効なコンテンツが見つかったら、ここからをトリミング
                if row > 0:
                    return img.crop((0, row, width, height))
                else:
                    return img
            else:
                # 5px以下の非白色部分なので5pxスキップ
                row += skip_threshold
        else:
            # この行は完全に白なので次の行へ
            row += 1
    
    # すべて白の場合は元の画像を返す
    return img

def remove_bottom_white_margin_with_skip(img, threshold=200):
    """
    画像下端の白いマージンのみを除去（5px以下の非白色部分をスキップ）
    
    Args:
        img: PIL Image object
        threshold: 白と判定する輝度値の閾値（デフォルト200）
    
    Returns:
        下端の白いマージンを除去したPIL Image
    """
    # グレースケールに変換
    gray = img.convert('L')
    
    # NumPy配列に変換
    img_array = np.array(gray)
    
    height, width = img_array.shape
    skip_threshold = 5  # スキップする非白色部分の最大高さ
    
    # 下から上に向かって境界を探す
    row = height - 1
    while row >= 0:
        # この行に白でないピクセルがあるか確認
        if np.any(img_array[row] < threshold):
            # 非白色ピクセルを発見、各x座標での連続長を調べる
            has_significant_content = False
            
            for col in range(width):
                if img_array[row, col] < threshold:
                    # この列での連続する非白色ピクセル数をカウント（上方向に）
                    consecutive_length = 0
                    for check_row in range(row, -1, -1):
                        if img_array[check_row, col] < threshold:
                            consecutive_length += 1
                        else:
                            break
                    
                    # 連続長が閾値より大きい場合、有効なコンテンツとみなす
                    if consecutive_length > skip_threshold:
                        has_significant_content = True
                        break
            
            if has_significant_content:
                # 有効なコンテンツが見つかったら、ここまでをトリミング
                if row < height - 1:
                    return img.crop((0, 0, width, row + 1))
                else:
                    return img
            else:
                # 5px以下の非白色部分なので5pxスキップ
                row -= skip_threshold
        else:
            # この行は完全に白なので次の行へ
            row -= 1
    
    # すべて白の場合は元の画像を返す
    return img

def remove_white_borders_with_skip(img, threshold=200):
    """
    輝度値を基準に白い余白を除去（5px以下の非白色部分をスキップ）
    上下左右すべての余白を一度に処理
    
    Args:
        img: PIL Image object
        threshold: 白と判定する輝度値の閾値（デフォルト200）
    
    Returns:
        余白除去後のPIL Image
    """
    # グレースケールに変換
    gray = img.convert('L')
    
    # NumPy配列に変換
    img_array = np.array(gray)
    
    height, width = img_array.shape
    skip_threshold = 5  # スキップする非白色部分の最大長
    
    # 上端の境界を探索
    top_boundary = 0
    row = 0
    while row < height:
        if np.any(img_array[row] < threshold):
            # 各列での連続長をチェック
            has_significant_content = False
            for col in range(width):
                if img_array[row, col] < threshold:
                    consecutive_length = 0
                    for check_row in range(row, height):
                        if img_array[check_row, col] < threshold:
                            consecutive_length += 1
                        else:
                            break
                    if consecutive_length > skip_threshold:
                        has_significant_content = True
                        break
            
            if has_significant_content:
                top_boundary = row
                break
            else:
                row += skip_threshold
        else:
            row += 1
    
    # 下端の境界を探索
    bottom_boundary = height - 1
    row = height - 1
    while row >= 0:
        if np.any(img_array[row] < threshold):
            # 各列での連続長をチェック
            has_significant_content = False
            for col in range(width):
                if img_array[row, col] < threshold:
                    consecutive_length = 0
                    for check_row in range(row, -1, -1):
                        if img_array[check_row, col] < threshold:
                            consecutive_length += 1
                        else:
                            break
                    if consecutive_length > skip_threshold:
                        has_significant_content = True
                        break
            
            if has_significant_content:
                bottom_boundary = row
                break
            else:
                row -= skip_threshold
        else:
            row -= 1
    
    # 左端の境界を探索
    left_boundary = 0
    col = 0
    while col < width:
        if np.any(img_array[:, col] < threshold):
            # 各行での連続長をチェック
            has_significant_content = False
            for row in range(height):
                if img_array[row, col] < threshold:
                    consecutive_length = 0
                    for check_col in range(col, width):
                        if img_array[row, check_col] < threshold:
                            consecutive_length += 1
                        else:
                            break
                    if consecutive_length > skip_threshold:
                        has_significant_content = True
                        break
            
            if has_significant_content:
                left_boundary = col
                break
            else:
                col += skip_threshold
        else:
            col += 1
    
    # 右端の境界を探索
    right_boundary = width - 1
    col = width - 1
    while col >= 0:
        if np.any(img_array[:, col] < threshold):
            # 各行での連続長をチェック
            has_significant_content = False
            for row in range(height):
                if img_array[row, col] < threshold:
                    consecutive_length = 0
                    for check_col in range(col, -1, -1):
                        if img_array[row, check_col] < threshold:
                            consecutive_length += 1
                        else:
                            break
                    if consecutive_length > skip_threshold:
                        has_significant_content = True
                        break
            
            if has_significant_content:
                right_boundary = col
                break
            else:
                col -= skip_threshold
        else:
            col -= 1
    
    # 境界が有効な場合のみクロップ
    if (top_boundary < bottom_boundary and left_boundary < right_boundary and
        top_boundary >= 0 and bottom_boundary < height and
        left_boundary >= 0 and right_boundary < width):
        return img.crop((left_boundary, top_boundary, right_boundary + 1, bottom_boundary + 1))
    else:
        return img

# 元のコードとの置き換え用関数
def remove_top_white_margin(img, threshold=200):
    """元の関数名でスキップ機能付きの関数を呼び出し"""
    return remove_top_white_margin_with_skip(img, threshold)

def remove_bottom_white_margin(img, threshold=200):
    """元の関数名でスキップ機能付きの関数を呼び出し"""
    return remove_bottom_white_margin_with_skip(img, threshold)

def remove_white_borders(img, threshold=200):
    """元の関数名でスキップ機能付きの関数を呼び出し"""
    return remove_white_borders_with_skip(img, threshold)

# 使用例とテスト用関数
def test_skip_functionality():
    """
    スキップ機能のテスト用関数
    """
    # テスト用の画像を作成（白い背景に薄いノイズとメインコンテンツ）
    width, height = 300, 400
    test_array = np.full((height, width), 255, dtype=np.uint8)  # 白い背景
    
    # 上部に2pxの薄いノイズを追加（スキップされるべき）
    test_array[10:12, 50:60] = 180
    
    # 上部にメインコンテンツを追加（検出されるべき）
    test_array[50:80, 20:280] = 0  # 30pxの高さの黒いコンテンツ
    
    # 下部に3pxの薄いノイズを追加（スキップされるべき）
    test_array[350:353, 100:110] = 190
    
    # 下部にメインコンテンツを追加（検出されるべき）
    test_array[300:340, 30:270] = 50  # 40pxの高さのグレーコンテンツ
    
    # PIL Imageに変換
    test_img = Image.fromarray(test_array, mode='L').convert('RGB')
    
    print("テスト画像作成完了:")
    print(f"  - 元画像サイズ: {width} x {height}")
    print(f"  - 上部ノイズ: 行10-12 (2px高)")
    print(f"  - 上部コンテンツ: 行50-80 (30px高)")
    print(f"  - 下部コンテンツ: 行300-340 (40px高)")
    print(f"  - 下部ノイズ: 行350-353 (3px高)")
    
    # スキップ機能付きで処理
    result_img = remove_white_borders_with_skip(test_img, threshold=200)
    
    print(f"\n処理結果:")
    print(f"  - 処理後サイズ: {result_img.size}")
    print(f"  - 期待される上端: 50 (ノイズをスキップ)")
    print(f"  - 期待される下端: 340 (ノイズをスキップ)")
    
    return result_img

if __name__ == "__main__":
    # テスト実行
    test_result = test_skip_functionality()