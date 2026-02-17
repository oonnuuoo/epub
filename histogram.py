import os
import numpy as np
from PIL import Image, ImageDraw


SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')
HIST_HEIGHT = 256
GAP = 4
BG_COLOR = 240
BAR_X_COLOR = 70
BAR_Y_COLOR = 70
AXIS_COLOR = 180


def draw_x_histogram(draw, x_profile, ox, oy, width):
    """X軸ヒストグラム（列ごとの平均輝度）を描画"""
    # 軸線
    draw.line([(ox, oy + HIST_HEIGHT), (ox + width, oy + HIST_HEIGHT)], fill=AXIS_COLOR)
    draw.line([(ox, oy), (ox, oy + HIST_HEIGHT)], fill=AXIS_COLOR)
    # バー
    for x in range(width):
        bar_h = int(x_profile[x] * HIST_HEIGHT / 255)
        if bar_h > 0:
            draw.line([(ox + x, oy + HIST_HEIGHT - bar_h), (ox + x, oy + HIST_HEIGHT)], fill=BAR_X_COLOR)


def draw_y_histogram(draw, y_profile, ox, oy, height):
    """Y軸ヒストグラム（行ごとの平均輝度）を描画"""
    # 軸線
    draw.line([(ox, oy), (ox, oy + height)], fill=AXIS_COLOR)
    draw.line([(ox, oy + height), (ox + HIST_HEIGHT, oy + height)], fill=AXIS_COLOR)
    # バー
    for y in range(height):
        bar_w = int(y_profile[y] * HIST_HEIGHT / 255)
        if bar_w > 0:
            draw.line([(ox, oy + y), (ox + bar_w, oy + y)], fill=BAR_Y_COLOR)


def generate_histogram(image_path, output_dir):
    """
    画像のX軸・Y軸方向の輝度ヒストグラムを生成して保存する。
    PIL直接描画で高速処理。
    """
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    height, width = img_array.shape

    x_profile = np.mean(img_array, axis=0)
    y_profile = np.mean(img_array, axis=1)

    # キャンバスサイズ: [元画像] GAP [Y軸ヒストグラム] の横幅、[X軸ヒストグラム] GAP [元画像] の縦幅
    canvas_w = width + GAP + HIST_HEIGHT
    canvas_h = HIST_HEIGHT + GAP + height
    canvas = Image.new('L', (canvas_w, canvas_h), BG_COLOR)

    # 元画像を右下に配置
    img_ox = 0
    img_oy = HIST_HEIGHT + GAP
    canvas.paste(img, (img_ox, img_oy))

    draw = ImageDraw.Draw(canvas)

    # X軸ヒストグラムを上部に描画（元画像の上）
    draw_x_histogram(draw, x_profile, img_ox, 0, width)

    # Y軸ヒストグラムを右側に描画（元画像の右）
    draw_y_histogram(draw, y_profile, width + GAP, img_oy, height)

    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{basename}_histogram.png")
    canvas.save(output_path)
    return output_path


def main():
    folder = input("フォルダのパスを入力してください (Drag & Drop): ").strip().strip('"').strip("'")

    if not os.path.isdir(folder):
        print(f"エラー: フォルダが見つかりません: {folder}")
        return

    output_dir = os.path.join(folder, "histograms")
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(SUPPORTED_EXTENSIONS)
    ])

    if not image_files:
        print("対象画像が見つかりませんでした。")
        return

    print(f"{len(image_files)} 枚の画像を処理します...")

    for i, filename in enumerate(image_files, 1):
        image_path = os.path.join(folder, filename)
        try:
            out = generate_histogram(image_path, output_dir)
            print(f"  [{i}/{len(image_files)}] {filename} -> {os.path.basename(out)}")
        except Exception as e:
            print(f"  [{i}/{len(image_files)}] {filename} エラー: {e}")

    print(f"\n完了。出力先: {output_dir}")


if __name__ == "__main__":
    main()
