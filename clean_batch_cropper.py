#!/usr/bin/env python3
"""
構文修正版バッチ画像トリミングプログラム
新しいトリミング基準:
1. 画像中央から90%の範囲内でコンテンツ検出
2. ピクセルの明度値を使用してコンテンツを判定
3. 検出されたコンテンツの周囲の白い余白を除去

必要なライブラリ:
pip install opencv-python pillow numpy tqdm

使用方法:
python clean_batch_cropper.py --input_dir "./images" --output_dir "./cropped"
"""

import os
import argparse
import cv2
import numpy as np
from PIL import Image
import logging
from pathlib import Path
from tqdm import tqdm
import time


class TextAreaCropper:
    """テキスト領域自動トリミングクラス"""
    
    def __init__(self, threshold=120, min_margin=0, debug_mode=False):
        """
        初期化
        
        Args:
            threshold (int): 二値化の閾値 (50-200)
            min_margin (int): 最小マージン (0-10)
            debug_mode (bool): デバッグモード
        """
        self.threshold = threshold
        self.min_margin = min_margin
        self.debug_mode = debug_mode
        self.setup_logging()
    
    def setup_logging(self):
        """ログ設定"""
        log_level = logging.DEBUG if self.debug_mode else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('crop_log.txt', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_image(self, image_path):
        """画像を読み込み"""
        try:
            pil_image = Image.open(image_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            height, width = cv_image.shape[:2]
            
            self.logger.info(f"画像読み込み成功: {os.path.basename(image_path)} ({width}x{height})")
            return pil_image, cv_image, width, height
            
        except Exception as e:
            self.logger.error(f"画像読み込みエラー: {os.path.basename(image_path)} - {str(e)}")
            return None
    
    def detect_content_bounds(self, cv_image, width, height):
        """
        コンテンツ領域の境界を検出（明度ベース）
        """
        # 中央90%の検出領域を定義
        detection_margin_x = int(width * 0.05)
        detection_margin_y = int(height * 0.05)
        
        detection_left = detection_margin_x
        detection_right = width - detection_margin_x
        detection_top = detection_margin_y
        detection_bottom = height - detection_margin_y
        
        self.logger.info(f"検出領域: ({detection_left},{detection_top}) - ({detection_right},{detection_bottom})")
        
        # 検出領域内の画像を抽出
        detection_roi = cv_image[detection_top:detection_bottom, detection_left:detection_right]
        gray_roi = cv2.cvtColor(detection_roi, cv2.COLOR_BGR2GRAY)
        
        # 明度値ベースでコンテンツ検出
        content_mask = self._detect_content_by_brightness(gray_roi)
        
        # コンテンツの境界ボックスを検出
        content_bounds = self._find_content_bounding_box(content_mask)
        
        if content_bounds is None:
            self.logger.warning("コンテンツが検出されませんでした")
            return {
                'left': int(width * 0.1),
                'top': int(height * 0.1),
                'right': int(width * 0.9),
                'bottom': int(height * 0.9),
                'width': int(width * 0.8),
                'height': int(height * 0.8)
            }
        
        # 検出領域の座標を全体座標に変換
        content_left = detection_left + content_bounds['left']
        content_top = detection_top + content_bounds['top']
        content_right = detection_left + content_bounds['right']
        content_bottom = detection_top + content_bounds['bottom']
        
        # 白い余白を除去
        final_bounds = self._remove_white_margins(
            cv_image, content_left, content_top, content_right, content_bottom
        )
        
        text_bounds = {
            'left': final_bounds['left'],
            'top': final_bounds['top'],
            'right': final_bounds['right'],
            'bottom': final_bounds['bottom'],
            'width': final_bounds['right'] - final_bounds['left'] + 1,
            'height': final_bounds['bottom'] - final_bounds['top'] + 1
        }
        
        # ログ出力
        left_margin = final_bounds['left']
        right_margin = width - final_bounds['right']
        top_margin = final_bounds['top']
        bottom_margin = height - final_bounds['bottom']
        
        self.logger.info(f"コンテンツ検出結果 - 余白 L:{left_margin} R:{right_margin} T:{top_margin} B:{bottom_margin}")
        
        return text_bounds
    
    def _detect_content_by_brightness(self, gray_roi):
        """明度値ベースでコンテンツを検出"""
        # 明度統計を計算
        mean_brightness = np.mean(gray_roi)
        std_brightness = np.std(gray_roi)
        
        # 動的閾値を計算
        brightness_threshold = mean_brightness - std_brightness * 0.5
        final_threshold = min(brightness_threshold, self.threshold)
        
        # コンテンツマスクを作成
        content_mask = gray_roi < final_threshold
        
        # ノイズ除去
        kernel = np.ones((3, 3), np.uint8)
        content_mask = cv2.morphologyEx(content_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_OPEN, kernel)
        
        # 小さな領域を除去
        content_mask = self._remove_small_regions(content_mask, min_area=50)
        
        content_pixels = np.sum(content_mask)
        total_pixels = gray_roi.size
        content_ratio = content_pixels / total_pixels
        
        self.logger.debug(f"明度検出: 平均={mean_brightness:.1f}, 閾値={final_threshold:.1f}, 比率={content_ratio:.1%}")
        
        return content_mask.astype(bool)
    
    def _remove_small_regions(self, mask, min_area=50):
        """小さな領域を除去"""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        filtered_mask = np.zeros_like(mask)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                filtered_mask[labels == i] = 1
        
        return filtered_mask
    
    def _find_content_bounding_box(self, content_mask):
        """コンテンツの境界ボックスを検出"""
        if not np.any(content_mask):
            return None
        
        coords = np.column_stack(np.where(content_mask))
        if len(coords) == 0:
            return None
        
        top = np.min(coords[:, 0])
        bottom = np.max(coords[:, 0])
        left = np.min(coords[:, 1])
        right = np.max(coords[:, 1])
        
        return {
            'left': left,
            'top': top,
            'right': right,
            'bottom': bottom
        }
    
    def _remove_white_margins(self, cv_image, left, top, right, bottom):
        """白い余白を除去"""
        roi = cv_image[top:bottom+1, left:right+1]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        white_threshold = 240
        adjusted_bounds = self._scan_white_margins(gray_roi, white_threshold)
        
        return {
            'left': left + adjusted_bounds['left'],
            'top': top + adjusted_bounds['top'],
            'right': left + adjusted_bounds['right'],
            'bottom': top + adjusted_bounds['bottom']
        }
    
    def _scan_white_margins(self, gray_roi, white_threshold):
        """各方向から白い余白をスキャン"""
        height, width = gray_roi.shape
        
        # 上から下へスキャン
        top_trim = 0
        for y in range(height):
            if np.mean(gray_roi[y, :]) < white_threshold:
                top_trim = max(0, y - 2)
                break
        
        # 下から上へスキャン
        bottom_trim = height - 1
        for y in range(height - 1, -1, -1):
            if np.mean(gray_roi[y, :]) < white_threshold:
                bottom_trim = min(height - 1, y + 2)
                break
        
        # 左から右へスキャン
        left_trim = 0
        for x in range(width):
            if np.mean(gray_roi[:, x]) < white_threshold:
                left_trim = max(0, x - 2)
                break
        
        # 右から左へスキャン
        right_trim = width - 1
        for x in range(width - 1, -1, -1):
            if np.mean(gray_roi[:, x]) < white_threshold:
                right_trim = min(width - 1, x + 2)
                break
        
        # 妥当性チェック
        if left_trim >= right_trim or top_trim >= bottom_trim:
            self.logger.warning("白余白除去で無効な境界が検出されました")
            return {
                'left': 0,
                'top': 0,
                'right': width - 1,
                'bottom': height - 1
            }
        
        return {
            'left': left_trim,
            'top': top_trim,
            'right': right_trim,
            'bottom': bottom_trim
        }
    
    def crop_image(self, pil_image, text_bounds):
        """画像をトリミング"""
        left = max(0, text_bounds['left'] - self.min_margin)
        top = max(0, text_bounds['top'] - self.min_margin)
        right = min(pil_image.width - 1, text_bounds['right'] + self.min_margin)
        bottom = min(pil_image.height - 1, text_bounds['bottom'] + self.min_margin)
        
        cropped = pil_image.crop((left, top, right + 1, bottom + 1))
        
        original_area = pil_image.width * pil_image.height
        cropped_area = cropped.width * cropped.height
        reduction = ((original_area - cropped_area) / original_area) * 100
        
        self.logger.info(f"トリミング完了: {cropped.width}x{cropped.height} (削減率: {reduction:.1f}%)")
        
        return cropped
    
    def process_image(self, input_path, output_path):
        """単一画像の処理"""
        try:
            result = self.load_image(input_path)
            if result is None:
                return False
            
            pil_image, cv_image, width, height = result
            text_bounds = self.detect_content_bounds(cv_image, width, height)
            
            if text_bounds['width'] <= 0 or text_bounds['height'] <= 0:
                self.logger.warning(f"有効な領域が検出されませんでした: {os.path.basename(input_path)}")
                return False
            
            cropped_image = self.crop_image(pil_image, text_bounds)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cropped_image.save(output_path, quality=95, optimize=True)
            
            self.logger.info(f"保存完了: {os.path.basename(output_path)}")
            return True
            
        except Exception as e:
            self.logger.error(f"処理エラー: {os.path.basename(input_path)} - {str(e)}")
            return False


def get_image_files(directory):
    """ディレクトリから画像ファイルを取得"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    return sorted(image_files)


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='簡素化版バッチ画像トリミングプログラム')
    parser.add_argument('--input_dir', '-i', required=True, help='入力ディレクトリパス')
    parser.add_argument('--output_dir', '-o', default='./cropped', help='出力ディレクトリパス')
    parser.add_argument('--threshold', '-t', type=int, default=120, help='二値化閾値 (50-200)')
    parser.add_argument('--margin', '-m', type=int, default=0, help='最小マージン (0-10)')
    parser.add_argument('--preserve_structure', '-p', action='store_true', help='ディレクトリ構造を保持')
    parser.add_argument('--debug', '-d', action='store_true', help='デバッグモード')
    
    args = parser.parse_args()
    
    print("🖼️ 簡素化版バッチコンテンツトリミングプログラム v2.1")
    print("📋 新トリミング基準:")
    print("   1. 画像中央90%範囲でコンテンツ検出")
    print("   2. 明度値ベースのシンプルな判定")
    print("   3. 周囲白余白の自動除去")
    print("=" * 60)
    
    # 入力ディレクトリの確認
    if not os.path.exists(args.input_dir):
        print(f"エラー: 入力ディレクトリが存在しません: {args.input_dir}")
        return
    
    # 画像ファイル取得
    image_files = get_image_files(args.input_dir)
    if not image_files:
        print(f"エラー: 画像ファイルが見つかりません: {args.input_dir}")
        return
    
    print(f"発見された画像ファイル: {len(image_files)}個")
    print(f"入力ディレクトリ: {args.input_dir}")
    print(f"出力ディレクトリ: {args.output_dir}")
    print(f"設定 - 閾値: {args.threshold}, マージン: {args.margin}")
    print("-" * 60)
    
    # トリミング処理
    cropper = TextAreaCropper(
        threshold=args.threshold,
        min_margin=args.margin,
        debug_mode=args.debug
    )
    
    successful = 0
    failed = 0
    start_time = time.time()
    
    for input_path in tqdm(image_files, desc="処理中"):
        try:
            # 出力パス生成
            if args.preserve_structure:
                rel_path = os.path.relpath(input_path, args.input_dir)
                output_path = os.path.join(args.output_dir, rel_path)
            else:
                filename = os.path.basename(input_path)
                output_path = os.path.join(args.output_dir, filename)
            
            # 既存ファイルスキップ
            if os.path.exists(output_path):
                print(f"スキップ（既存）: {os.path.basename(output_path)}")
                continue
            
            # 処理実行
            if cropper.process_image(input_path, output_path):
                successful += 1
            else:
                failed += 1
                
        except KeyboardInterrupt:
            print("\n処理が中断されました。")
            break
        except Exception as e:
            print(f"予期しないエラー: {os.path.basename(input_path)} - {str(e)}")
            failed += 1
    
    # 結果表示
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("処理完了")
    print(f"成功: {successful}個")
    print(f"失敗: {failed}個")
    print(f"処理時間: {elapsed_time:.2f}秒")
    if len(image_files) > 0:
        print(f"平均処理時間: {elapsed_time/len(image_files):.2f}秒/ファイル")
    print(f"出力ディレクトリ: {args.output_dir}")


if __name__ == "__main__":
    main()
