#!/usr/bin/env python3
"""
簡素化版バッチ画像トリミングプログラム
新しいトリミング基準:
1. 画像中央から90%の範囲内でコンテンツ検出
2. ピクセルの明度値を使用してコンテンツを判定
3. 検出されたコンテンツの周囲の白い余白を除去

必要なライブラリ:
pip install opencv-python pillow numpy tqdm

使用方法:
1. コマンドライン版:
   python complete_batch_cropper.py --mode cli --input_dir "./images" --output_dir "./cropped"

2. 設定ファイル版:
   python complete_batch_cropper.py --mode config

3. GUI版:
   python complete_batch_cropper.py --mode gui

4. デバッグモード:
   python complete_batch_cropper.py --mode cli --input_dir "./images" --debug

5. ヘルプ:
   python complete_batch_cropper.py --help
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
import json


class TextAreaCropper:
    """テキスト領域自動トリミングクラス"""
    
    def __init__(self, threshold=120, min_margin=0, debug_mode=False):
        """
        初期化
        
        Args:
            threshold (int): 二値化の閾値 (50-200)
            min_margin (int): 最小マージン (0-10)
            debug_mode (bool): デバッグモード（詳細ログと可視化）
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
    
    def save_debug_visualization(self, image_path, row_projection, top_bound, bottom_bound, height):
        """デバッグ用の可視化画像を保存"""
        if not self.debug_mode:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # プロジェクション可視化
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
            
            # 行プロジェクション
            ax1.plot(row_projection, range(len(row_projection)))
            ax1.axhline(y=top_bound, color='red', linestyle='--', label=f'Top: {top_bound}')
            ax1.axhline(y=bottom_bound, color='red', linestyle='--', label=f'Bottom: {bottom_bound}')
            ax1.set_ylabel('Row (pixels)')
            ax1.set_xlabel('Density')
            ax1.set_title('Row Projection')
            ax1.legend()
            ax1.invert_yaxis()
            
            # 境界領域の拡大表示
            zoom_start = max(0, top_bound - 50)
            zoom_end = min(len(row_projection), top_bound + 100)
            ax2.plot(row_projection[zoom_start:zoom_end], range(zoom_start, zoom_end))
            ax2.axhline(y=top_bound, color='red', linestyle='--', label=f'Top: {top_bound}')
            ax2.set_ylabel('Row (pixels)')
            ax2.set_xlabel('Density')
            ax2.set_title('Top Boundary Detail')
            ax2.legend()
            ax2.invert_yaxis()
            
            # 保存
            debug_path = os.path.splitext(image_path)[0] + '_debug.png'
            plt.savefig(debug_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.debug(f"デバッグ可視化を保存: {debug_path}")
            
        except ImportError:
            self.logger.debug("matplotlib未インストール - 可視化をスキップ")
        except Exception as e:
            self.logger.debug(f"可視化エラー: {str(e)}")
    
    def load_image(self, image_path):
        """
        画像を読み込み
        
        Args:
            image_path (str): 画像ファイルパス
            
        Returns:
            tuple: (PIL Image, OpenCV Image, width, height) または None
        """
        try:
            # PILで読み込み（日本語パス対応）
            pil_image = Image.open(image_path)
            
            # RGBに変換（必要に応じて）
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # OpenCV形式に変換
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            height, width = cv_image.shape[:2]
            
            self.logger.info(f"画像読み込み成功: {os.path.basename(image_path)} ({width}x{height})")
            
            return pil_image, cv_image, width, height
            
        except Exception as e:
            self.logger.error(f"画像読み込みエラー: {os.path.basename(image_path)} - {str(e)}")
            return None
    
    def detect_text_bounds(self, cv_image, width, height):
        """
        コンテンツ領域の境界を検出（簡素化版）
        新しい基準:
        1. 画像中央から90%の範囲内でコンテンツ検出
        2. ピクセルの明度値を使用してコンテンツを判定
        3. 検出されたコンテンツの周囲の白い余白を除去
        
        Args:
            cv_image: OpenCV画像
            width (int): 画像幅
            height (int): 画像高さ
            
        Returns:
            dict: コンテンツ境界情報 {'left', 'top', 'right', 'bottom', 'width', 'height'}
        """
        # 中央90%の検出領域を定義
        detection_margin_x = int(width * 0.05)  # 左右5%ずつ除外
        detection_margin_y = int(height * 0.05)  # 上下5%ずつ除外
        
        detection_left = detection_margin_x
        detection_right = width - detection_margin_x
        detection_top = detection_margin_y
        detection_bottom = height - detection_margin_y
        
        self.logger.info(f"検出領域: ({detection_left},{detection_top}) - ({detection_right},{detection_bottom})")
        
        # 検出領域内の画像を抽出
        detection_roi = cv_image[detection_top:detection_bottom, detection_left:detection_right]
        
        # グレースケール変換
        gray_roi = cv2.cvtColor(detection_roi, cv2.COLOR_BGR2GRAY)
        
        # 明度値ベースでコンテンツ検出
        content_mask = self._detect_content_by_brightness(gray_roi)
        
        # コンテンツの境界ボックスを検出
        content_bounds = self._find_content_bounding_box(content_mask)
        
        if content_bounds is None:
            self.logger.warning("コンテンツが検出されませんでした")
            # フォールバック: 画像全体の10%マージンを使用
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
        
        # 白い余白を除去するための微調整
        final_bounds = self._remove_white_margins_simple(
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
        
        # デバッグ可視化（debug_modeの場合）
        if hasattr(self, '_current_image_path'):
            self.save_brightness_debug_visualization(
                self._current_image_path, cv_image, content_mask, text_bounds, 
                detection_left, detection_top, detection_right, detection_bottom
            )
        
        return text_bounds
    
    def _detect_content_by_brightness(self, gray_roi):
        """
        明度値ベースでコンテンツを検出（簡素化）
        
        Args:
            gray_roi: グレースケールROI画像
            
        Returns:
            numpy.ndarray: コンテンツマスク（bool型）
        """
        # 明度統計を計算
        mean_brightness = np.mean(gray_roi)
        std_brightness = np.std(gray_roi)
        
        # 動的閾値を計算
        # 背景（白い部分）と前景（コンテンツ）を分離
        brightness_threshold = mean_brightness - std_brightness * 0.5
        
        # ユーザー設定の閾値も考慮
        final_threshold = min(brightness_threshold, self.threshold)
        
        # コンテンツマスクを作成（閾値より暗いピクセルをコンテンツとみなす）
        content_mask = gray_roi < final_threshold
        
        # ノイズ除去（小さな点を除去、穴を埋める）
        kernel = np.ones((3, 3), np.uint8)
        
        # モルフォロジー演算でノイズ除去
        content_mask = cv2.morphologyEx(content_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_OPEN, kernel)
        
        # 小さな領域を除去
        content_mask = self._remove_small_regions(content_mask, min_area=50)
        
        content_pixels = np.sum(content_mask)
        total_pixels = gray_roi.size
        content_ratio = content_pixels / total_pixels
        
        self.logger.debug(f"明度検出: 平均={mean_brightness:.1f}, 標準偏差={std_brightness:.1f}, 閾値={final_threshold:.1f}")
        self.logger.debug(f"コンテンツ比率: {content_ratio:.1%} ({content_pixels}/{total_pixels})")
        
        return content_mask.astype(bool)
    
    def _remove_small_regions(self, mask, min_area=50):
        """小さな領域を除去"""
        # 連結成分を検出
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        
        # 新しいマスクを作成
        filtered_mask = np.zeros_like(mask)
        
        # 各連結成分をチェック
        for i in range(1, num_labels):  # 0はバックグラウンド
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                filtered_mask[labels == i] = 1
        
        return filtered_mask
    
    def _remove_white_margins_simple(self, cv_image, left, top, right, bottom):
        """
        検出されたコンテンツ周囲の白い余白を除去（簡素化版）
        
        Args:
            cv_image: 元画像
            left, top, right, bottom: 初期境界
            
        Returns:
            dict: 調整された境界
        """
        # ROIを抽出
        roi = cv_image[top:bottom+1, left:right+1]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 白色の閾値（明度が高い領域）
        white_threshold = 240
        
        # 各方向から白い余白をスキャン
        adjusted_bounds = self._scan_white_margins(gray_roi, white_threshold)
        
        return {
            'left': left + adjusted_bounds['left'],
            'top': top + adjusted_bounds['top'],
            'right': left + adjusted_bounds['right'],
            'bottom': top + adjusted_bounds['bottom']
        }
    
    def _scan_white_margins(self, gray_roi, white_threshold):
        """各方向から白い余白をスキャン（簡素化版）"""
        height, width = gray_roi.shape
        
        # 上から下へスキャン
        top_trim = 0
        for y in range(height):
            row_mean = np.mean(gray_roi[y, :])
            if row_mean < white_threshold:  # 白くない行が見つかった
                top_trim = max(0, y - 2)  # 2ピクセルのマージンを残す
                break
        
        # 下から上へスキャン
        bottom_trim = height - 1
        for y in range(height - 1, -1, -1):
            row_mean = np.mean(gray_roi[y, :])
            if row_mean < white_threshold:  # 白くない行が見つかった
                bottom_trim = min(height - 1, y + 2)  # 2ピクセルのマージンを残す
                break
        
        # 左から右へスキャン
        left_trim = 0
        for x in range(width):
            col_mean = np.mean(gray_roi[:, x])
            if col_mean < white_threshold:  # 白くない列が見つかった
                left_trim = max(0, x - 2)  # 2ピクセルのマージンを残す
                break
        
        # 右から左へスキャン
        right_trim = width - 1
        for x in range(width - 1, -1, -1):
            col_mean = np.mean(gray_roi[:, x])
            if col_mean < white_threshold:  # 白くない列が見つかった
                right_trim = min(width - 1, x + 2)  # 2ピクセルのマージンを残す
                break
        
        # 妥当性チェック
        if left_trim >= right_trim or top_trim >= bottom_trim:
            self.logger.warning("白余白除去で無効な境界が検出されました - 元の境界を使用")
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
    
    def save_brightness_debug_visualization(self, image_path, cv_image, content_mask, bounds, 
                                          det_left, det_top, det_right, det_bottom):
        """明度ベース検出のデバッグ可視化"""
        if not self.debug_mode:
            return
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 元画像
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            axes[0, 0].imshow(rgb_image)
            axes[0, 0].set_title('Original Image')
            
            # 検出領域を表示（青枠）
            detection_rect = patches.Rectangle((det_left, det_top), 
                                             det_right - det_left, det_bottom - det_top,
                                             linewidth=2, edgecolor='blue', facecolor='none', 
                                             label='Detection Area (90%)')
            axes[0, 0].add_patch(detection_rect)
            
            # 最終境界を表示（赤枠）
            final_rect = patches.Rectangle((bounds['left'], bounds['top']), 
                                         bounds['width'], bounds['height'],
                                         linewidth=2, edgecolor='red', facecolor='none',
                                         label='Content Bounds')
            axes[0, 0].add_patch(final_rect)
            axes[0, 0].legend()
            
            # 明度ベースのコンテンツマスク
            axes[0, 1].imshow(content_mask, cmap='gray')
            axes[0, 1].set_title('Brightness-based Content Mask')
            
            # トリミング結果プレビュー
            cropped = rgb_image[bounds['top']:bounds['bottom']+1, bounds['left']:bounds['right']+1]
            axes[1, 0].imshow(cropped)
            axes[1, 0].set_title('Cropped Result')
            
            # 明度ヒストグラム
            detection_roi = cv_image[det_top:det_bottom, det_left:det_right]
            gray_roi = cv2.cvtColor(detection_roi, cv2.COLOR_BGR2GRAY)
            axes[1, 1].hist(gray_roi.ravel(), bins=256, range=[0, 256], alpha=0.7)
            axes[1, 1].axvline(x=self.threshold, color='red', linestyle='--', label=f'Threshold: {self.threshold}')
            axes[1, 1].set_title('Brightness Histogram')
            axes[1, 1].set_xlabel('Brightness Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
            
            # 統計情報をヒストグラムに追加
            mean_brightness = np.mean(gray_roi)
            std_brightness = np.std(gray_roi)
            axes[1, 1].axvline(x=mean_brightness, color='green', linestyle=':', label=f'Mean: {mean_brightness:.1f}')
            axes[1, 1].axvline(x=mean_brightness - std_brightness, color='orange', linestyle=':', 
                             label=f'Mean-Std: {mean_brightness - std_brightness:.1f}')
            axes[1, 1].legend()
            
            # 保存
            debug_path = os.path.splitext(image_path)[0] + '_brightness_debug.png'
            plt.tight_layout()
            plt.savefig(debug_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.debug(f"明度ベース検出デバッグ可視化を保存: {debug_path}")
            
        except ImportError:
            self.logger.debug("matplotlib未インストール - 可視化をスキップ")
        except Exception as e:
            self.logger.debug(f"可視化エラー: {str(e)})", edgecolor='blue', facecolor='none')
            axes[0, 0].add_patch(detection_rect)
            
            # 最終境界を表示
            final_rect = patches.Rectangle((bounds['left'], bounds['top']), 
                                         bounds['width'], bounds['height'],
                                         linewidth=2, edgecolor='red', facecolor='none')
            axes[0, 0].add_patch(final_rect)
            
            # コンテンツマスク
            axes[0, 1].imshow(content_mask, cmap='gray')
            axes[0, 1].set_title('Content Mask')
            
            # トリミング結果プレビュー
            cropped = rgb_image[bounds['top']:bounds['bottom']+1, bounds['left']:bounds['right']+1]
            axes[1, 0].imshow(cropped)
            axes[1, 0].set_title('Cropped Result')
            
            # 統計情報
            axes[1, 1].text(0.1, 0.9, f"Detection Area: {det_right-det_left}x{det_bottom-det_top}", transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.8, f"Content Area: {bounds['width']}x{bounds['height']}", transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.7, f"Left Margin: {bounds['left']}", transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.6, f"Right Margin: {cv_image.shape[1] - bounds['right']}", transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.5, f"Top Margin: {bounds['top']}", transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.4, f"Bottom Margin: {cv_image.shape[0] - bounds['bottom']}", transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Statistics')
            axes[1, 1].axis('off')
            
            # 保存
            debug_path = os.path.splitext(image_path)[0] + '_content_debug.png'
            plt.tight_layout()
            plt.savefig(debug_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.debug(f"コンテンツ検出デバッグ可視化を保存: {debug_path}")
            
        except ImportError:
            self.logger.debug("matplotlib未インストール - 可視化をスキップ")
        except Exception as e:
            self.logger.debug(f"可視化エラー: {str(e)}")
    
    def _detect_horizontal_bounds(self, col_projection, width, min_density):
        """左右境界の検出"""
        left_bound = 0
        right_bound = width - 1
        
        # 左境界：5列連続でテキストが検出される最初の位置
        consecutive_cols = 0
        for x in range(width):
            if col_projection[x] > min_density:
                consecutive_cols += 1
                if consecutive_cols >= 5:
                    left_bound = max(0, x - consecutive_cols + 1)
                    break
            else:
                consecutive_cols = 0
        
        # 右境界：5列連続でテキストが検出される最後の位置
        consecutive_cols = 0
        for x in range(width - 1, -1, -1):
            if col_projection[x] > min_density:
                consecutive_cols += 1
                if consecutive_cols >= 5:
                    right_bound = min(width - 1, x + consecutive_cols - 1)
                    break
            else:
                consecutive_cols = 0
        
        return left_bound, right_bound
    
    def _detect_vertical_bounds(self, row_projection, height, min_density, main_density):
        """上下境界の検出（ページ番号除去機能付き）"""
        top_bound = 0
        bottom_bound = height - 1
        
        # 上境界の厳密な検出
        top_bound = self._detect_top_boundary_strict(row_projection, height, min_density, main_density)
        
        # 下境界の検出（ページ番号除去）
        bottom_bound = self._detect_bottom_boundary(row_projection, height, min_density, main_density)
        
        return top_bound, bottom_bound
    
    def _detect_top_boundary_strict(self, row_projection, height, min_density, main_density):
        """上境界の厳密な検出"""
        # 複数のアプローチで上境界を検出し、最も厳密な結果を採用
        
        # アプローチ1: 連続テキスト行検出
        top_bound_1 = self._find_top_by_consecutive_text(row_projection, height, min_density, main_density)
        
        # アプローチ2: 密度勾配検出
        top_bound_2 = self._find_top_by_density_gradient(row_projection, height, min_density)
        
        # アプローチ3: 統計的外れ値検出
        top_bound_3 = self._find_top_by_statistical_analysis(row_projection, height, min_density)
        
        # アプローチ4: 累積密度による検出
        top_bound_4 = self._find_top_by_cumulative_density(row_projection, height, min_density)
        
        # 最も厳密（最大値）な境界を選択
        candidate_bounds = [top_bound_1, top_bound_2, top_bound_3, top_bound_4]
        valid_bounds = [b for b in candidate_bounds if b > 0]
        
        if valid_bounds:
            # 統計的に妥当な範囲内で最大値を選択
            median_bound = np.median(valid_bounds)
            # 中央値から大きく外れていない最大値を選択
            filtered_bounds = [b for b in valid_bounds if abs(b - median_bound) < height * 0.1]
            top_bound = max(filtered_bounds) if filtered_bounds else max(valid_bounds)
        else:
            top_bound = 0
        
        self.logger.info(f"上境界検出結果: 連続={top_bound_1}, 勾配={top_bound_2}, 統計={top_bound_3}, 累積={top_bound_4} -> 最終={top_bound}")
        
        return top_bound
    
    def _find_top_by_consecutive_text(self, row_projection, height, min_density, main_density):
        """連続テキスト行による上境界検出"""
        consecutive_rows = 0
        for y in range(int(height * 0.6)):  # 上60%を検索
            density = row_projection[y]
            
            if density > min_density:
                consecutive_rows += 1
                
                # より厳しい条件: 高密度3行連続 または 中密度5行連続 または 軽密度8行連続
                if (density > main_density and consecutive_rows >= 3) or \
                   (density > min_density * 2 and consecutive_rows >= 5) or \
                   (consecutive_rows >= 8):
                    return max(0, y - consecutive_rows + 1)
            else:
                consecutive_rows = 0
        
        # 最終的な境界の妥当性チェック
        top_bound, bottom_bound = self._validate_boundaries(
            top_bound, bottom_bound, height, row_projection, min_density
        )
        
        return top_bound, bottom_bound
    
    def _validate_boundaries(self, top_bound, bottom_bound, height, row_projection, min_density):
        """境界の妥当性チェックと調整"""
        
        # 上境界の妥当性チェック
        if top_bound > height * 0.4:  # 上境界が画像の40%を超えている場合は異常
            self.logger.warning(f"上境界が異常に大きい: {top_bound} -> 再検出")
            # より保守的な検出を実行
            for y in range(int(height * 0.3)):
                if row_projection[y] > min_density:
                    # 前後2行をチェック
                    valid_start = True
                    for check_y in range(y, min(height, y + 3)):
                        if row_projection[check_y] <= min_density * 0.5:
                            valid_start = False
                            break
                    
                    if valid_start:
                        top_bound = y
                        self.logger.info(f"保守的検出による上境界修正: {top_bound}")
                        break
        
        # 下境界の妥当性チェック
        if bottom_bound < height * 0.6:  # 下境界が画像の60%未満の場合は異常
            self.logger.warning(f"下境界が異常に小さい: {bottom_bound} -> 再検出")
            # より保守的な検出を実行
            for y in range(int(height * 0.9), int(height * 0.6), -1):
                if row_projection[y] > min_density:
                    bottom_bound = y
                    self.logger.info(f"保守的検出による下境界修正: {bottom_bound}")
                    break
        
        # 上下境界の最小距離チェック
        min_text_height = height * 0.2  # 最小テキスト高さは画像の20%
        if bottom_bound - top_bound < min_text_height:
            self.logger.warning(f"テキスト領域が小さすぎる: {bottom_bound - top_bound} < {min_text_height}")
            # 境界を拡張
            expand_amount = int((min_text_height - (bottom_bound - top_bound)) / 2)
            top_bound = max(0, top_bound - expand_amount)
            bottom_bound = min(height - 1, bottom_bound + expand_amount)
            self.logger.info(f"境界を拡張: 上={top_bound}, 下={bottom_bound}")
        
        return top_bound, bottom_bound
    
    def _find_top_by_density_gradient(self, row_projection, height, min_density):
        """密度勾配による上境界検出"""
        # 上から下への密度変化を分析
        max_gradient = 0
        gradient_position = 0
        
        # 移動平均でスムージング
        window_size = max(3, height // 100)
        smoothed = np.convolve(row_projection[:int(height * 0.6)], 
                              np.ones(window_size)/window_size, mode='valid')
        
        for i in range(1, len(smoothed) - 1):
            # 前後の勾配を計算
            gradient = smoothed[i + 1] - smoothed[i - 1]
            
            # 急激な密度上昇を検出
            if gradient > max_gradient and smoothed[i] > min_density:
                max_gradient = gradient
                gradient_position = i
        
        # 勾配位置から実際のテキスト開始位置を逆算
        if gradient_position > 0:
            # 勾配位置から前方にスキャンして実際のテキスト開始を発見
            for y in range(max(0, gradient_position - window_size), gradient_position + window_size):
                if y < len(row_projection) and row_projection[y] > min_density:
                    return y
        
        return 0
    
    def _find_top_by_statistical_analysis(self, row_projection, height, min_density):
        """統計的外れ値による上境界検出"""
        # 上40%の領域を分析
        upper_region = row_projection[:int(height * 0.4)]
        
        # 非ゼロ要素の統計を計算
        non_zero_densities = upper_region[upper_region > min_density]
        
        if len(non_zero_densities) == 0:
            return 0
        
        # 統計値計算
        mean_density = np.mean(non_zero_densities)
        std_density = np.std(non_zero_densities)
        threshold = mean_density + std_density * 0.5  # より保守的な閾値
        
        # 閾値を超える最初の連続領域を検出
        consecutive_count = 0
        for y, density in enumerate(upper_region):
            if density > threshold:
                consecutive_count += 1
                if consecutive_count >= 3:  # 3行連続
                    return max(0, y - consecutive_count + 1)
            else:
                consecutive_count = 0
        
        return 0
    
    def _find_top_by_cumulative_density(self, row_projection, height, min_density):
        """累積密度による上境界検出"""
        # 上50%の領域で累積密度を計算
        upper_region = row_projection[:int(height * 0.5)]
        
        # 累積密度を計算
        cumulative_density = np.cumsum(upper_region)
        total_density = cumulative_density[-1]
        
        if total_density == 0:
            return 0
        
        # 全体の5%の密度が蓄積された位置を探す
        target_density = total_density * 0.05
        
        for y, cum_density in enumerate(cumulative_density):
            if cum_density >= target_density:
                # その位置から実際のテキスト開始を探す
                for search_y in range(max(0, y - 5), min(len(row_projection), y + 10)):
                    if row_projection[search_y] > min_density:
                        return search_y
                return y
        
        return 0
    
    def _detect_bottom_boundary(self, row_projection, height, min_density, main_density):
        """下境界の検出（ページ番号除去）"""
        bottom_search_start = int(height * 0.95)  # 下5%は無視
        consecutive_rows = 0
        bottom_bound = height - 1
        
        for y in range(bottom_search_start, int(height * 0.5), -1):  # 下から上へ検索
            density = row_projection[y]
            
            if density > min_density:
                consecutive_rows += 1
                
                # 本文レベルの密度 + 3行連続 または 軽いテキスト + 7行連続
                if (density > main_density and consecutive_rows >= 3) or consecutive_rows >= 7:
                    bottom_bound = min(height - 1, y + consecutive_rows - 1)
                    break
            else:
                consecutive_rows = 0
        
        # ページ番号領域の特別検出と除去
        page_number_zone_top = int(height * 0.85)
        page_number_detected = False
        
        for y in range(page_number_zone_top, height):
            density = row_projection[y]
            
            # ページ番号らしき特徴：中程度の密度で孤立
            if min_density < density < main_density:
                prev_density = row_projection[y - 1] if y > 0 else 0
                next_density = row_projection[y + 1] if y < height - 1 else 0
                
                # 前後の行が空白または密度が低い
                if prev_density < min_density and next_density < min_density:
                    self.logger.info(f"ページ番号検出: 行{y} (密度: {density:.2f})")
                    page_number_detected = True
                    
                    # ページ番号より上で本文の最後を探す
                    for search_y in range(y - 1, int(height * 0.5), -1):
                        if row_projection[search_y] > main_density:
                            bottom_bound = min(bottom_bound, search_y)
                            break
                    break
        
        if page_number_detected:
            self.logger.info("✓ ページ番号を除去しました")
        
        # ヘッダー領域の強化検出と除去
        header_removed = self._remove_header_area(row_projection, height, min_density, main_density)
        if header_removed > 0:
            bottom_bound = min(bottom_bound, height - header_removed)
            self.logger.info(f"✓ ヘッダー領域を除去しました: {header_removed}px")
        
        return bottom_bound
    
    def _remove_header_area(self, row_projection, height, min_density, main_density):
        """ヘッダー領域の検出と除去"""
        # 上部20%の領域でヘッダーパターンを検索
        header_zone_end = int(height * 0.2)
        removed_pixels = 0
        
        # パターン1: 孤立した軽密度テキスト（タイトルなど）
        for y in range(header_zone_end):
            density = row_projection[y]
            
            if min_density < density < main_density:
                # 前後数行の密度をチェック
                surrounding_densities = []
                for check_y in range(max(0, y - 2), min(height, y + 3)):
                    if check_y != y:
                        surrounding_densities.append(row_projection[check_y])
                
                avg_surrounding = np.mean(surrounding_densities)
                
                # 周囲の密度が大幅に低い場合、ヘッダーとして判定
                if avg_surrounding < min_density * 0.5:
                    # このy位置以降で本文の開始を探す
                    for search_y in range(y + 1, int(height * 0.4)):
                        if row_projection[search_y] > main_density:
                            # 連続する本文が確認できた場合
                            consecutive = 0
                            for check_y in range(search_y, min(height, search_y + 5)):
                                if row_projection[check_y] > min_density:
                                    consecutive += 1
                            
                            if consecutive >= 3:
                                removed_pixels = search_y - y
                                self.logger.info(f"ヘッダーパターン検出: 行{y}-{search_y}")
                                return removed_pixels
        
        return 0
    
    def crop_image(self, pil_image, text_bounds):
        """
        画像をトリミング
        
        Args:
            pil_image: PIL画像
            text_bounds (dict): テキスト境界情報
            
        Returns:
            PIL.Image: トリミング済み画像
        """
        # マージンを適用
        left = max(0, text_bounds['left'] - self.min_margin)
        top = max(0, text_bounds['top'] - self.min_margin)
        right = min(pil_image.width - 1, text_bounds['right'] + self.min_margin)
        bottom = min(pil_image.height - 1, text_bounds['bottom'] + self.min_margin)
        
        # トリミング実行
        cropped = pil_image.crop((left, top, right + 1, bottom + 1))
        
        # 削減率計算
        original_area = pil_image.width * pil_image.height
        cropped_area = cropped.width * cropped.height
        reduction = ((original_area - cropped_area) / original_area) * 100
        
        self.logger.info(f"トリミング完了: {cropped.width}x{cropped.height} (削減率: {reduction:.1f}%)")
        
        return cropped
    
    def process_image(self, input_path, output_path):
        """
        単一画像の処理
        
        Args:
            input_path (str): 入力画像パス
            output_path (str): 出力画像パス
            
        Returns:
            bool: 処理成功フラグ
        """
        try:
            # デバッグ用に現在の画像パスを保存
            self._current_image_path = input_path
            
            # 画像読み込み
            result = self.load_image(input_path)
            if result is None:
                return False
            
            pil_image, cv_image, width, height = result
            
            # テキスト領域検出
            text_bounds = self.detect_text_bounds(cv_image, width, height)
            
            # 有効な領域が検出されたかチェック
            if text_bounds['width'] <= 0 or text_bounds['height'] <= 0:
                self.logger.warning(f"有効なテキスト領域が検出されませんでした: {os.path.basename(input_path)}")
                return False
            
            # トリミング実行
            cropped_image = self.crop_image(pil_image, text_bounds)
            
            # 保存
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cropped_image.save(output_path, quality=95, optimize=True)
            
            self.logger.info(f"保存完了: {os.path.basename(output_path)}")
            return True
            
        except Exception as e:
            self.logger.error(f"処理エラー: {os.path.basename(input_path)} - {str(e)}")
            return False
        finally:
            # クリーンアップ
            if hasattr(self, '_current_image_path'):
                delattr(self, '_current_image_path')


def get_image_files(directory):
    """
    ディレクトリから画像ファイルを取得
    
    Args:
        directory (str): 検索ディレクトリ
        
    Returns:
        list: 画像ファイルパスのリスト
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    return sorted(image_files)


def run_cli_mode(args):
    """コマンドライン版の実行"""
    print("=" * 60)
    print("簡素化版バッチコンテンツトリミング")
    print("1. 画像中央90%範囲でコンテンツ検出")
    print("2. 明度値ベースのシンプルな判定")
    print("3. 周囲白余白の自動除去")
    print("=" * 60)
    
    # 入力ディレクトリの存在確認
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
    
    # トリミング処理開始
    cropper = TextAreaCropper(
        threshold=args.threshold, 
        min_margin=args.margin,
        debug_mode=args.debug
    )
    
    successful = 0
    failed = 0
    start_time = time.time()
    
    # プログレスバー付きで処理
    for input_path in tqdm(image_files, desc="処理中"):
        try:
            # 出力パス生成
            if args.preserve_structure:
                # ディレクトリ構造を保持
                rel_path = os.path.relpath(input_path, args.input_dir)
                output_path = os.path.join(args.output_dir, rel_path)
            else:
                # 全て同一ディレクトリに保存
                filename = os.path.basename(input_path)
                output_path = os.path.join(args.output_dir, filename)
            
            # 既に存在する場合はスキップ
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
    print(f"平均処理時間: {elapsed_time/len(image_files):.2f}秒/ファイル")
    print(f"出力ディレクトリ: {args.output_dir}")


def run_config_mode():
    """設定ファイル版の実行"""
    print("=" * 60)
    print("簡素化版バッチコンテンツトリミング")
    print("1. 画像中央90%範囲でコンテンツ検出")
    print("2. 明度値ベースのシンプルな判定")
    print("3. 周囲白余白の自動除去")
    print("=" * 60)
    
    # デフォルト設定
    default_config = {
        "input_directory": "./images",
        "output_directory": "./cropped", 
        "settings": {
            "threshold": 120,
            "min_margin": 0,
            "preserve_directory_structure": True,
            "skip_existing_files": True
        }
    }
    
    config_path = "config.json"
    
    # 設定ファイルの作成または読み込み
    if not os.path.exists(config_path):
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=4, ensure_ascii=False)
        print(f"デフォルト設定ファイルを作成しました: {config_path}")
        print("config.json を編集してから再実行してください。")
        return
    
    # 設定読み込み
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        print(f"設定ファイル読み込みエラー: {e}")
        return
    
    # 設定確認
    input_dir = config["input_directory"]
    output_dir = config["output_directory"]
    
    if not os.path.exists(input_dir):
        print(f"エラー: 入力ディレクトリが存在しません: {input_dir}")
        return
    
    print(f"入力ディレクトリ: {input_dir}")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"閾値: {config['settings']['threshold']}")
    print(f"マージン: {config['settings']['min_margin']}")
    
    # 確認
    response = input("\n処理を開始しますか？ (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("処理をキャンセルしました。")
        return
    
    # 画像ファイル取得
    image_files = get_image_files(input_dir)
    if not image_files:
        print("画像ファイルが見つかりません。")
        return
    
    print(f"発見された画像ファイル: {len(image_files)}個")
    print("-" * 60)
    
    # 処理実行
    cropper = TextAreaCropper(
        threshold=config['settings']['threshold'],
        min_margin=config['settings']['min_margin'],
        debug_mode=config['settings'].get('debug_mode', False)
    )
    
    successful = 0
    failed = 0
    skipped = 0
    start_time = time.time()
    
    for input_path in tqdm(image_files, desc="処理中"):
        try:
            # 出力パス生成
            if config['settings']['preserve_directory_structure']:
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
            else:
                filename = os.path.basename(input_path)
                output_path = os.path.join(output_dir, filename)
            
            # 既存ファイルのスキップ
            if config['settings']['skip_existing_files'] and os.path.exists(output_path):
                skipped += 1
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
    print(f"スキップ: {skipped}個")
    print(f"処理時間: {elapsed_time:.2f}秒")
    print(f"出力先: {output_dir}")


def run_gui_mode():
    """GUI版の実行"""
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox, scrolledtext
        import threading
    except ImportError:
        print("エラー: tkinterライブラリが見つかりません。")
        print("GUI版を使用するには、tkinterがインストールされている必要があります。")
        return
    
    class BatchCropperGUI:
        """GUI版バッチトリミングアプリケーション"""
        
        def __init__(self, root):
            self.root = root
            self.root.title("バッチ画像トリミングツール v1.0")
            self.root.geometry("800x700")
            
            # 変数
            self.input_dir = tk.StringVar()
            self.output_dir = tk.StringVar(value="./cropped")
            self.threshold = tk.IntVar(value=120)
            self.margin = tk.IntVar(value=0)
            self.preserve_structure = tk.BooleanVar(value=True)
            self.skip_existing = tk.BooleanVar(value=True)
            
            # 処理状態
            self.is_processing = False
            self.cropper = None
            
            self.setup_ui()
        
        def setup_ui(self):
            """UI構築"""
            # メインフレーム
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # グリッド設定
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(0, weight=1)
            main_frame.columnconfigure(1, weight=1)
            main_frame.rowconfigure(7, weight=1)
            
            # タイトル
            title_label = ttk.Label(main_frame, text="🖼️ バッチ画像トリミングツール", 
                                   font=("Arial", 16, "bold"))
            title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
            
            # 入力ディレクトリ選択
            ttk.Label(main_frame, text="📁 入力ディレクトリ:").grid(row=1, column=0, sticky=tk.W, pady=5)
            ttk.Entry(main_frame, textvariable=self.input_dir, width=50).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
            ttk.Button(main_frame, text="参照", command=self.select_input_dir).grid(row=1, column=2, padx=(5, 0))
            
            # 出力ディレクトリ選択
            ttk.Label(main_frame, text="💾 出力ディレクトリ:").grid(row=2, column=0, sticky=tk.W, pady=5)
            ttk.Entry(main_frame, textvariable=self.output_dir, width=50).grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
            ttk.Button(main_frame, text="参照", command=self.select_output_dir).grid(row=2, column=2, padx=(5, 0))
            
            # 設定フレーム
            settings_frame = ttk.LabelFrame(main_frame, text="⚙️ トリミング設定", padding="10")
            settings_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
            settings_frame.columnconfigure(1, weight=1)
            
            # 閾値設定
            ttk.Label(settings_frame, text="検出感度 (50-200):").grid(row=0, column=0, sticky=tk.W, pady=2)
            threshold_frame = ttk.Frame(settings_frame)
            threshold_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
            threshold_scale = ttk.Scale(threshold_frame, from_=50, to=200, orient=tk.HORIZONTAL, 
                                       variable=self.threshold, length=200)
            threshold_scale.grid(row=0, column=0, sticky=(tk.W, tk.E))
            threshold_frame.columnconfigure(0, weight=1)
            ttk.Label(threshold_frame, textvariable=self.threshold).grid(row=0, column=1, padx=(10, 0))
            
            # マージン設定
            ttk.Label(settings_frame, text="マージン (0-10):").grid(row=1, column=0, sticky=tk.W, pady=2)
            margin_frame = ttk.Frame(settings_frame)
            margin_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
            margin_scale = ttk.Scale(margin_frame, from_=0, to=10, orient=tk.HORIZONTAL, 
                                    variable=self.margin, length=200)
            margin_scale.grid(row=0, column=0, sticky=(tk.W, tk.E))
            margin_frame.columnconfigure(0, weight=1)
            ttk.Label(margin_frame, textvariable=self.margin).grid(row=0, column=1, padx=(10, 0))
            
            # オプション設定
            options_frame = ttk.LabelFrame(main_frame, text="📋 処理オプション", padding="10")
            options_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
            
            ttk.Checkbutton(options_frame, text="ディレクトリ構造を保持", 
                           variable=self.preserve_structure).grid(row=0, column=0, sticky=tk.W)
            ttk.Checkbutton(options_frame, text="既存ファイルをスキップ", 
                           variable=self.skip_existing).grid(row=0, column=1, sticky=tk.W, padx=(20, 0))
            
            # 情報表示
            info_frame = ttk.LabelFrame(main_frame, text="📊 処理情報", padding="10")
            info_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
            info_frame.columnconfigure(1, weight=1)
            
            self.file_count_var = tk.StringVar(value="ファイル数: -")
            self.status_var = tk.StringVar(value="待機中...")
            
            ttk.Label(info_frame, textvariable=self.file_count_var).grid(row=0, column=0, sticky=tk.W)
            ttk.Label(info_frame, textvariable=self.status_var).grid(row=0, column=1, sticky=tk.W, padx=(20, 0))
            
            # ボタンフレーム
            button_frame = ttk.Frame(main_frame)
            button_frame.grid(row=6, column=0, columnspan=3, pady=20)
            
            self.scan_button = ttk.Button(button_frame, text="🔍 ファイルスキャン", 
                                         command=self.scan_files)
            self.scan_button.grid(row=0, column=0, padx=5)
            
            self.start_button = ttk.Button(button_frame, text="▶️ 処理開始", 
                                          command=self.start_processing, state=tk.DISABLED)
            self.start_button.grid(row=0, column=1, padx=5)
            
            self.stop_button = ttk.Button(button_frame, text="⏹️ 停止", 
                                         command=self.stop_processing, state=tk.DISABLED)
            self.stop_button.grid(row=0, column=2, padx=5)=2, padx=5)
            
            # プログレスバー
            self.progress = ttk.Progressbar(main_frame, mode='determinate')
            self.progress.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
            
            # ログ表示
            log_frame = ttk.LabelFrame(main_frame, text="📝 処理ログ", padding="5")
            log_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
            log_frame.columnconfigure(0, weight=1)
            log_frame.rowconfigure(0, weight=1)
            
            self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=80)
            self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # 変数更新時の処理
            self.input_dir.trace('w', self.on_input_change)
            
            self.log("アプリケーションが開始されました。")
            self.log("入力ディレクトリを選択してファイルスキャンを実行してください。")
        
        def select_input_dir(self):
            """入力ディレクトリ選択"""
            directory = filedialog.askdirectory(title="入力ディレクトリを選択")
            if directory:
                self.input_dir.set(directory)
        
        def select_output_dir(self):
            """出力ディレクトリ選択"""
            directory = filedialog.askdirectory(title="出力ディレクトリを選択")
            if directory:
                self.output_dir.set(directory)
        
        def on_input_change(self, *args):
            """入力ディレクトリ変更時の処理"""
            self.start_button.config(state=tk.DISABLED)
            self.file_count_var.set("ファイル数: -")
            self.status_var.set("スキャンが必要です")
        
        def scan_files(self):
            """ファイルスキャン実行"""
            input_path = self.input_dir.get()
            if not input_path:
                messagebox.showerror("エラー", "入力ディレクトリを選択してください。")
                return
            
            if not os.path.exists(input_path):
                messagebox.showerror("エラー", "指定されたディレクトリが存在しません。")
                return
            
            try:
                self.log("ファイルスキャンを開始...")
                image_files = get_image_files(input_path)
                
                if not image_files:
                    self.file_count_var.set("ファイル数: 0")
                    self.status_var.set("画像ファイルが見つかりません")
                    self.log("画像ファイルが見つかりませんでした。")
                    return
                
                self.image_files = image_files
                self.file_count_var.set(f"ファイル数: {len(image_files)}")
                self.status_var.set("処理準備完了")
                self.start_button.config(state=tk.NORMAL)
                
                self.log(f"スキャン完了: {len(image_files)}個の画像ファイルを発見")
                
                # サンプルファイル表示
                for i, file_path in enumerate(image_files[:5]):
                    self.log(f"  {i+1}. {os.path.basename(file_path)}")
                if len(image_files) > 5:
                    self.log(f"  ... 他 {len(image_files)-5} ファイル")
                    
            except Exception as e:
                messagebox.showerror("エラー", f"スキャン中にエラーが発生しました: {str(e)}")
                self.log(f"スキャンエラー: {str(e)}")
        
        def start_processing(self):
            """処理開始"""
            if not hasattr(self, 'image_files') or not self.image_files:
                messagebox.showerror("エラー", "まずファイルスキャンを実行してください。")
                return
            
            # 出力ディレクトリの確認
            output_path = self.output_dir.get()
            if not output_path:
                messagebox.showerror("エラー", "出力ディレクトリを指定してください。")
                return
            
            # 確認ダイアログ
            result = messagebox.askyesno("確認", 
                                       f"以下の設定で処理を開始しますか？\n\n"
                                       f"入力: {self.input_dir.get()}\n"
                                       f"出力: {output_path}\n"
                                       f"ファイル数: {len(self.image_files)}\n"
                                       f"閾値: {self.threshold.get()}\n"
                                       f"マージン: {self.margin.get()}")
            
            if not result:
                return
            
            # UI状態変更
            self.is_processing = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.scan_button.config(state=tk.DISABLED)
            
            # プログレスバー初期化
            self.progress.config(maximum=len(self.image_files), value=0)
            
            # 別スレッドで処理実行
            self.processing_thread = threading.Thread(target=self.process_images)
            self.processing_thread.daemon = True
            self.processing_thread.start()
        
        def stop_processing(self):
            """処理停止"""
            self.is_processing = False
            self.log("停止要求を受信しました...")
        
        def process_images(self):
            """画像処理メイン"""
            try:
                self.log("=" * 60)
                self.log("バッチトリミング処理を開始します")
                self.log(f"対象ファイル数: {len(self.image_files)}")
                self.log("=" * 60)
                
                # トリミング処理器作成
                self.cropper = TextAreaCropper(
                    threshold=self.threshold.get(),
                    min_margin=self.margin.get(),
                    debug_mode=False  # GUIでは通常デバッグオフ
                )
                
                successful = 0
                failed = 0
                skipped = 0
                start_time = time.time()
                
                for i, input_path in enumerate(self.image_files):
                    if not self.is_processing:
                        self.log("処理が中断されました。")
                        break
                    
                    try:
                        # 出力パス生成
                        if self.preserve_structure.get():
                            rel_path = os.path.relpath(input_path, self.input_dir.get())
                            output_path = os.path.join(self.output_dir.get(), rel_path)
                        else:
                            filename = os.path.basename(input_path)
                            output_path = os.path.join(self.output_dir.get(), filename)
                        
                        # 既存ファイルのスキップ
                        if self.skip_existing.get() and os.path.exists(output_path):
                            skipped += 1
                            self.log(f"スキップ: {os.path.basename(input_path)} (既存)")
                            self.update_progress(i + 1, f"スキップ中... ({i+1}/{len(self.image_files)})")
                            continue
                        
                        # 処理実行
                        self.update_progress(i + 1, f"処理中: {os.path.basename(input_path)}")
                        
                        if self.cropper.process_image(input_path, output_path):
                            successful += 1
                            self.log(f"✓ 完了: {os.path.basename(input_path)}")
                        else:
                            failed += 1
                            self.log(f"✗ 失敗: {os.path.basename(input_path)}")
                        
                    except Exception as e:
                        failed += 1
                        self.log(f"✗ エラー: {os.path.basename(input_path)} - {str(e)}")
                    
                    # UI更新
                    self.update_progress(i + 1, f"処理中... ({i+1}/{len(self.image_files)})")
                
                # 処理完了
                elapsed_time = time.time() - start_time
                self.log("=" * 60)
                self.log("処理完了")
                self.log(f"成功: {successful}個")
                self.log(f"失敗: {failed}個")
                self.log(f"スキップ: {skipped}個")
                self.log(f"処理時間: {elapsed_time:.2f}秒")
                if len(self.image_files) > 0:
                    self.log(f"平均処理時間: {elapsed_time/len(self.image_files):.2f}秒/ファイル")
                self.log(f"出力先: {self.output_dir.get()}")
                self.log("=" * 60)
                
                # 完了通知
                if self.is_processing:  # 中断されていない場合
                    messagebox.showinfo("完了", 
                                      f"処理が完了しました。\n\n"
                                      f"成功: {successful}個\n"
                                      f"失敗: {failed}個\n"
                                      f"スキップ: {skipped}個")
                
            except Exception as e:
                self.log(f"予期しないエラー: {str(e)}")
                messagebox.showerror("エラー", f"処理中にエラーが発生しました: {str(e)}")
            
            finally:
                # UI状態復元
                self.is_processing = False
                self.start_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
                self.scan_button.config(state=tk.NORMAL)
                self.update_progress(0, "待機中...")
        
        def update_progress(self, value, status):
            """プログレス更新"""
            self.root.after(0, lambda: self._update_progress_ui(value, status))
        
        def _update_progress_ui(self, value, status):
            """プログレスUI更新（メインスレッド）"""
            self.progress.config(value=value)
            self.status_var.set(status)
        
        def log(self, message):
            """ログ出力"""
            self.root.after(0, lambda: self._log_ui(message))
        
        def _log_ui(self, message):
            """ログUI更新（メインスレッド）"""
            timestamp = time.strftime("%H:%M:%S")
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)
        
        def select_input_dir(self):
            """入力ディレクトリ選択"""
            directory = filedialog.askdirectory(title="入力ディレクトリを選択")
            if directory:
                self.input_dir.set(directory)
        
        def select_output_dir(self):
            """出力ディレクトリ選択"""
            directory = filedialog.askdirectory(title="出力ディレクトリを選択")
            if directory:
                self.output_dir.set(directory)
        
        def on_input_change(self, *args):
            """入力ディレクトリ変更時の処理"""
            self.start_button.config(state=tk.DISABLED)
            self.file_count_var.set("ファイル数: -")
            self.status_var.set("スキャンが必要です")
        
        def scan_files(self):
            """ファイルスキャン実行"""
            input_path = self.input_dir.get()
            if not input_path:
                messagebox.showerror("エラー", "入力ディレクトリを選択してください。")
                return
            
            if not os.path.exists(input_path):
                messagebox.showerror("エラー", "指定されたディレクトリが存在しません。")
                return
            
            try:
                self.log("ファイルスキャンを開始...")
                image_files = get_image_files(input_path)
                
                if not image_files:
                    self.file_count_var.set("ファイル数: 0")
                    self.status_var.set("画像ファイルが見つかりません")
                    self.log("画像ファイルが見つかりませんでした。")
                    return
                
                self.image_files = image_files
                self.file_count_var.set(f"ファイル数: {len(image_files)}")
                self.status_var.set("処理準備完了")
                self.start_button.config(state=tk.NORMAL)
                
                self.log(f"スキャン完了: {len(image_files)}個の画像ファイルを発見")
                
                # サンプルファイル表示
                for i, file_path in enumerate(image_files[:5]):
                    self.log(f"  {i+1}. {os.path.basename(file_path)}")
                if len(image_files) > 5:
                    self.log(f"  ... 他 {len(image_files)-5} ファイル")
                    
            except Exception as e:
                messagebox.showerror("エラー", f"スキャン中にエラーが発生しました: {str(e)}")
                self.log(f"スキャンエラー: {str(e)}")
        
        def start_processing(self):
            """処理開始"""
            if not hasattr(self, 'image_files') or not self.image_files:
                messagebox.showerror("エラー", "まずファイルスキャンを実行してください。")
                return
            
            # 出力ディレクトリの確認
            output_path = self.output_dir.get()
            if not output_path:
                messagebox.showerror("エラー", "出力ディレクトリを指定してください。")
                return
            
            # 確認ダイアログ
            result = messagebox.askyesno("確認", 
                                       f"以下の設定で処理を開始しますか？\n\n"
                                       f"入力: {self.input_dir.get()}\n"
                                       f"出力: {output_path}\n"
                                       f"ファイル数: {len(self.image_files)}\n"
                                       f"閾値: {self.threshold.get()}\n"
                                       f"マージン: {self.margin.get()}")
            
            if not result:
                return
            
            # UI状態変更
            self.is_processing = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.scan_button.config(state=tk.DISABLED)
            
            # プログレスバー初期化
            self.progress.config(maximum=len(self.image_files), value=0)
            
            # 別スレッドで処理実行
            self.processing_thread = threading.Thread(target=self.process_images)
            self.processing_thread.daemon = True
            self.processing_thread.start()
        
        def stop_processing(self):
            """処理停止"""
            self.is_processing = False
            self.log("停止要求を受信しました...")
        
        def process_images(self):
            """画像処理メイン"""
            try:
                self.log("=" * 60)
                self.log("バッチトリミング処理を開始します")
                self.log(f"対象ファイル数: {len(self.image_files)}")
                self.log("=" * 60)
                
                # トリミング処理器作成
                self.cropper = TextAreaCropper(
                    threshold=self.threshold.get(),
                    min_margin=self.margin.get(),
                    debug_mode=False  # GUIでは通常デバッグオフ
                )
                
                successful = 0
                failed = 0
                skipped = 0
                start_time = time.time()
                
                for i, input_path in enumerate(self.image_files):
                    if not self.is_processing:
                        self.log("処理が中断されました。")
                        break
                    
                    try:
                        # 出力パス生成
                        if self.preserve_structure.get():
                            rel_path = os.path.relpath(input_path, self.input_dir.get())
                            output_path = os.path.join(self.output_dir.get(), rel_path)
                        else:
                            filename = os.path.basename(input_path)
                            output_path = os.path.join(self.output_dir.get(), filename)
                        
                        # 既存ファイルのスキップ
                        if self.skip_existing.get() and os.path.exists(output_path):
                            skipped += 1
                            self.log(f"スキップ: {os.path.basename(input_path)} (既存)")
                            self.update_progress(i + 1, f"スキップ中... ({i+1}/{len(self.image_files)})")
                            continue
                        
                        # 処理実行
                        self.update_progress(i + 1, f"処理中: {os.path.basename(input_path)}")
                        
                        if self.cropper.process_image(input_path, output_path):
                            successful += 1
                            self.log(f"✓ 完了: {os.path.basename(input_path)}")
                        else:
                            failed += 1
                            self.log(f"✗ 失敗: {os.path.basename(input_path)}")
                        
                    except Exception as e:
                        failed += 1
                        self.log(f"✗ エラー: {os.path.basename(input_path)} - {str(e)}")
                    
                    # UI更新
                    self.update_progress(i + 1, f"処理中... ({i+1}/{len(self.image_files)})")
                
                # 処理完了
                elapsed_time = time.time() - start_time
                self.log("=" * 60)
                self.log("処理完了")
                self.log(f"成功: {successful}個")
                self.log(f"失敗: {failed}個")
                self.log(f"スキップ: {skipped}個")
                self.log(f"処理時間: {elapsed_time:.2f}秒")
                if len(self.image_files) > 0:
                    self.log(f"平均処理時間: {elapsed_time/len(self.image_files):.2f}秒/ファイル")
                self.log(f"出力先: {self.output_dir.get()}")
                self.log("=" * 60)
                
                # 完了通知
                if self.is_processing:  # 中断されていない場合
                    messagebox.showinfo("完了", 
                                      f"処理が完了しました。\n\n"
                                      f"成功: {successful}個\n"
                                      f"失敗: {failed}個\n"
                                      f"スキップ: {skipped}個")
                
            except Exception as e:
                self.log(f"予期しないエラー: {str(e)}")
                messagebox.showerror("エラー", f"処理中にエラーが発生しました: {str(e)}")
            
            finally:
                # UI状態復元
                self.is_processing = False
                self.start_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
                self.scan_button.config(state=tk.NORMAL)
                self.update_progress(0, "待機中...")
        
        def update_progress(self, value, status):
            """プログレス更新"""
            self.root.after(0, lambda: self._update_progress_ui(value, status))
        
        def _update_progress_ui(self, value, status):
            """プログレスUI更新（メインスレッド）"""
            self.progress.config(value=value)
            self.status_var.set(status)
        
        def log(self, message):
            """ログ出力"""
            self.root.after(0, lambda: self._log_ui(message))
        
        def _log_ui(self, message):
            """ログUI更新（メインスレッド）"""
            timestamp = time.strftime("%H:%M:%S")
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)
    
    # GUI起動
    root = tk.Tk()
    app = BatchCropperGUI(root)
    
    # ウィンドウを中央に配置
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("アプリケーションが終了されました。")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='簡素化版バッチ画像トリミングプログラム',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # コマンドライン版
  python complete_batch_cropper.py --mode cli --input_dir "./images" --output_dir "./cropped"
  
  # 設定ファイル版
  python complete_batch_cropper.py --mode config
  
  # GUI版
  python complete_batch_cropper.py --mode gui
        """
    )
    
    parser.add_argument('--mode', choices=['cli', 'config', 'gui'], default='cli',
                       help='実行モード: cli(コマンドライン), config(設定ファイル), gui(GUI)')
    
    # CLI用オプション
    parser.add_argument('--input_dir', '-i', 
                       help='入力ディレクトリパス (cliモード用)')
    parser.add_argument('--output_dir', '-o', default='./cropped',
                       help='出力ディレクトリパス (デフォルト: ./cropped)')
    parser.add_argument('--threshold', '-t', type=int, default=120,
                       help='二値化閾値 (50-200, デフォルト: 120)')
    parser.add_argument('--margin', '-m', type=int, default=0,
                       help='最小マージン (0-10, デフォルト: 0)')
    parser.add_argument('--preserve_structure', '-p', action='store_true',
                       help='入力ディレクトリの構造を保持')
    parser.add_argument('--debug', '-d', action='store_true',
                       help='デバッグモード（詳細ログと可視化を有効化）')
    
    args = parser.parse_args()
    
    print("🖼️ 簡素化版バッチコンテンツトリミングプログラム v2.1")
    print("📋 新トリミング基準:")
    print("   1. 画像中央90%範囲でコンテンツ検出")
    print("   2. 明度値ベースのシンプルな判定")
    print("   3. 周囲白余白の自動除去")
    print("=" * 60)
    
    if args.mode == 'cli':
        if not args.input_dir:
            print("エラー: CLIモードでは --input_dir が必要です。")
            print("使用方法: python complete_batch_cropper.py --mode cli --input_dir ./images")
            return
        run_cli_mode(args)
    elif args.mode == 'config':
        run_config_mode()
    elif args.mode == 'gui':
        run_gui_mode()


if __name__ == "__main__":
    main()