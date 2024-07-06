import cv2
import numpy as np
from deepfill_v2 import DeepFill
import torch
from configs import model_path, deepfill_model_path
from pathlib import Path
import tempfile
import os

class ObjectReplacer:
    def __init__(self, model_name=model_path, deepfill_model_path=deepfill_model_path):
        # 加載YOLOv5模型
        if Path(model_name).is_file():
            self.model = torch.hub.load("ultralytics/yolov5", 'custom', path=model_name)
        else:
            self.model = torch.hub.load("ultralytics/yolov5", model_name)

        # 加載DeepFill模型
        self.deepfill = DeepFill(deepfill_model_path)

    def load_image(self, image_path):
        # 讀取輸入圖片
        self.input_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if self.input_image is None:
            raise ValueError(f"無法加載位於 {image_path} 的圖片。")
        # 確保輸入圖像有透明通道
        if self.input_image.shape[2] == 3:
            self.input_image = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2BGRA)

    def detect_objects(self):
        # 進行物體檢測
        results = self.model(self.input_image)
        return results.pandas().xyxy[0]

    def replace_object(self, target_label, background_path, target_image_path, output_image_path=None, expand_pixels=10, scale_factor=1.0):
        detections = self.detect_objects()
        if target_label in detections['name'].values:
            target_index = detections[detections['name'] == target_label].index[0]
            xmin, ymin, xmax, ymax = [int(detections.loc[target_index, col]) for col in ['xmin', 'ymin', 'xmax', 'ymax']]
            
            # 擴展邊界框並確保不超出圖像邊界
            xmin = max(0, xmin - expand_pixels)
            ymin = max(0, ymin - expand_pixels)
            xmax = min(self.input_image.shape[1], xmax + expand_pixels)
            ymax = min(self.input_image.shape[0], ymax + expand_pixels)

            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as mask_file, \
                 tempfile.NamedTemporaryFile(delete=False, suffix='.png') as repaired_image_file:
                mask_path = mask_file.name
                repaired_image_path = repaired_image_file.name

                # 生成掩碼
                mask = np.zeros(self.input_image.shape[:2], dtype=np.uint8)
                mask[ymin:ymax, xmin:xmax] = 255
                cv2.imwrite(mask_path, mask)

                # 使用DeepFill進行圖像修復
                self.deepfill.inpaint(background_path, mask_path, repaired_image_path)

                # 讀取修復後的圖像
                repaired_image = cv2.imread(repaired_image_path, cv2.IMREAD_UNCHANGED)
                if repaired_image.shape[2] != 4:
                    repaired_image = cv2.cvtColor(repaired_image, cv2.COLOR_BGR2BGRA)

                # 讀取並調整目標圖片大小
                target_image = cv2.imread(target_image_path, cv2.IMREAD_UNCHANGED)
                if target_image is None:
                    raise ValueError(f"無法加載位於 {target_image_path} 的目標圖片。")
                
                # 計算縮放比例，保持目標圖片的比例，並應用縮放因子
                target_h, target_w = target_image.shape[:2]
                bbox_w, bbox_h = xmax - xmin, ymax - ymin
                scale = max(bbox_w / target_w, bbox_h / target_h) * scale_factor
                new_w, new_h = int(target_w * scale), int(target_h * scale)

                # 縮放目標圖像和alpha通道
                target_image_resized = cv2.resize(target_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # 計算放置位置，使得目標圖像中心對齊
                center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2
                top_left_x = center_x - new_w // 2
                top_left_y = center_y - new_h // 2
                bottom_right_x = top_left_x + new_w
                bottom_right_y = top_left_y + new_h

                # 處理邊界情況
                target_start_x = max(0, top_left_x)
                target_start_y = max(0, top_left_y)
                target_end_x = min(repaired_image.shape[1], bottom_right_x)
                target_end_y = min(repaired_image.shape[0], bottom_right_y)

                overlay_start_x = max(0, -top_left_x)
                overlay_start_y = max(0, -top_left_y)
                overlay_end_x = new_w - max(0, bottom_right_x - repaired_image.shape[1])
                overlay_end_y = new_h - max(0, bottom_right_y - repaired_image.shape[0])

                # 將目標圖片放置到修復後的背景圖上
                alpha_mask = target_image_resized[overlay_start_y:overlay_end_y, overlay_start_x:overlay_end_x, 3] / 255.0
                for c in range(3):  # BGR通道
                    repaired_image[target_start_y:target_end_y, target_start_x:target_end_x, c] = alpha_mask * target_image_resized[overlay_start_y:overlay_end_y, overlay_start_x:overlay_end_x, c] + \
                                                                                                  (1 - alpha_mask) * repaired_image[target_start_y:target_end_y, target_start_x:target_end_x, c]

                # 更新修復圖像的alpha通道
                repaired_alpha = repaired_image[:, :, 3]
                repaired_alpha[target_start_y:target_end_y, target_start_x:target_end_x] = np.maximum(repaired_alpha[target_start_y:target_end_y, target_start_x:target_end_x], 
                                                                                                      target_image_resized[overlay_start_y:overlay_end_y, overlay_start_x:overlay_end_x, 3])
                repaired_image[:, :, 3] = repaired_alpha

                if output_image_path:
                    cv2.imwrite(output_image_path, repaired_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                    print(f"輸出圖像已保存至 {output_image_path}")

            # 刪除臨時文件
            os.remove(mask_path)
            os.remove(repaired_image_path)

            return True

        else:
            print(f"在圖片中未找到 {target_label}。")
            return False

def check_process(target_label, background_path, target_image_path, output_image_path, scale):
    replacer = ObjectReplacer()
    replacer.load_image(background_path)
    result = replacer.replace_object(target_label, background_path, target_image_path, output_image_path, expand_pixels=10, scale_factor=scale)
    return result    

if __name__ == "__main__":
    scale=1.2
    result = check_process(target_label="test_bottle", background_path="test_back", target_image_path="product_test/product_transparent.png", output_image_path="test_result.png", scale=scale)
    if result:
        print("處理成功")
    else:
        print("處理失敗")
