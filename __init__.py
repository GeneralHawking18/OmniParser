import os
import json
import time
import torch
import gc
import importlib
from PIL import Image
from datasets import load_dataset
# from ultralytics import YOLO
from PIL import Image

from util.utils import (
    get_som_labeled_img,
    check_ocr_box,
    get_caption_model_processor,
    get_yolo_model
)


def retrieve_data(folder = "./screenshots"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    ds = load_dataset("HuggingFaceM4/WebSight", "v0.2", streaming=True, split="train")
    cnt = 0
    for i in ds:
        file_path = os.path.join(folder, f'{cnt}.png')
        if os.path.exists(file_path):
            print(f"File {file_path} existed.")
        else:
            i['image'].save(file_path)
            print(f"Saved {file_path}")
        cnt += 1
        if cnt == 19:
            break

def load_all_images(folder="screenshots"):
    image_paths = []
    for filename in sorted(os.listdir(folder)):  # Sắp xếp để giữ thứ tự ảnh
        if filename.endswith(".png"):  # Chỉ lấy file PNG
            image_path = os.path.join(folder, filename)
            image_paths.append(image_path)
    return image_paths


class ModelLoader:
    def __init__(self, detect_model_path='weights/icon_detect/model.pt', caption_model_path = "weights/icon_caption_florence", caption_model_name = "florence2"):
        self.detect_model_path = detect_model_path
        self.caption_model_name = caption_model_name
        self.caption_model_path = caption_model_path


    def load_detect_model(self, device):
        """Load model detect and move to the specified device."""
        model = get_yolo_model(self.detect_model_path)
        model.to(device)
        print(f'Detect Model moved to {device}')
        return model

    def load_caption_processor(self, device):
        """Initialize caption model processor with Florence2 model."""
        processor = get_caption_model_processor(
            model_name=self.caption_model_name,
            model_name_or_path=self.caption_model_path,
            device=device
        )
        print(f'Caption Processor loaded for device: {device}')
        return processor


class ImageProcessor:
    def __init__(self, box_threshold=0.05):
        self.box_threshold = box_threshold

    def load_image(self, image_path):
        """Open image, convert to RGB, and get filename (without extension)."""
        image = Image.open(image_path)
        image_rgb = image.convert('RGB')
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        print('Image size:', image.size)
        return image, image_rgb, image_name

    def get_draw_bbox_config(self, image_size):
        """Calculate overlay ratio and config for drawing bounding boxes."""
        box_overlay_ratio = max(image_size) / 3200
        config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }
        return config

    def process_ocr(self, image_path):
        """Process OCR and return results with processing time."""
        start_time = time.time()
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
            image_path,
            display_img=False,
            output_bb_format='xyxy',
            goal_filtering=None,
            easyocr_args={'paragraph': False, 'text_threshold': 0.9},
            use_paddleocr=True
        )
        text, ocr_bbox = ocr_bbox_rslt
        ocr_time = time.time() - start_time
        print(f'OCR processing time: {ocr_time:.2f} seconds')
        return text, ocr_bbox

    def process_caption(self, image_path, som_model, draw_bbox_config, caption_model_processor, ocr_bbox, text):
        """Generate labeled image and content list with caption processing time."""
        start_time = time.time()
        dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image_path,
            som_model,
            BOX_TRESHOLD=self.box_threshold,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=caption_model_processor,
            ocr_text=text,
            use_local_semantics=True,
            iou_threshold=0.7,
            scale_img=False,
            batch_size=128
        )
        caption_time = time.time() - start_time
        print(f'Caption processing time: {caption_time:.2f} seconds')
        return parsed_content_list


class DataExporter:
    def export_to_json(self, data, output_path):
        """Save data to a JSON file at the output path."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        print(f'Results saved to {output_path}')

    def convert_parsed_to_uied(self, parsed_content_list, img_shape):
        """
        Convert data from parsed_content_list format to uied format.

        Parameters:
            parsed_content_list (list): List of objects in parsed_content_list format.
            img_shape (list): Image shape as [height, width, channels].

        Returns:
            dict: Data in uied format.
        """
        img_height, img_width, _ = img_shape
        uied_compos = []

        for idx, element in enumerate(parsed_content_list):
            bbox = element.get("bbox", [0, 0, 0, 0])
            if len(bbox) != 4:
                continue
            x_min, y_min, x_max, y_max = bbox

            col_min = int(round(x_min * img_width))
            row_min = int(round(y_min * img_height))
            col_max = int(round(x_max * img_width))
            row_max = int(round(y_max * img_height))

            width = col_max - col_min
            height = row_max - row_min

            element_type = element.get("type", "").lower()
            compo_class = "Text" if element_type == "text" else "Compo"

            uied_element = {
                "id": idx,
                "class": compo_class,
                "height": height,
                "width": width,
                "position": {
                    "column_min": col_min,
                    "row_min": row_min,
                    "column_max": col_max,
                    "row_max": row_max
                }
            }
            if compo_class == "Text":
                uied_element["text_content"] = element.get("content", "")

            uied_compos.append(uied_element)

        uied_data = {
            "compos": uied_compos,
            "img_shape": img_shape
        }
        return uied_data


class OmniGUIParser:
    def __init__(self,
            detect_model_path='weights/icon_detect/model.pt', 
            caption_model_path ="weights/icon_caption_florence", 
            caption_model_name="florence2", device='cuda'
    ):
        self.device = device
        self.model_loader = ModelLoader(detect_model_path, caption_model_path, caption_model_name)
        self.image_processor = ImageProcessor()
        self.data_exporter = DataExporter()

        self.load_models()

    def load_models(self):
        """Load YOLO and Caption models."""
        self.som_model = self.model_loader.load_detect_model(self.device)
        self.caption_model_processor = self.model_loader.load_caption_processor(self.device)

    def process_image(self, image_path, image_uied_dir_output="./output/uied", image_omni_dir_output="./output/omni"):
        """Process a single image through OCR, captioning, and export results."""
        image, image_rgb, image_name = self.image_processor.load_image(image_path)
        img_shape = [image_rgb.height, image_rgb.width, 3]  # Correctly get shape from image_rgb
        draw_bbox_config = self.image_processor.get_draw_bbox_config(image.size)

        text, ocr_bbox = self.image_processor.process_ocr(image_path)
        parsed_content_list = self.image_processor.process_caption(
            image_path,
            self.som_model,
            draw_bbox_config,
            self.caption_model_processor,
            ocr_bbox,
            text
        )

        output_omni_parsed_path = os.path.join(image_omni_dir_output, f'{image_name}.json')
        self.data_exporter.export_to_json(parsed_content_list, output_omni_parsed_path)

        uied_data = self.data_exporter.convert_parsed_to_uied(parsed_content_list, img_shape)
        output_uied_parsed_path = os.path.join(image_uied_dir_output, f'{image_name}.json')
        self.data_exporter.export_to_json(uied_data, output_uied_parsed_path)

        if hasattr(image, "close"):
            image.close()
        del image, image_rgb, draw_bbox_config, text, ocr_bbox, parsed_content_list, uied_data
        gc.collect()


    def process_images(self, image_dir):
        """Process all images in the specified directory."""
        image_paths = load_all_images(image_dir) # Use the standalone function
        for image_path in image_paths:
            self.process_image(image_path)



if __name__ == "__main__":
    device = 'cuda'
    image_dir = "./screenshots"
    yolo_model_path = 'weights/icon_detect/model.pt'

    # retrieve_data(image_dir) # Keep retrieve_data as standalone if needed

    gui_analyzer = OmniGUIParser(yolo_model_path, device=device)
    gui_analyzer.process_images(image_dir)