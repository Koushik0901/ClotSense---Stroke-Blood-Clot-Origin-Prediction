import warnings
import gc
import numpy as np
import torch
import torchvision.transforms as T
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore")


labels = ["LAA", "CE"]


class InferenceEngine:
    def __init__(
        self,
        model_path="source/checkpoints/optimized_scripted_lite.ptl",
        bg_model_path="source/checkpoints/bg_classifier_optimized_scripted.pt",
        img_size=256,
        bg_img_size=128,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path).eval().to(self.device)
        self.bg_model = torch.jit.load(bg_model_path).eval().to(self.device)
        self.main_transforms = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.bg_transforms = T.Compose(
            [
                T.Resize((bg_img_size, bg_img_size)),
                T.ToTensor(),
            ]
        )

    @torch.no_grad()
    def predict_image(self, image_path):
        results = []
        slide = open_slide(image_path)
        tiles = DeepZoomGenerator(slide, tile_size=600, overlap=0, limit_bounds=False)
        level = tiles.level_count - 1
        col, row = tiles.level_tiles[level]
        for row in tqdm(range(row), unit="row"):
            bg_inputs, inputs = [], []
            for col in range(col):
                temp_tile = tiles.get_tile(level, (col, row))
                rgb = temp_tile.convert("RGB")
                bg_inputs.append(self.bg_transforms(rgb))
                inputs.append(self.main_transforms(rgb))
            if len(bg_inputs) != 0:
                bg_inputs = torch.stack(bg_inputs).to(self.device)
                inputs = torch.stack(inputs).to(self.device)
                bg_preds = self.bg_model(bg_inputs).squeeze().sigmoid().round()
                cell_img_indices = torch.where(bg_preds == 1)
                inp_data = inputs[cell_img_indices]
                if len(inp_data) != 0:
                    logits = self.model(inp_data)
                    preds = torch.softmax(logits, dim=-1).argmax(dim=-1)
                    try:
                        results.extend(preds.tolist())
                    except TypeError as e:
                        results.append(preds)

        class_idx = int(np.mean(results).round())
        return labels[class_idx]


if __name__ == "__main__":
    image_path = "sample data/008e5c_0_CE.tif"
    engine = InferenceEngine()
    print(engine.predict_image(image_path))
