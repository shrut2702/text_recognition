from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DemoImageConfig:
    image_url: str
    image_dir: Path

@dataclass(frozen=True)
class ImageProcessingConfig:
    image_path: Path
    result_path: Path
    norm_mean: tuple
    norm_variance: tuple
    canvas_size: int
    mag_ratio:float

@dataclass(frozen=True)
class TextDetectionConfig:
    craft_weights: Path
    refiner_weights: Path
    loaded_image_path: Path
    normalized_image_path: Path
    resized_data_path: Path
    text_threshold: float
    low_text: float
    link_threshold: float
    poly: bool
    refine: bool

@dataclass(frozen=True)
class TextRecognitionConfig:
    crnn_weights_digital: Path
    crnn_weights_handwritten: Path
    crnn_input_path: Path
    crnn_output_path: Path
    resize_canvas: tuple
    sorting_threshold: float
    char_list: list

@dataclass(frozen=True)
class TextPostProcessingConfig:
    text_input_path: Path