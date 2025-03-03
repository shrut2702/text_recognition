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
class CraftModelConfig:
    craft_weights: Path
    refiner_wieghts: Path