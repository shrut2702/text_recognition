import os
from box.exceptions import BoxValueError
import yaml
from mlOCR import logger
import json
import joblib
import cv2
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any



@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data, filename:str, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=os.path.join(path,filename))
    logger.info(f"binary file {filename} saved at: {path}")


@ensure_annotations
def load_bin(path: Path):
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data



@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"



@ensure_annotations
def save_image(image, filename:str,path: Path):
    """Save an image to a specified path.

    Args:
        image (numpy.ndarray): Image data to be saved.
        path (Path): Path where the image should be saved.
    """
    cv2.imwrite(os.path.join(path,filename), image)
    logger.info(f"Image saved at: {path}")


@ensure_annotations
def save_text_file(text: str, filename: str, path: Path):
    """Save a text file to a specified path.

    Args:
        text (str): Text data to be saved.
        filename (str): Name of the text file.
        path (Path): Path where the text file should be saved.
    """
    with open(os.path.join(path, filename), 'w') as file:
        file.write(text)
    logger.info(f"Text file saved at: {path}")


@ensure_annotations
def load_text_file(path: Path) -> str:
    """Load a text file from a specified path.

    Args:
        filename (str): Name of the text file to load.
        path (Path): Path where the text file is located.

    Returns:
        str: Contents of the text file.
    """
    with open(path, 'r') as file:
        text = file.read()
    logger.info(f"Text file loaded from: {path}")
    return text