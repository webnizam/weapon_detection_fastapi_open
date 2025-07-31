from PIL import Image
import io
import pandas as pd
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator


# Initialize the models with proper PyTorch loading configuration
def load_yolo_model(model_path: str):
    """
    Load YOLO model with proper handling for PyTorch 2.6+ weights_only behavior
    """
    # Store original torch.load function
    original_torch_load = torch.load
    
    try:
        # Temporarily patch torch.load to use weights_only=False
        def patched_torch_load(f, map_location=None, pickle_module=None, **pickle_load_args):
            # Remove weights_only if it's in pickle_load_args to avoid conflicts
            pickle_load_args.pop('weights_only', None)
            return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, weights_only=False, **pickle_load_args)
        
        # Apply the patch
        torch.load = patched_torch_load
        
        # Load the model
        model = YOLO(model_path)
        
        return model
        
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        raise e
    finally:
        # Always restore the original torch.load function
        torch.load = original_torch_load

loaded_model = load_yolo_model("./models/11_02_23_10_01.pt")


def get_image_from_bytes(binary_image: bytes) -> Image:
    """Convert image from bytes to PIL RGB format
    
    Args:
        binary_image (bytes): The binary representation of the image
    
    Returns:
        PIL.Image: The image in PIL RGB format
    """
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    return input_image


def get_bytes_from_image(image: Image) -> bytes:
    """
    Convert PIL image to Bytes
    
    Args:
    image (Image): A PIL image instance
    
    Returns:
    bytes : BytesIO object that contains the image in JPEG format with quality 85
    """
    return_image = io.BytesIO()
    image.save(return_image, format='JPEG', quality=85)  # save the image in JPEG format with quality 85
    return_image.seek(0)  # set the pointer to the beginning of the file
    return return_image

def transform_predict_to_df(results: list, labeles_dict: dict) -> pd.DataFrame:
    """
    Transform predict from yolov8 (torch.Tensor) to pandas DataFrame.

    Args:
        results (list): A list containing the predict output from yolov8 in the form of a torch.Tensor.
        labeles_dict (dict): A dictionary containing the labels names, where the keys are the class ids and the values are the label names.
        
    Returns:
        predict_bbox (pd.DataFrame): A DataFrame containing the bounding box coordinates, confidence scores and class labels.
    """
    # Transform the Tensor to numpy array
    predict_bbox = pd.DataFrame(results[0].to("cpu").numpy().boxes.xyxy, columns=['xmin', 'ymin', 'xmax','ymax'])
    # Add the confidence of the prediction to the DataFrame
    predict_bbox['confidence'] = results[0].to("cpu").numpy().boxes.conf
    # Add the class of the prediction to the DataFrame
    predict_bbox['class'] = (results[0].to("cpu").numpy().boxes.cls).astype(int)
    # Replace the class number with the class name from the labeles_dict
    predict_bbox['name'] = predict_bbox["class"].replace(labeles_dict)
    return predict_bbox

def get_model_predict(model: YOLO, input_image: Image, save: bool = False, image_size: int = 1248, conf: float = 0.5, augment: bool = False) -> pd.DataFrame:
    """
    Get the predictions of a model on an input image.
    
    Args:
        model (YOLOV8): The trained YOLO model.
        input_image (Image): The image on which the model will make predictions.
        save (bool, optional): Whether to save the image with the predictions. Defaults to False.
        image_size (int, optional): The size of the image the model will receive. Defaults to 1248.
        conf (float, optional): The confidence threshold for the predictions. Defaults to 0.5.
        augment (bool, optional): Whether to apply data augmentation on the input image. Defaults to False.
    
    Returns:
        pd.DataFrame: A DataFrame containing the predictions.
    """
    # Make predictions
    predictions = model.predict(
                        imgsz=image_size, 
                        source=input_image, 
                        conf=conf,
                        save=save, 
                        augment=augment,
                        flipud= 0.0,
                        fliplr= 0.0,
                        mosaic = 0.0,
                        )
    
    # Transform predictions to pandas dataframe
    predictions = transform_predict_to_df(predictions, model.model.names)
    return predictions


################################# BBOX Func #####################################


def add_bboxs_on_img(image: Image, predict: pd.DataFrame) -> Image:
    """
    add a bounding box on the image

    Args:
    image (Image): input image
    predict (pd.DataFrame): predict from model

    Returns:
    Image: image whis bboxs
    """
    # Create an annotator object
    annotator = Annotator(np.array(image))

    # sort predict by xmin value
    predict = predict.sort_values(by=['xmin'], ascending=True)

    # iterate over the rows of predict dataframe
    for i, row in predict.iterrows():
        # create the text to be displayed on image
        text = f"{row['name']}: {int(row['confidence']*100)}%"
        # get the bounding box coordinates
        bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        # add the bounding box and text on the image
        print(str(row['class']))
        annotator.box_label(bbox, text, color=(255, 0, 0), txt_color=(255, 255, 255))
    # convert the annotated image to PIL image
    return Image.fromarray(annotator.result())


################################# Model Prediction #####################################


def detect_trained_model(input_image: Image) -> pd.DataFrame:
    """
    Predict from trained weapon detection model.
    Base on YoloV8

    Args:
        input_image (Image): The input image.

    Returns:
        pd.DataFrame: DataFrame containing the object location.
    """
    predict = get_model_predict(
        model=loaded_model,
        input_image=input_image,
        save=False,
        image_size=640,
        augment=False,
        conf=0.50,
    )
    return predict
