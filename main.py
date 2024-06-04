####################################### IMPORT #################################
import json
import pandas as pd
from PIL import Image
from loguru import logger
import sys

from fastapi import FastAPI, File, status
from fastapi.responses import RedirectResponse
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException

from io import BytesIO

from app import get_image_from_bytes
from app import detect_trained_model
from app import add_bboxs_on_img
from app import get_bytes_from_image

####################################### logger #################################

logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level=10,
)
logger.add("log.log", rotation="1 MB", level="DEBUG", compression="zip")

###################### FastAPI Setup #############################

# title
app = FastAPI(
    title="Weapons Detection Falcons.ai",
    description="""Fine-tuned YoloV8 model for weapons detection by Falcons.ai""",
    version="2023.10.29",
)

# This function is needed if you want to allow client requests 
# from specific domains (specified in the origins argument) 
# to access resources from the FastAPI server, 
# and the client and server are hosted on different domains.
origins = [
    "http://localhost",
    "http://localhost:8008",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# redirect
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")




######################### Support Func #################################

def crop_image_by_predict(image: Image, predict: pd.DataFrame(), crop_class_name: str,) -> Image:
    """Crop an image based on the detection of a certain object in the image.
    
    Args:
        image: Image to be cropped.
        predict (pd.DataFrame): Dataframe containing the prediction results of object detection model.
        crop_class_name (str, optional): The name of the object class to crop the image by. if not provided, function returns the first object found in the image.
    
    Returns:
        Image: Cropped image or None
    """
    crop_predicts = predict[(predict['name'] == crop_class_name)]

    if crop_predicts.empty:
        raise HTTPException(status_code=400, detail=f"{crop_class_name} not found in photo")

    # if there are several detections, choose the one with more confidence
    if len(crop_predicts) > 1:
        crop_predicts = crop_predicts.sort_values(by=['confidence'], ascending=False)

    crop_bbox = crop_predicts[['xmin', 'ymin', 'xmax','ymax']].iloc[0].values
    # crop
    img_crop = image.crop(crop_bbox)
    return(img_crop)


######################### MAIN Func #################################


@app.post("/detect_weapon")
def img_object_detection_to_img(file: bytes = File(...)):
    """
    Weapon Detection from an image plot bbox on image

    Returns:
        Image: Image in bytes with bbox annotations.
    """
    # get image from bytes
    input_image = get_image_from_bytes(file)

    # model predict
    predict = detect_trained_model(input_image)

    # add bbox on image
    final_image = add_bboxs_on_img(image = input_image, predict = predict)

    # return image in bytes format
    return StreamingResponse(content=get_bytes_from_image(final_image), media_type="image/jpeg")

