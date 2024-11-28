import os
from os.path import join
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import common_fn as cmn
import analyze_satellite_data
import llm


# 環境変数読み込み
base_path = os.path.abspath("")
dotenv_path = join(base_path, ".env")
load_dotenv(dotenv_path)


# ------------------------------------------------------------
# Input/Output
class AnalyzeInputParams(BaseModel):
    lat: float  # 緯度
    lon: float  # 経度


class AnalyzeOutputParams(BaseModel):
    image: str
    mark_image: str


class GenerateInputParams(BaseModel):
    image: str
    mark_image: str


class GenerateOutputParams(BaseModel):
    stepbystep: list
    result: list


# ------------------------------------------------------------
# FastAPI
app = FastAPI()


@app.get("/hc")
async def health_check():
    return {"message": "ok"}


@app.post("/analyze")
async def analyze(params: AnalyzeInputParams):
    try :
        img_byte, combined_img_byte, _ = analyze_satellite_data.analyze(
            center_lat=params.lat,
            center_lon=params.lon,
            zoom_level=20, #10~15s 旧:48s
            # zoom_level=21, #旧:5~10s
            osm_scale=0.5,
        )
    except:
        raise HTTPException(
            status_code=404,
            detail="Polygon data could not be found for the specified location. Please check the coordinates or try a different area."
        )
    return AnalyzeOutputParams(
        image=cmn.bytes_to_base64(img_byte),
        mark_image=cmn.bytes_to_base64(combined_img_byte),
    )


@app.post("/generate")
async def generate(params: GenerateInputParams):
    json_obj = llm.generate_classification(params.image, params.mark_image)
    return GenerateOutputParams(
        stepbystep=json_obj["stepbystep"],
        result=json_obj["result"],
    )

@app.get("/generate_sample")
async def generate_sample():
    result = llm.generate_sample("こんにちは、あなたは何が出来ますか？")
    return {"text": result}
