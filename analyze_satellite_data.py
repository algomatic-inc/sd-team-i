import os
import math
import requests
import numpy as np

from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

import overpy
import geopandas as gpd
from shapely.geometry import Polygon
from rasterio.transform import from_bounds
from rasterio.io import MemoryFile

# 定数
MERCATOR_RANGE = 256
CRS = "EPSG:4326"
MIN_AREA_SQUARE_METER = 40


# 投影関連のユーティリティ関数
def bound(value, opt_min, opt_max):
    if opt_min is not None:
        value = max(value, opt_min)
    if opt_max is not None:
        value = min(value, opt_max)
    return value


def degrees_to_radians(deg):
    return deg * (math.pi / 180)


def radians_to_degrees(rad):
    return rad / (math.pi / 180)


class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


class LatLng:
    def __init__(self, lat, lng):
        self.lat = lat
        self.lng = lng


class MercatorProjection:
    """
    緯度経度をピクセル座標に変換するためのメルカトル投影法クラス。
    """

    def __init__(self):
        self.pixel_origin = Point(MERCATOR_RANGE / 2, MERCATOR_RANGE / 2)
        self.pixels_per_lon_degree = MERCATOR_RANGE / 360
        self.pixels_per_lon_radian = MERCATOR_RANGE / (2 * math.pi)

    def from_lat_lng_to_point(self, lat_lng, opt_point=None):
        """
        緯度経度をピクセル座標に変換します。
        """
        point = opt_point if opt_point is not None else Point(0, 0)
        origin = self.pixel_origin
        point.x = origin.x + lat_lng.lng * self.pixels_per_lon_degree
        siny = bound(math.sin(degrees_to_radians(lat_lng.lat)), -0.9999, 0.9999)
        point.y = (
            origin.y
            + 0.5 * math.log((1 + siny) / (1 - siny)) * -self.pixels_per_lon_radian
        )
        return point

    def from_point_to_lat_lng(self, point):
        """
        ピクセル座標を緯度経度に変換します。
        """
        origin = self.pixel_origin
        lng = (point.x - origin.x) / self.pixels_per_lon_degree
        lat_radians = (point.y - origin.y) / -self.pixels_per_lon_radian
        lat = radians_to_degrees(2 * math.atan(math.exp(lat_radians)) - math.pi / 2)
        return LatLng(lat, lng)


def get_corners(center, zoom, map_width, map_height):
    """
    地図の中心点、ズームレベル、幅と高さから地図の四隅の緯度経度を取得します。
    """
    scale = 2**zoom
    proj = MercatorProjection()
    center_px = proj.from_lat_lng_to_point(center)
    sw_point = Point(
        center_px.x - (map_width / 2) / scale, center_px.y + (map_height / 2) / scale
    )
    sw_lat_lng = proj.from_point_to_lat_lng(sw_point)
    ne_point = Point(
        center_px.x + (map_width / 2) / scale, center_px.y - (map_height / 2) / scale
    )
    ne_lat_lng = proj.from_point_to_lat_lng(ne_point)
    return {
        "N": ne_lat_lng.lat,
        "E": ne_lat_lng.lng,
        "S": sw_lat_lng.lat,
        "W": sw_lat_lng.lng,
    }


def get_building_polygons(north: float, south: float, east: float, west: float, get_scale: float = 1.0) -> gpd.GeoDataFrame:
    api = overpy.Overpass()

    # Calculate the center of the bounding box
    center_lat = (north + south) / 2
    center_lon = (east + west) / 2

    # Adjust bounds based on the ratio
    _north = center_lat + (north - center_lat) * get_scale
    _south = center_lat + (south - center_lat) * get_scale
    _east = center_lon + (east - center_lon) * get_scale
    _west = center_lon + (west - center_lon) * get_scale


    # クエリの定義（引数を使って動的に範囲を設定）
    query = f"""
    [out:json][timeout:25];
    (
      way["building"]({_south},{_west},{_north},{_east});  
    );
    out body; 
    >;
    out skel qt;
    """
    
    result = api.query(query)


    polygons = []

    for way in result.ways:
        nodes = []
        for node in way.nodes:
            nodes.append((node.lon, node.lat))
        if len(nodes) > 2:  # 2点以上の時にポリゴンが成立するため
            polygons.append(Polygon(nodes))

    print(polygons)

    # GeoDataFrameに変換
    return gpd.GeoDataFrame(geometry=polygons, crs=CRS)


def get_static_map_image(center, zoom_level, img_size, google_api_key):
    static_map_url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={center.lat},{center.lng}&zoom={zoom_level}&size={img_size[0]}x{img_size[1]}&maptype=satellite&key={google_api_key}"
    )

    # Google Static Maps APIから画像を取得
    response = requests.get(static_map_url)
    if response.status_code != 200:
        response.raise_for_status()

    # 画像をPillowで開く
    map_image = Image.open(BytesIO(response.content)).convert("RGB")

    return map_image


def draw_polygons_on_image(geo_tiff_src, polygon_df: gpd.GeoDataFrame, line_width=5):
    """
    GeoTIFF画像に指定されたポリゴンを描画して画像を返します。
    """

    # GeoTIFFのデータを取得（RGBチャネル）
    img_array = geo_tiff_src.read([1, 2, 3])  # [R, G, B]
    img_array = np.transpose(img_array, (1, 2, 0))  # HWC形式に変換

    # PIL画像オブジェクトを作成
    img = Image.fromarray(img_array.astype("uint8"))
    img = img.convert("RGBA")

    # ポリゴンを描画するための新しいレイヤーを作成
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # フォント設定
    font_size = 22.0
    half_font_size = font_size/2.0
    font = ImageFont.load_default(size=font_size)  # デフォルトフォントを使用

    # ポリゴンをピクセル座標に変換して描画
    for _, row in polygon_df.iterrows():
        geom = row["geometry"]
        polygon_id = row["id"]  # GeoDataFrameのid列を使用

        if geom.is_valid:
            if geom.geom_type == "Polygon":
                pixel_coords = [
                    ~geo_tiff_src.transform * (x, y) for x, y in geom.exterior.coords
                ]
                draw.line(pixel_coords + [pixel_coords[0]], fill="red", width=line_width)
            elif geom.geom_type == "MultiPolygon":
                for poly in geom.geoms:
                    pixel_coords = [
                        ~geo_tiff_src.transform * (x, y) for x, y in poly.exterior.coords
                    ]
                    draw.line(pixel_coords + [pixel_coords[0]], fill="red", width=line_width)

            # 重心のピクセル座標を計算
            centroid_px = ~geo_tiff_src.transform * (geom.centroid.x, geom.centroid.y)

            bbox = draw.textbbox(centroid_px, str(polygon_id), font=font)
            bbox = (bbox[0] - half_font_size, bbox[1] - half_font_size, bbox[2] - half_font_size, bbox[3] - half_font_size)
            draw.rectangle(bbox, fill="white")

            # ラベルの描画（赤色）
            centroid_px = (centroid_px[0] - half_font_size, centroid_px[1] - half_font_size)
            draw.text(centroid_px, str(polygon_id), fill="red", font=font)

    # 画像をpolygonと合成
    img = Image.alpha_composite(img, overlay)

    byte_io = BytesIO()
    img.save(byte_io, format="PNG")
    byte_io.seek(0)
    return byte_io.getvalue()


def convert_polygon_to_pixel_coordinates(polygon_df, transform):
    """
    GeoDataFrameのポリゴンをピクセル座標に変換し、JSON形式で返します。
    """

    pixel_polygons = []

    for _, row in polygon_df.iterrows():
        geom = row["geometry"]
        polygon_id = row["id"]  # GeoDataFrameのid列を参照

        if geom.is_valid:
            if geom.geom_type == "Polygon":
                pixel_coords = [
                    {
                        "x": int((x - transform.c) / transform.a),
                        "y": int((y - transform.f) / transform.e),
                    }
                    for x, y in geom.exterior.coords
                ]
                pixel_polygons.append({"id": polygon_id, "coords": pixel_coords})
            elif geom.geom_type == "MultiPolygon":
                for poly in geom.geoms:
                    pixel_coords = [
                        {
                            "x": int((x - transform.c) / transform.a),
                            "y": int((y - transform.f) / transform.e),
                        }
                        for x, y in poly.exterior.coords
                    ]
                    pixel_polygons.append({"id": polygon_id, "coords": pixel_coords})

    return pixel_polygons


def filter_small_buildings(gdf: gpd.GeoDataFrame, min_area=MIN_AREA_SQUARE_METER) -> gpd.GeoDataFrame:
    print(gdf)
    utm_crs = gdf.estimate_utm_crs() 
    projected_gdf = gdf.to_crs(utm_crs)

    # 面積を計算してフィルタリング
    projected_gdf["area"] = projected_gdf["geometry"].area  # 面積は平方メートル単位
    filtered_gdf = projected_gdf.dropna(subset=["area"])
    filtered_gdf = filtered_gdf[filtered_gdf["area"] >= min_area].copy()
    print(filtered_gdf)

    # 元のCRSに戻して返す
    return filtered_gdf.to_crs(gdf.crs)


def analyze(center_lat: float, center_lon: float, img_size=(640, 640), zoom_level=19, osm_scale=0.5):
    google_api_key = os.getenv("GOOGLE_API_KEY")

    width, height = img_size

    center = LatLng(center_lat, center_lon)

    map_image = get_static_map_image(center, zoom_level, img_size, google_api_key)

    map_array = np.array(map_image)

    bounds = get_corners(center, zoom_level, width, height)

    transform = from_bounds(
        bounds["W"], bounds["S"], bounds["E"], bounds["N"], width, height
    )

    # GeoTIFFをメモリに保存
    memfile = MemoryFile()
    with memfile.open(
        driver="GTiff",
        height=height,
        width=width,
        count=3,  # RGBチャネル
        dtype=map_array.dtype,
        crs=CRS,
        transform=transform,
    ) as dst:
        # RGBチャネルごとに書き込む
        dst.write(map_array[:, :, 0], 1)
        dst.write(map_array[:, :, 1], 2)
        dst.write(map_array[:, :, 2], 3)

    geo_tiff_data = memfile.open()  # メモリ内のGeoTIFFを開く

    # 取得範囲を0~1で指定
    scale = osm_scale
    polygon_df = get_building_polygons(bounds['N'], bounds['S'], bounds['E'], bounds['W'],scale)

    if len(polygon_df) == 0:
        raise Exception("polygon is not found")

    # 閾値以上の面積をもつPolygonのみ取得
    polygon_df = filter_small_buildings(polygon_df,MIN_AREA_SQUARE_METER)

    # 後の処理でid列が必要な為追加
    polygon_df["id"] = range(1, len(polygon_df) + 1)

    combined_img_byte = draw_polygons_on_image(geo_tiff_data, polygon_df, line_width=5)

    pixel_polygon_json = convert_polygon_to_pixel_coordinates(polygon_df, transform)

    # 衛星データを画像バイナリデータに変換
    buffer = BytesIO()
    image = Image.fromarray(map_array)
    image.save(buffer, format="PNG")
    img_byte = buffer.getvalue()

    return img_byte, combined_img_byte, pixel_polygon_json