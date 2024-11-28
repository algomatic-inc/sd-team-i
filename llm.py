import os
import json

import vertexai
from vertexai.preview.generative_models import (
    GenerativeModel,
    GenerationConfig,
    Part,
    Content,
)
from google.oauth2 import service_account


def credentials():
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]

    return service_account.Credentials.from_service_account_info(
        {
            "type": "service_account",
            "project_id": os.getenv("GEMINI_PROJECT_ID"),
            "private_key_id": os.getenv("GEMINI_PRIVATE_KEY_ID"),
            "private_key": os.getenv("GEMINI_PRIVATE_KEY").replace("\\n", "\n"),
            "client_email": os.getenv("GEMINI_CLIENT_EMAIL"),
            "client_id": os.getenv("GEMINI_CLIENT_ID"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": os.getenv("GEMINI_CLIENT_X509_CERT_URL"),
        },
        scopes=scopes,
    )


def generate(contents: list) -> str | dict:
    temperature = 1.0
    generation_config = GenerationConfig(
        temperature=temperature,
        response_mime_type="application/json",
    )

    vertexai.init(project=os.getenv("GEMINI_PROJECT_ID"), credentials=credentials())
    # vertexai.init(project=os.getenv("GEMINI_PROJECT_ID"))

    api_version = os.getenv("GEMINI_API_VERSION")
    model = GenerativeModel(api_version)
    response = model.generate_content(
        contents=contents,
        generation_config=generation_config,
    )

    result = response.candidates[0].content.parts[0].text
    return json.loads(result)


def create_contents(parts: list) -> Content:
    return Content(
        role="user",
        parts=parts,
    )


def generate_classification(base64_img, base64_mark_img):
    system_prompt = Part.from_text(
        """
あなたはとても優秀なデータサイエンティスト。
宇宙ビジネスにも長けており、[衛生データ]を利用した分析を行っている。
[制約条件]に従って[出力フォーマット(Json)]の形で問いに答える。

### 衛生データ ###
- 地表の様子（光学センサ）
- 高さなどの地表の変化（SARセンサ）
- 地表の温度（熱赤外センサ）

### 制約条件 ###
- 屋上(屋根)を分析したそれぞれの割合については「0.0~1.0」の間で数値化
- 陸屋根(屋根が平たい)か勾配屋根(屋根に傾斜がある)かを判断
- 勾配屋根の場合は「ソーラーパネル、遊休資産、その他」から分類
- 陸屋根の場合はすべてのカテゴリから分類
- 給水設備、高架水槽、室外機、空調設備、ヘリポートなどの屋上設備は「その他」に割り当てる
- 緑地の場合、単一の緑一色では無く緑色の濃淡が表れていることが条件
- 駐車場の場合、車を止める目安となる白線があり、かつ 車で上るための道路が屋上に繋がていることが条件
- 駐車場の場合、周りに写っている車・人・建物から屋上の広さが十分であることを判断
- ステップバイステップで考え、考察した内容や結果を詳細に出す

### 出力フォーマット ###
```
{
"stepbystep": [{ステップバイステップで考察した結果}],
"result": [
    {
    "番号": {赤枠内に記載された番号}
    "ソーラーパネル": {屋上のソーラーパネルの設置割合},
    "緑地": {屋上の緑地化されている割合},
    "レジャー": {屋上のフットサルコートやゴルフなどのレジャー施設割合},
    "テラス": {屋上のビアガーデンなどのテラス化されている割合},
    "遊休資産": {屋上を持て余している割合},
    "駐車場": {屋上に駐車場がある割合},
    "その他": {屋上設備の割合}
    }
]
}
```
""".strip()
    )

    contents = create_contents(
        [
            system_prompt,
            Part.from_data(
                mime_type="image/png",
                data=base64_mark_img,
            ),
            Part.from_text(
                """
１つ目の画像は都市の一部の光学画像に赤枠と番号を入れた写真です。
赤枠で囲われた箇所が分析対象の家の屋上(屋根)を表しています。
""".strip()
            ),
            Part.from_data(
                mime_type="image/png",
                data=base64_img,
            ),
            Part.from_text(
                """
２つ目の画像は１つ目の画像から赤枠と番号を外した光学画像です。
１つ目の画像で分析箇所を認識し、該当箇所の分析結果をそれぞれ出力して。
""".strip()
            ),
        ]
    )

    return generate(contents=contents)


# ------------------------------
# 以下、テスト用コード


def generate_sample(prompt: str) -> str:
    temperature = 1.0
    generation_config = GenerationConfig(
        temperature=temperature,
    )

    vertexai.init(project=os.getenv("GEMINI_PROJECT_ID"), credentials=credentials())
    # vertexai.init(project=os.getenv("GEMINI_PROJECT_ID"))

    api_version = os.getenv("GEMINI_API_VERSION")
    model = GenerativeModel(api_version)
    response = model.generate_content(
        contents=create_contents(
            [
                Part.from_text(prompt),
            ]
        ),
        generation_config=generation_config,
    )

    result = response.candidates[0].content.parts[0].text
    return result
