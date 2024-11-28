# えこやね＋

## 利用前に

同階層に「.env」ファイルを作成し、以下の値を設定している必要があります。

```sh
# Google Map API key
GOOGLE_API_KEY="APIキーを入れる"

# Gemini認証情報
GEMINI_PROJECT_ID="GCPサービスアカウントjsonファイルの内容を入れる"
GEMINI_PRIVATE_KEY_ID="GCPサービスアカウントjsonファイルの内容を入れる"
GEMINI_PRIVATE_KEY="GCPサービスアカウントjsonファイルの内容を入れる"
GEMINI_CLIENT_EMAIL="GCPサービスアカウントjsonファイルの内容を入れる"
GEMINI_CLIENT_ID="GCPサービスアカウントjsonファイルの内容を入れる"
GEMINI_CLIENT_X509_CERT_URL="GCPサービスアカウントjsonファイルの内容を入れる"
GEMINI_API_VERSION="gemini-1.5-pro-preview-0514"
```

## API関連

```sh
# Pythonパッケージのインストール
# ※環境によっては「sudo」が必要
pip --disable-pip-version-check --no-cache-dir install -r requirements.txt

# FastAPI 起動
# 開発時: uvicorn main_fapi:app --port 9080 --reload
uvicorn main_fapi:app --port 9080
```

### API呼出し例

```sh
# 緯度経度情報から解析結果を受け取るAPI
curl -X POST "http://127.0.0.1:9080/analyze" -H "Content-Type: application/json" -d '{"lat": "35.9037751", "lon": "139.5901251"}' -o analyze_result.json

# 解析結果画像から生成AIによる分析結果を受け取るAPI
curl -X POST -H "Content-Type: application/json" -d @analyze_result.json http://127.0.0.1:9080/generate
```

## Streamlit関連

### 起動方法

```sh
streamlit run app.py
```
