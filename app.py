import streamlit as st
import requests
import json
import base64
import re
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np  # NumPyをインポート

# タイトル
st.title('えこやね＋')

# 地点情報
locations = [
    { 'name': 'ソーラーと緑地', 'latitude': 35.678839, 'longitude': 139.670472 },
    { 'name': '駐車場', 'latitude': 33.26512332046375, 'longitude': 130.2990732847798 },
    { 'name': '住宅地１', 'latitude': 33.283136905958024, 'longitude': 130.29063271676938 },
    { 'name': '住宅地２', 'latitude': 33.28483633746525, 'longitude': 130.28816182014825 }
]

# セッションステートの初期化メソッド
def initialize_session_state():
    if 'latitude' not in st.session_state:
        st.session_state.latitude = ''
    if 'longitude' not in st.session_state:
        st.session_state.longitude = ''



# API呼び出しとレスポンス処理のメソッド
def call_api_and_display_image(latitude, longitude, chk_nollm):
    with container:
        st.write(f'あなたが入力した緯度・経度は、{latitude}、{longitude}です。')
        st.write(f'LLM問合せをスキップするかどうか: {chk_nollm}')

        # 送信パラメータ
        params = {
            'lat': latitude,
            'lon': longitude
        }

        with st.spinner('地図データ取得中...'):
            # API呼び出し
            #curl -X POST "http://127.0.0.1:9080/analyze" -H "Content-Type: application/json" -d '{"lat": "35.9037751", "lon": "139.5901251"}' -o analyze_result.json
            response = requests.post("http://localhost:9080/analyze", headers={"Content-Type": "application/json"}, data=json.dumps(params))


        # レスポンスの処理
        if response.status_code == 200:
            # 戻り値取得
            data = json.loads(response.content)

            # 画像データを表示
            image_data = base64.b64decode(data['image'])
            mark_image_data = base64.b64decode(data['mark_image'])

            image = Image.open(BytesIO(image_data))
            mark_image = Image.open(BytesIO(mark_image_data))

            imagecol1, imagecol2 = st.columns((1,1))
            with imagecol1:
                st.image(image, caption='地図画像')
            with imagecol2:
                st.image(mark_image, caption='マーカー画像')
            
            if chk_nollm:
                st.write('LLM問合せをSkip')
            else:
                # LLM問合せAPI呼び出し
                call_llm_api(data)
        else:
            st.write('API呼び出しに失敗しました。')

# LLM問合せAPI呼び出し
def call_llm_api( imagedata ):

    if chk_nollm:
        st.write('LLM問合せをスキップします。')
        return  

    # jsonパラメータ作成  image と mark_image
    llm_params = {    
        "image": imagedata['image'],
        "mark_image": imagedata['mark_image']
    }

    #curl -X POST -H "Content-Type: application/json" -d @analyze_result.json http://127.0.0.1:9080/generate
    with st.spinner('LLM問合せ中...'):
        response = requests.post("http://localhost:9080/generate", headers={"Content-Type": "application/json"}, data=json.dumps(llm_params))

    with container:
        #LLM問合せ結果を表示
        # llmresponse = {
        #     'stepbystep': [
        #         {'番号': 1, '考察': '屋根の色が周囲の家の屋根と比べて同じであり、太陽光パネルや緑化などはされていない。\nまた、航空写真からは影も見られないため、その他設備もないと考えられる。\nよって、この屋上は遊休資産だと考えられる。', 'result': {'ソーラーパネル': 0.0, '緑地': 0.0, 'レジャー': 0.0, 'テラス': 0.0, '遊休資産': 1.0, '駐車場': 0.0, 'その他': 0.0}},
        #         {'番号': 2, '考察': '屋根の色が周囲の家の屋根と比べて同じであり、太陽光パネルや緑化などはされていない。\nまた、航空写真からは影も見られないため、その他設備もないと考えられる。\nよって、この屋上は遊休資産だと考えられる。', 'result': {'ソーラーパネル': 0.0, '緑地': 0.0, 'レジャー': 0.0, 'テラス': 0.0, '遊休資産': 1.0, '駐車場': 0.0, 'その他': 0.0}}
        #     ],
        #     'result': [
        #         {'番号': 1, 'ソーラーパネル': 0.3, '緑地': 0.2, 'レジャー': 0.0, 'テラス': 0.0, '遊休資産': 0.5, '駐車場': 0.0, 'その他': 0.0},
        #         {'番号': 2, 'ソーラーパネル': 0.0, '緑地': 0.0, 'レジャー': 0.0, 'テラス': 0.2, '遊休資産': 0.3, '駐車場': 0.2, 'その他': 0.3}
        #     ]
        # }

        if response.status_code == 200:
            llmresponse = json.loads(response.content)

            # カテゴリと色の設定
            categories = ['ソーラーパネル', '緑地', 'レジャー', 'テラス', '遊休資産', '駐車場', 'その他']
            colors = ['#118ab2','#06d6a0','#FF6347','#ffcc99','#ffcc33','#b8b8ff', '#999999']

            result = llmresponse['result']
            # データの準備
            data = np.array([[float(item[cat]) for cat in categories] for item in result])[::-1]
            data_cumsum = np.cumsum(data, axis=1)

            # グラフの作成
            fig, ax = plt.subplots(figsize=(10, 5))

            for i, cat in enumerate(categories):
                widths = data[:, i]
                starts = data_cumsum[:, i] - widths
                ax.barh(range(len(result)), widths, left=starts, height=0.5, label=cat, color=colors[i])

            # ラベルと凡例の設定
            ax.set_yticks(range(len(result)))
            ax.set_yticklabels([f"番号 {item['番号']}" for item in result][::-1])
            ax.legend(loc='upper right')

            plt.xlabel('比率')
            plt.ylabel('番号')
            plt.title('屋上用途分析')
            st.pyplot(fig)

            #st.write(f'LLM問合せ結果: {llmresponse}')
            st.write('LLM問合せ結果')
            st.json(llmresponse)


        else:
            st.write('LLM問合せに失敗しました。')



# セッションステートの初期化を呼び出し
initialize_session_state()

# サイドバーにボタンを動的に生成
st.sidebar.text('登録地点')
for location in locations:
    if st.sidebar.button(location['name']):
        st.session_state.latitude = location['latitude']
        st.session_state.longitude = location['longitude']

mapurl = st.sidebar.text_input('GooglemapURLから緯度経度' )
if st.sidebar.button('設定'):
    # GooglemapURLから正規表現で緯度経度取得
    # https://www.google.co.jp/maps/@43.1487815,141.3584129,802m/data=!3m1!1e3?hl=ja&entry=ttu&g_ep=EgoyMDI0MTExOC4wIKXMDSoASAFQAw%3D%3D
    match = re.search(r'@(.+),(.+),', mapurl)
    if match:
        st.session_state.latitude = match.group(1)
        st.session_state.longitude = match.group(2)
    else:
        st.write('GooglemapURLから緯度経度を取得できませんでした。')



# 緯度、経度の入力フィールドと送信ボタンを横並びに配置
col1, col2, col3, col4, col5 = st.columns((4,4,1.5,2,2))
container = st.container()

with col1:
    latitude = st.text_input('緯度', value=st.session_state.latitude)
with col2:
    longitude = st.text_input('経度', value=st.session_state.longitude)
with col5:
    # チェックボックス
    chk_nollm = st.checkbox('No LLM', value=False)
with col3:
    if st.button('送信'):
        call_api_and_display_image(latitude, longitude, chk_nollm)
with col4:
    if st.button('リセット'):
        # # TODO:リアクティブしないので、 text_inputは keyを指定して値を同期させる必要がありそう
        st.session_state.latitude = ''
        st.session_state.longitude = ''
