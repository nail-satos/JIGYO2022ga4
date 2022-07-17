""" streamlit_demo
streamlitでIrisデータセットの分析結果を Web アプリ化するモジュール

【Streamlit 入門 1】Streamlit で機械学習のデモアプリ作成 – DogsCox's tech. blog
https://dogscox-trivial-tech-blog.com/posts/streamlit_demo_iris_decisiontree/
"""

# 追加インストールしたライブラリ
import numpy as np
import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt 
import japanize_matplotlib
import seaborn as sns 

# ロゴの表示用
from PIL import Image

# 標準ライブラリ
import random
import copy
import math
import base64

# 自作パッケージ
from gafunc import display_table 
# from gafunc import display_individual 
from gafunc import generate_0th_generation 
from gafunc import add_unit_switch 
from gafunc import evaluation_individual 
from gafunc import generate_n_generation 
from gafunc import uniform_crossover_individuals
from gafunc import generate_next_generation
from gafunc import transform_norma



sns.set()
japanize_matplotlib.japanize()  # 日本語フォントの設定

# matplotlib / seaborn の日本語の文字化けを直す、汎用的かつ一番簡単な設定方法 | BOUL
# https://boul.tech/mplsns-ja/


def main():
    """ メインモジュール
    """

    # セッションステートを初期化する
    if 'ini_flg' not in st.session_state:

        st.session_state.ini_flg = True

        # パラーメータの初期設定（CO2排出量）
        temp = []
        temp.append([10, 7, 3])   # マシンA（製造時、交換時、遊休時）
        temp.append([ 5, 4, 2])   # マシンB（製造時、交換時、遊休時）
        temp.append([ 3, 2, 1])   # マシンC（製造時、交換時、遊休時）
        st.session_state.co2_params_list = temp

        # パラーメータの初期設定（製造能力:キャパシティ）
        temp = []
        temp.append([20, 10, 15])   # マシンA（部品α、部品β、部品γ）
        temp.append([30, 15, 10])   # マシンB（部品α、部品β、部品γ）
        temp.append([50, 30, 20])   # マシンC（部品α、部品β、部品γ）
        st.session_state.cap_params_list = temp

        # パラメータの初期設定（稼働率）
        st.session_state.operating_rate = 75

    # stのタイトル表示
    st.title("遺伝的アルゴリズム\n（製造機器の稼働におけるCO2排出量の最適化問題)")

    # サイドメニューの設定
    activities = ["生産計画確認", "ＣＯ２排出量", "部品製造能力", "最適化の実行", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == '生産計画確認':

        # ファイルのアップローダー
        uploaded_file = st.sidebar.file_uploader("データのアップロード", type='csv') 

        # アップロードの有無を確認
        if uploaded_file is not None:

            # 一度、read_csvをするとインスタンスが消えるので、コピーしておく
            ufile = copy.deepcopy(uploaded_file)

            try:
                # 文字コードの判定（utf-8 bomで試し読み）
                pd.read_csv(ufile, encoding="utf_8_sig")
                enc = "utf_8_sig"
            except:
                enc = "shift-jis"

            finally:
                # データフレームの読み込み（一番左端の列をインデックスに設定）
                df_plan = pd.read_csv(uploaded_file, encoding=enc, index_col=0) 

                # データフレームをセッションステートに退避
                st.session_state.df_plan = copy.deepcopy(df_plan)

        # セッションステートにデータフレームがあるかを確認
        if 'df_plan' in st.session_state:

            # セッションステートに退避していたデータフレームを復元
            df_plan = copy.deepcopy(st.session_state.df_plan)

            # テーブルの表示
            display_table('生産計画（指示書）', df_plan)

            # 時間ごとの生産計画（ノルマ）を、0時台からの累積台数に変換する関数
            df_norma = transform_norma(df_plan)

            display_table('生産計画（累積数）', df_norma)

        else:
            st.subheader('生産計画データをアップロードしてください')


    if choice == 'ＣＯ２排出量':

        # カラムの作成
        col1, col2, col3 = st.columns(3)

        # セッションステートから値をコピー（セッションステートをそのまま使うと遅いため）
        params_list = copy.deepcopy(st.session_state.co2_params_list)   # DeepCopyしないとダメ

        captions = ['製造時のCO2排出量(/h)', '交換時のCO2排出量(/h)', '遊休時のCO2排出量(/h)', ]

        with col1:

            st.text('マシンＡの性能')

            for idx, caption in enumerate(captions):
                params_list[0][idx] = st.number_input(caption, value=params_list[0][idx], key=params_list[0][idx] * 0)

        with col2:

            st.text('マシンＢの性能')

            for idx, caption in enumerate(captions):
                params_list[1][idx] = st.number_input(caption, value=params_list[1][idx], key=params_list[0][idx] * 1)

        with col3:

            st.text('マシンＣの性能')

            for idx, caption in enumerate(captions):
                params_list[2][idx] = st.number_input(caption, value=params_list[2][idx], key=params_list[0][idx] * 2)


        # 保存ボタンの作成（これは、この位置 = params_listの定義より後ろにないとNG）
        st.sidebar.text('パラメータの保存')
        if st.sidebar.button('保存の実行'):
            st.session_state.co2_params_list = copy.deepcopy(params_list)   # DeepCopyで値を戻す


    if choice == '部品製造能力':

        # カラムの作成
        col1, col2, col3 = st.columns(3)

        # セッションステートから値をコピー（セッションステートをそのまま使うと遅いため）
        params_list = copy.deepcopy(st.session_state.cap_params_list)   # DeepCopyしないとダメ
        op_rate = st.session_state.operating_rate

        captions = ['部品αの製造能力(/h)', '部品βの製造能力(/h)', '部品γの製造能力(/h)', ]

        with col1:

            st.text('マシンＡの性能')

            for idx, caption in enumerate(captions):
                params_list[0][idx] = st.number_input(caption, value=params_list[0][idx], key=params_list[0][idx] * 0)

        with col2:

            st.text('マシンＢの性能')

            for idx, caption in enumerate(captions):
                val = params_list[1][idx]
                params_list[1][idx] = st.number_input(caption, value=params_list[1][idx], key=params_list[0][idx] * 1)

        with col3:

            st.text('マシンＣの性能')

            for idx, caption in enumerate(captions):
                params_list[2][idx] = st.number_input(caption, value=params_list[2][idx], key=params_list[0][idx] * 2)

        op_rate = st.number_input('第0世代の稼働率(%)', value=op_rate)

        # 保存ボタンの作成（これは、この位置 = params_listの定義より後ろにないとNG）
        st.sidebar.text('パラメータの保存')
        if st.sidebar.button('保存の実行'):
            st.session_state.cap_params_list = copy.deepcopy(params_list)   # DeepCopyで値を戻す
            st.session_state.operating_rate =  copy.deepcopy(op_rate)


    if choice == '最適化の実行':

        # ペナルティの重み設定
        incomplete_loss = st.sidebar.number_input('生産不足のペナルティ（重み）', value=1)
        co2_loss = st.sidebar.number_input('ＣＯ２排出量のペナルティ（重み）', value=1)

        # complete_loss = st.sidebar.number_input('生産過多のペナルティ（重み）', value=0)
        complete_loss = 0

        # change_loss = st.sidebar.number_input('交換作業のペナルティ（重み）', value=0)
        change_loss = 0

        max_individual = st.sidebar.slider('世代の個体数', value=200, min_value=5, max_value=1000, step=5)
        max_generation = st.sidebar.slider('生成する世代数(n)', value=100, min_value=5, max_value=1000, step=5)
        mutation_rate = st.sidebar.slider('突然変異の頻度', value=10, min_value=0, max_value=100, step=1)

        choice_crossover = st.sidebar.selectbox("交叉の種類", ['一様交叉', '一点交叉'])

        # セッションステートにデータフレームがあるかを確認
        if 'df_plan' not in st.session_state:

            # 仮データフレームの読み込み（一番左端の列をインデックスに設定） ※デバック用
            df_plan = pd.read_csv('生産計画(仮).csv', encoding="utf_8_sig", index_col=0) 

            # データフレームをセッションステートに退避  ※デバック用
            st.session_state.df_plan = copy.deepcopy(df_plan)
        else:
            # セッションステートに退避していたデータフレームを復元
            df_plan = copy.deepcopy(st.session_state.df_plan)


        # テーブルの表示
        display_table('生産計画（指示書）', df_plan)

        # 時間ごとの生産計画（ノルマ）を、0時台からの累積台数に変換する関数
        df_norma = transform_norma(df_plan)

        # テーブルの表示
        display_table('生産計画（累積数）', df_norma)

        # データフレームをセッションステートに退避(generate_next_generationで呼び出す)
        st.session_state.df_norma = copy.deepcopy(df_norma)
        

        # 全世代の個体を格納するリストを初期化
        df_shift_list = []

        if st.sidebar.button('アルゴリズム実行'):

            for i in range(max_individual):

                # 第0世代の生成（引数：稼働率）
                df_shift_0th = generate_0th_generation(st.session_state.operating_rate)
                # display_individual('第0世代(個体:' + str(i) + '番)', df_shift_0th, [0, 0, 0, 0, 0])
                df_shift = copy.deepcopy(df_shift_0th)
                df_shift_list.append(df_shift)    # リストに格納


            # ペナルティの重みをリスト化する
            loss_list = [incomplete_loss, complete_loss, co2_loss, change_loss]

            # 全世代のベストスコアを格納しておくリストを初期化
            best_score_lists = []

            # 指定された世代分を繰り返すループ
            for n in range(1, max_generation + 1):

                # 次世代の個体群を生成（世代数, 全個体のシフト, 各種ペナルティのリスト, 突然変異の割合, 一点交叉 or 一様交叉）
                df_shift_next_list, best_score_list = generate_next_generation(n, df_shift_list, loss_list, mutation_rate, choice_crossover)

                # 現世代のベストスコアをリストに追加
                best_score_lists.append(best_score_list)

                # 次世代の個体は少し多めに交叉しているので、個数をここで調整
                df_shift_next_list = df_shift_next_list[:max_individual]

                # 次世代を現世代にコピーして次のループへ
                df_shift_list = copy.deepcopy(df_shift_next_list)

            # 処理終了のバルーンを表示
            st.balloons()

            st.subheader('～ シフト表で用いられる記号 ～')
            st.text('遊休=0, 製造(部品α)=1, 製造(部品β)=2, 製造(部品γ)=3, 交換作業=9')


            # グラフ用のデータフレームを作成
            df_best_score = pd.DataFrame(best_score_lists, columns=['生産不足率(％)', '生産不足(評価値)', 'CO2排出量率(％)', 'CO2排出量(評価値)', '合計(評価値)'])

            # 評価値のみのデータフレームを作成 -> 線グラフを出力
            st.subheader('～ 評価値のグラフ ～')
            st.caption('※負の値をとる・0に近い方が良い')
            df_chart = df_best_score.drop(['生産不足率(％)', 'CO2排出量率(％)'], axis=1)
            st.line_chart(df_chart)

            # 生産不足とCO2排出量のみのデータフレームを作成 -> 線グラフを出力
            st.subheader('～ 生産不足率とCO2排出量率のグラフ ～')
            st.caption('※正の値をとる・0に近い方が良い')
            df_chart = df_best_score.drop(['生産不足(評価値)', 'CO2排出量(評価値)', '合計(評価値)'], axis=1)
            st.line_chart(df_chart)

            # CSVファイルのダウンロードリンクを生成
            df_best_score.index = np.arange(1, len(df_best_score)+1)    # インデックスを1から振り直し（初期値が0なので）
            csv = df_best_score.to_csv(index=True) 
            b64 = base64.b64encode(csv.encode('utf-8-sig')).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="result_ga.csv">Download Link</a>'
            st.markdown(f"CSVファイルのダウンロード(utf-8 BOM):  {href}", unsafe_allow_html=True)

        st.sidebar.caption('Built by [Nail Team]')  # 下マージン用
                                
        # else:
        #     st.subheader('生産計画データをアップロードしてください')


    if choice == 'About':

        image = Image.open('logo_nail.png')
        st.image(image)

        st.markdown("Built by [Nail Team]")
        st.text("Version 4.1")
        st.markdown("For More Information check out   (https://nai-lab.com/)")
        

if __name__ == "__main__":
    main()


