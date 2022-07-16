# 追加インストールしたライブラリ
from optparse import Values
import numpy as np
import pandas as pd 
import streamlit as st
# import matplotlib.pyplot as plt 
# import japanize_matplotlib
# import seaborn as sns 

# ロゴの表示用
# from PIL import Image

# 標準ライブラリ
import random
import copy
import itertools
import math

# sns.set()
# japanize_matplotlib.japanize()  # 日本語フォントの設定

# matplotlib / seaborn の日本語の文字化けを直す、汎用的かつ一番簡単な設定方法 | BOUL
# https://boul.tech/mplsns-ja/


def display_table(title, df: pd.DataFrame):
    st.subheader(title)
    st.table(df)


# def display_individual(title, df: pd.DataFrame, score_list: list):

#     # データフレームを表示
#     st.subheader(title)
#     st.text(f'生産不足：{score_list[0]} + 生産過多：{score_list[1]} + CO2排出量：{score_list[2]} + 交換作業：{score_list[3]} = 合計：{score_list[4]} 点')
#     st.table(df)

#     # Streamlitでdataframeを表示させる | ITブログ
#     # https://kajiblo.com/streamlit-dataframe/


# 時間ごとの製造指示（ノルマ）を、0時台からの累積台数に変換する関数
def transform_norma(df_norma: pd.DataFrame):

    # 戻り値用にデータフレームをディープコピー
    df_new_norma = copy.deepcopy(df_norma)

    # データフレームのインデックスを振り直し（0～）
    df_norma = df_norma.reset_index(drop=True)

    # 製造指示から1行ずつ取り出し（部品α, β, γ）
    for parts_no, row in df_norma.iterrows():

        # 1行（部品）から時間帯ごとの状態（ステータス）を取り出し
        for hour, norma in enumerate(row):

            if norma != 0:
                for i in range(hour + 1, 24):
                    df_new_norma.iloc[parts_no, i] = df_new_norma.iloc[parts_no, i] + df_norma.iloc[parts_no, hour]

    return df_new_norma


def generate_0th_generation(operating_rate : int):

    # 全てゼロのデータフレームを作成
    zero = np.zeros((3,24), dtype=np.int8)
    df_shift = pd.DataFrame(zero, index=['マシンＡ', 'マシンＢ', 'マシンＣ'])

    # 初期状態では（とりあえず）フル稼働（Ａ=α、Ｂ=β、Ｃ=γ）
    df_shift.loc['マシンＡ'] = df_shift.loc['マシンＡ'].replace(0, 1)
    df_shift.loc['マシンＢ'] = df_shift.loc['マシンＢ'].replace(0, 2)
    df_shift.loc['マシンＣ'] = df_shift.loc['マシンＣ'].replace(0, 3)

    i = 0
    while i < 100:
        # さすがに偏り過ぎているので、製造部品をシャッフルする
        # ランダム：稼働させる機器の番号(0～2)
        c1 = random.randint(0, 2)
        r1 = random.randint(0, 23)
        c2 = random.randint(0, 2)
        r2 = random.randint(0, 23)

        temp = df_shift.iloc[c1, r1]
        df_shift.iloc[c1, r1] = df_shift.iloc[c2, r2]
        df_shift.iloc[c2, r2] = temp
        
        i = i + 1

    # 稼働率に準拠して、「遊休」の個数を算出する
    size = df_shift.size
    op_rate = operating_rate / 100
    idle_cnt = size - int(size * op_rate)

    # 決められた個数の遊休を挿入するループ
    i = 0
    while i < idle_cnt:
        c = random.randint(0, 2)
        r = random.randint(0, 23)
        if df_shift.iloc[c, r] != 0:
            df_shift.iloc[c, r] = 0
            i = i + 1

    return df_shift


### 部品の交換箇所をチェックして、2hの交換(9)を挿入する関数
def add_unit_switch(sr: pd.Series):

    new_shift = []
    # unit_prev = sr.values.tolist()[0]   # 最初の部品を記録
    unit_prev = -999    # 番兵をセット

    for unit in sr.values.tolist():
        if unit == unit_prev or unit == 0:
            # 記録していた前の部品と同じだったら...
            #（または、遊休中(0)の場合はノーカン）
            new_shift.append(unit)
        else:
            # 記録していた前の部品と違っていたら...
            # 2hの交換(9)を挿入する
            new_shift.append(9)
            new_shift.append(9)
            new_shift.append(unit)
            unit_prev = unit    # 部品を記録しなおし

    # 交換中を挿入したシフトの冒頭24hのみを戻す ※これで大丈夫なのか？
    return new_shift[0:24]


# 個体（データフレーム1個分）に対する評価を算出（個体, ノルマ, 生産能力, CO2排出量, ペナルティの重さ）
def evaluation_individual(df_shift: pd.DataFrame, df_norma: pd.DataFrame, cap_params_list: list, co2_params_list: list, loss_list: list):

    # ノルマの総数（生産不足スコアの分母）を求める
    norma_sum = df_norma.sum().sum()

    # 作業用データフレームの作成
    df_remain = copy.deepcopy(df_norma)     # ノルマをコピーして、製造残を管理するデータフレームを作成
    df_co2    = copy.deepcopy(df_shift)     # ノルマをコピーして、CO2排出量を管理するデータフレームを作成
    df_co2    = df_co2.mask(df_remain != -1, 0)     # CO2排出量をオール0で初期化

    # データフレームのインデックスを振り直し（0～）
    df_shift = df_shift.reset_index(drop=True)

    # ペナルティをリストから復元
    incomplete_loss = loss_list[0]  # 生産不足のペナルティ
    complete_loss   = loss_list[1]  # 生産過多のペナルティ
    co2_loss        = loss_list[2]  # CO2排出量のペナルティ
    change_loss     = loss_list[3]  # 交換作業のペナルティ

    # 交換作業のスコアを初期化
    change_score = 0

    # 個体から1行ずつ取り出し（マシンＡ, Ｂ, Ｃ）
    for machine_no, row in df_shift.iterrows():

        # 1行（マシン）から時間帯ごとの状態（ステータス）を取り出し
        for hour, status in enumerate(row):

            # ステータスが1=部品α、2=部品β、3=部品γを作る
            if status == 1 or status ==2 or status ==3:

                parts_no = status - 1   # 添字の調整

                # 製造残から、製造した量を減算（その時、後ろの時間帯(h)に関しても、スライスで全て減算する）
                df_remain.iloc[parts_no, hour: ] = df_remain.iloc[parts_no, hour: ] - cap_params_list[machine_no][parts_no]

            # ステータスごとに添え字を設定
            if status == 1 or status ==2 or status ==3:
                status_idx = 0  # 製造時
            if status == 9:
                status_idx = 1  # 交換時
            if status == 0:
                status_idx = 2  # 遊休時
            if status == -1:
                status_idx = 2  # 淘汰した親類（パラメータは交換時で代用）

            # CO2排出量を加算
            df_co2.iloc[machine_no, hour] = df_co2.iloc[machine_no, hour] + co2_params_list[machine_no][status_idx]

            if status == 9:
                change_score = change_score - change_loss

    # 生産不足の算出：製造残が0以下（ノルマ以上は作れた）のものを0にする -> 残るのは生産不足のみとなる
    df_incomplete = df_remain.mask(df_remain <= 0, 0)     # maskの代わりにwhereを使うと挙動が逆になる
    # incomplete_score = df_incomplete.sum().sum() * incomplete_loss * -1

    # ノルマの総数（生産不足スコアの分母）を求める
    incomplete_sum = df_incomplete.sum().sum()
    incomplete_per = ( incomplete_sum / norma_sum )
    incomplete_score = incomplete_per * incomplete_loss * -1

    # # 生産過多の算出：製造残が1以上（ノルマに達しなかった）のものを0にする -> 残るのは生産過多のみとなる
    # df_complete = df_remain.mask(df_remain >= 1, 0)     # maskの代わりにwhereを使うと挙動が逆になる
    # complete_score = df_complete.sum().sum() * complete_loss

    # CO2排出量スコアの算出
    co2_sum = df_co2.sum().sum()
    co2_max = (max(co2_params_list[0]) + max(co2_params_list[1]) + max(co2_params_list[2])) * 24
    co2_per = (co2_sum / co2_max)
    co2_score = co2_per * co2_loss * -1

    # 戻り値 = ['生産不足率(％)', '生産不足(評価値)', 'CO2排出量率(％)', 'CO2排出量(評価値)', '合計(評価値)'])
    return [incomplete_per, incomplete_score, co2_per, co2_score, (incomplete_score + co2_score)]


def generate_n_generation(df: pd.DataFrame):
    return(df)


# 2つの個体（遺伝子）を受け取り、一様交叉を行う関数
def uniform_crossover_individuals(df1: pd.DataFrame, df2: pd.DataFrame, mutation_rate: int):

    # データフレームを1次元のリストに変換
    list1 = df1.values.tolist()
    list1 = list(itertools.chain.from_iterable(list1))

    list2 = df2.values.tolist()
    list2 = list(itertools.chain.from_iterable(list2))

    # 新しい遺伝子を格納するリストを初期化
    new_list = []

    # 一様交叉のループ
    for idx in range(len(list1)):

        rnd = random.randint(1, 2)
        if rnd == 1:
            new_list.append(list1[idx])
        else:
            new_list.append(list2[idx])

    rnd_per = random.randint(0, 100)
    if rnd_per < mutation_rate:

        while 1 == 1:
            # 突然変異を発生
            rnd_idx = random.randint(0, len(new_list) - 1)
            rnd_sts = random.randint(0,3)

            if new_list[rnd_idx] != rnd_sts:
                # print(f'{rnd_idx} →　{new_list[rnd_idx]}')
                new_list[rnd_idx] = rnd_sts
                # print(f'{rnd_idx} →　{new_list[rnd_idx]}')
                # print(f'に突然変異しました')            
                break

    # 1次元リストを2次元リストに変換
    new_list_2d = [new_list[i:i + 24] for i in range(0, len(new_list), 24)]

    # 2次元リストをデータフレームに変換
    df_new = pd.DataFrame(new_list_2d, index=df1.index, columns=df1.columns)

    return df_new


# 2つの個体（遺伝子）を受け取り、一点交叉を行う関数
def single_crossover_individuals(df1: pd.DataFrame, df2: pd.DataFrame, mutation_rate: int):

    # データフレームを1次元のリストに変換
    list1 = df1.values.tolist()
    list1 = list(itertools.chain.from_iterable(list1))

    list2 = df2.values.tolist()
    list2 = list(itertools.chain.from_iterable(list2))

    # 新しい遺伝子を格納するリストを初期化
    new_list = []

    # 交叉する箇所をランダムで決定
    rnd = random.randint(1, 70)

    # 交叉箇所を抜き取り結合
    list1 = list1[:rnd]
    list2 = list2[rnd:]
    list1.extend(list2)
    new_list = list1

    # 1次元リストを2次元リストに変換
    new_list_2d = [new_list[i:i + 24] for i in range(0, len(new_list), 24)]

    # 2次元リストをデータフレームに変換
    df_new = pd.DataFrame(new_list_2d, index=df1.index, columns=df1.columns)

    return df_new


# 世代の全個体（データフレーム）が格納されたリストを受け取り、重複している個体をALL-1で置き換える関数
def replace_duplicates_individuals(df_list: list):

    ret_df_list = []    # 戻り値用のリスト（2次元に整形し直した全個体dfを格納する）

    # インデックス名とカラム名を保存（戻り値用）
    df_template = copy.deepcopy(df_list[0])

    lists = []
    for df in df_list:
        # データフレームを1次元のリストに変換
        list1 = df.values.tolist()
        list1 = list(itertools.chain.from_iterable(list1))
        lists.append(list1)

    # 作業用データフレームに変換
    df_temp = pd.DataFrame(lists)

    # 評価用の遺伝子が重複している個体（親戚）は、最初の1件を残して-1に置換
    df_temp[df_temp.duplicated(keep='first')] = -1
    new_lists = df_temp.values.tolist()

    for new_list in new_lists:
        # 1次元リストを2次元リストに変換
        new_list_2d = [new_list[i:i + 24] for i in range(0, len(new_list), 24)]

        # リストをデータフレームに変換
        df_new = pd.DataFrame(new_list_2d, index=df_template.index, columns=df_template.columns)
        ret_df_list.append(df_new)
    
    # print('重複置換後：len(ret_df_list)')
    # print(ret_df_list)

    return ret_df_list


# 引数（世代数, 全個体のシフト, 各種ペナルティのリスト, 突然変異の割合, 一点交叉 or 一様交叉, 総生産数）
def generate_next_generation(n: int, df_shift_list : list, loss_list: list, mutation_rate: int, choice_crossover: str):


    # 全個体のスコアを格納しておくデータフレームを生成
    # return [incomplete_sum, incomplete_score, co2_sum, co2_score, (incomplete_score + co2_score)]

    df_score = pd.DataFrame(columns=['生産不足率(％)', '生産不足(評価値)', 'CO2排出量率(％)', 'CO2排出量(評価値)', '合計(評価値)'])
    score_lists = []    # df_scoreに代入するための作業用のリスト

    # 評価用の個体を格納しておくリスト
    df_shift_evaluation_list = []

    # リストから個体を1つずつ取り出し
    for idx, df_shift in enumerate(df_shift_list):

        temp_shift_list = []     # 交換(9)を挿入した行を3行まとめるためのリスト

        # 個体から1行ずつ取り出し（マシンＡ, Ｂ, Ｃ）
        for index, row in df_shift.iterrows():
            temp_shift = add_unit_switch(row)     # 部品の交換をチェックして、2hの交換(9)を挿入する
            temp_shift_list.append(temp_shift)

        # 個体評価用のデータフレームを作成
        df_shift_evaluation = pd.DataFrame(temp_shift_list,  index=['マシンＡ', 'マシンＢ', 'マシンＣ'])
        df_shift_evaluation_list.append(df_shift_evaluation)

    # 世代の全個体（データフレーム）が格納されたリストを受け取り、重複している個体をALL-1で置き換える
    # 評価用の遺伝子が同じ個体（親類）を、最初の1件以外を残して淘汰する（親類が増えることによる収束防止）
    df_shift_evaluation_list = replace_duplicates_individuals(df_shift_evaluation_list)

    # 個体を評価する
    df_norma = st.session_state.df_norma                # 製造指示（ノルマ）を読み込み
    cap_params_list = st.session_state.cap_params_list  # 部品製造能力を読み込み
    co2_params_list = st.session_state.co2_params_list  # ＣＯ２排出量を読み込み

    # リストから個体を1つずつ取り出し
    for idx, df_shift_evaluation in enumerate(df_shift_evaluation_list):

        # 生産ノルマを守れているかの評価 ＆ ＣＯ２排出量を評価
        # 戻り値 = ['生産不足率(％)', '生産不足(評価値)', 'CO2排出量率(％)', 'CO2排出量(評価値)', '合計(評価値)'])
        score_list = evaluation_individual(df_shift_evaluation, df_norma, cap_params_list, co2_params_list, loss_list)

        # スコア群をリストに格納
        score_lists.append(score_list)

    # リストに格納しておいた全個体のスコア群をデータフレームに変換
    df_score_sort = pd.DataFrame(score_lists, columns=df_score.columns)

    # 合計スコアの降順に並び替え
    df_score_sort = df_score_sort.sort_values('合計(評価値)', ascending=False)
    display_table('第' + str(n) + '世代 スコア一覧表（ベスト10）', df_score_sort.head(10))

    # スコアの降順に並び替えたときのインデックス番号を取得
    idx_list = list(df_score_sort.index)

    # 合計スコアの降順に個体（遺伝子）を格納するループ
    df_shift_sort_list = []
    df_shift_evaluation_sort_list = []
    for idx in idx_list:
        # リストに個体（遺伝子）を格納
        df_shift_sort_list.append(df_shift_list[idx])
        df_shift_evaluation_sort_list.append(df_shift_evaluation_list[idx])    

    # 個体数の平方根を求めて整数に丸める（エリートの個体数を算出）
    elite_count = round(math.sqrt(len(df_shift_sort_list))) + 1
    df_shift_sort_list = df_shift_sort_list[:elite_count]
    
    # 世代の最優秀個体は、遺伝子組換えをせずに次世代に残す
    df_shift_next_list = []
    df_shift_next_list.append(df_shift_sort_list[0])

    # 世代の最優秀スコアを記録する（戻り値用）
    best_score_list = df_score_sort.iloc[0, :].values.tolist()

    # No.1 評価用個体の表示（交換の9を取り除かない0～23hの遺伝子）
    # display_individual('第' + str(n) + '世代 最優秀個体', df_shift_evaluation_sort_list[0], best_score_list)
    display_table('第' + str(n) + '世代 最優秀個体', df_shift_evaluation_sort_list[0])

    # No.2 評価用個体の表示（交換の9を取り除かない0～23hの遺伝子）
    # display_individual('第' + str(n) + '世代 No.2 個体', df_shift_evaluation_sort_list[1], df_score_sort.iloc[1, :].values.tolist())
    display_table('第' + str(n) + '世代 No.2 個体', df_shift_evaluation_sort_list[1])


    i = 1
    for idx1, df1 in enumerate(df_shift_sort_list):
        for idx2, df2 in enumerate(df_shift_sort_list):

            # 同じ個体（遺伝子）同士は交叉させない
            if idx1 != idx2:
                if choice_crossover == '一点交叉':
                    # 2つの個体（遺伝子）を渡して、一点交叉を行う → 結果を次世代のリストに格納
                    df_shift_next_list.append(single_crossover_individuals(df1, df2, mutation_rate))
                else:
                    # 2つの個体（遺伝子）を渡して、一様交叉を行う → 結果を次世代のリストに格納
                    df_shift_next_list.append(uniform_crossover_individuals(df1, df2, mutation_rate))

    # 次世代のリストを戻す
    return df_shift_next_list, best_score_list
