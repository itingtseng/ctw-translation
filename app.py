"""
CSV 翻譯助手 (MVP)
------------------
一個 chat 式的 Streamlit app：
使用者上傳含中文欄位的 CSV、指定目標語言，
agent 自動偵測中文欄、逐欄翻譯，輸出一份新的 CSV
（原始欄 + 各目標語言的翻譯欄）供下載。

設計取捨（面試可講）：
- 用 Streamlit 而非 Dify：這個任務的重點在 CSV 的批次進出與結果可控，
  自己掌控 pipeline 比現學編排平台更快交出可用的 MVP。
- 自動偵測中文欄：降低使用者負擔，符合「用 chat 完成」的低摩擦體驗。
- 去重後再翻譯：同一欄常有重複值，只翻 unique 值可省 token、加快速度
  （對應 JD 提到的 token consumption 意識）。
- 分批 + JSON 回傳 + 單筆 fallback：兼顧效率與可靠性
  （對應 JD 的 fallback / degradation 思維）。
"""

import io
import json
import re

import pandas as pd
import streamlit as st
from openai import OpenAI

# ------------------------------------------------------------------
# 基本設定
# ------------------------------------------------------------------
st.set_page_config(page_title="CSV 翻譯助手", page_icon="🌐")

MODEL = "gpt-4o-mini"          # 便宜、快、翻譯品質足夠的模型
CHUNK_SIZE = 25                # 每次送給模型翻譯的筆數（分批，避免單次過長）
CJK_RE = re.compile(r"[\u4e00-\u9fff]")   # 判斷是否含中日韓漢字的正則


# ------------------------------------------------------------------
# 工具函式（不需要 API，可獨立測試）
# ------------------------------------------------------------------
def contains_chinese(text) -> bool:
    """判斷一段文字是否含中文字。"""
    return bool(CJK_RE.search(str(text)))


def detect_chinese_columns(df: pd.DataFrame, threshold: float = 0.3):
    """
    自動偵測哪些欄是中文欄。
    規則：該欄非空值中，含中文的比例 >= threshold 就算中文欄。
    用比例而非「有一個就算」，是為了避免把偶爾夾雜中文的欄（如地址）誤判。
    """
    chinese_cols = []
    for col in df.columns:
        values = df[col].dropna().astype(str)
        if len(values) == 0:
            continue
        frac = values.apply(contains_chinese).mean()
        if frac >= threshold:
            chinese_cols.append(col)
    return chinese_cols


def parse_languages(raw: str):
    """
    把使用者輸入的目標語言字串拆成清單。
    支援中英文逗號、頓號、分號、'and' 等分隔。
    例：'日文、韓文' -> ['日文', '韓文']；'Japanese, Korean' -> [...]
    """
    parts = re.split(r"[,，、;；]|\band\b|\s{2,}", raw)
    langs = [p.strip() for p in parts if p.strip()]
    # 去重但保留順序
    seen = set()
    result = []
    for l in langs:
        if l.lower() not in seen:
            seen.add(l.lower())
            result.append(l)
    return result


# ------------------------------------------------------------------
# 翻譯核心
# ------------------------------------------------------------------
def translate_chunk(client, texts, target_language):
    """
    翻譯一批文字。要求模型回傳 JSON 陣列，順序與輸入對齊。
    回傳 list[str]，長度與 texts 相同。
    失敗時回傳 None，讓上層做 fallback。
    """
    numbered = {str(i): t for i, t in enumerate(texts)}
    system = (
        "You are a professional translator. "
        f"Translate each value into {target_language}. "
        "Keep it faithful and natural. Do not add explanations. "
        "Return ONLY a JSON object mapping each key to its translation, "
        "with exactly the same keys as the input."
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(numbered, ensure_ascii=False)},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        # 依原始索引順序取回，缺的補原文
        return [data.get(str(i), texts[i]) for i in range(len(texts))]
    except Exception:
        return None


def translate_values(client, unique_values, target_language, progress_cb=None):
    """
    翻譯一組 unique 值（已去重）。
    分批處理；某批 JSON 失敗時，退回逐筆翻譯該批（fallback）。
    回傳 dict：原文 -> 譯文。
    """
    mapping = {}
    total = len(unique_values)
    done = 0
    for start in range(0, total, CHUNK_SIZE):
        chunk = unique_values[start:start + CHUNK_SIZE]
        translated = translate_chunk(client, chunk, target_language)

        if translated is None:
            # fallback：這批改成逐筆翻，單筆再失敗就保留原文
            translated = []
            for t in chunk:
                one = translate_chunk(client, [t], target_language)
                translated.append(one[0] if one else t)

        for original, tr in zip(chunk, translated):
            mapping[original] = tr

        done += len(chunk)
        if progress_cb:
            progress_cb(done / total)
    return mapping


def build_translated_df(client, df, chinese_cols, target_languages, status):
    """
    對每個中文欄、每個目標語言，各新增一欄翻譯結果。
    欄名格式：<原欄名>_<語言>，例如 name_日文。
    """
    out = df.copy()
    steps = len(chinese_cols) * len(target_languages)
    step = 0

    for lang in target_languages:
        for col in chinese_cols:
            step += 1
            status.write(f"翻譯中：欄位「{col}」→ {lang}（{step}/{steps}）")
            # 去重：同欄重複值只翻一次
            unique_vals = df[col].dropna().astype(str)
            unique_list = list(dict.fromkeys(unique_vals))  # 去重保序

            bar = st.progress(0.0)
            mapping = translate_values(
                client, unique_list, lang,
                progress_cb=lambda p: bar.progress(p),
            )
            bar.empty()

            new_col = f"{col}_{lang}"
            out[new_col] = df[col].map(
                lambda v: mapping.get(str(v), "") if pd.notna(v) else ""
            )
    return out


# ------------------------------------------------------------------
# 介面（chat 式）
# ------------------------------------------------------------------
def get_client():
    """從 Streamlit secrets 讀 API key，避免把 key 寫死在程式裡。"""
    key = st.secrets.get("OPENAI_API_KEY", None)
    if not key:
        st.error("尚未設定 OPENAI_API_KEY。請到 Streamlit 的 Secrets 設定後重試。")
        st.stop()
    return OpenAI(api_key=key)


def main():
    st.title("🌐 CSV 翻譯助手")

    # 對話歷史
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant",
             "content": "嗨！請在下方上傳一份含中文欄位的 CSV，"
                        "我會自動找出中文欄。上傳後，再告訴我要翻譯成哪些語言即可。"}
        ]
    if "df" not in st.session_state:
        st.session_state.df = None
        st.session_state.chinese_cols = []
        st.session_state.result = None

    # 顯示歷史訊息
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # 檔案上傳
    uploaded = st.file_uploader("上傳 CSV", type=["csv"], label_visibility="collapsed")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"讀取 CSV 失敗：{e}")
            st.stop()

        # 只有換了新檔才重新偵測，避免每次 rerun 重跑
        if st.session_state.df is None or not df.equals(st.session_state.df):
            st.session_state.df = df
            st.session_state.chinese_cols = detect_chinese_columns(df)
            st.session_state.result = None
            cols = st.session_state.chinese_cols
            if cols:
                msg = (f"收到，共 {len(df)} 筆資料。"
                       f"偵測到中文欄位：{', '.join(cols)}。\n\n"
                       f"請輸入目標語言（例如：日文、英文、韓文）。")
            else:
                msg = "我在這份 CSV 沒有偵測到明顯的中文欄位，請確認檔案內容。"
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.rerun()

    # 使用者輸入目標語言
    user_input = st.chat_input("輸入目標語言，例如：日文、英文")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        if st.session_state.df is None:
            st.session_state.messages.append(
                {"role": "assistant", "content": "請先上傳一份 CSV，再告訴我目標語言。"})
            st.rerun()

        if not st.session_state.chinese_cols:
            st.session_state.messages.append(
                {"role": "assistant", "content": "這份 CSV 沒有偵測到中文欄位，無法翻譯。"})
            st.rerun()

        target_languages = parse_languages(user_input)
        if not target_languages:
            st.session_state.messages.append(
                {"role": "assistant", "content": "我沒讀懂語言，請再輸入一次，例如：日文、英文。"})
            st.rerun()

        client = get_client()
        with st.chat_message("assistant"):
            status = st.empty()
            result_df = build_translated_df(
                client,
                st.session_state.df,
                st.session_state.chinese_cols,
                target_languages,
                status,
            )
            status.write("翻譯完成！")
        st.session_state.result = result_df
        st.session_state.messages.append(
            {"role": "assistant",
             "content": f"完成！已翻譯成：{', '.join(target_languages)}。"
                        f"可在下方下載結果 CSV。"})
        st.rerun()

    # 下載結果
    if st.session_state.result is not None:
        st.dataframe(st.session_state.result.head(20), use_container_width=True)
        buf = io.StringIO()
        st.session_state.result.to_csv(buf, index=False)
        st.download_button(
            "⬇️ 下載翻譯後的 CSV",
            data=buf.getvalue().encode("utf-8-sig"),  # utf-8-sig 讓 Excel 開中文不亂碼
            file_name="translated.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
