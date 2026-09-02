# CSV 翻譯助手 (MVP)

一個 chat 式的 Streamlit app：上傳含中文欄位的 CSV、指定目標語言，
自動偵測中文欄並翻譯，輸出一份新的 CSV（原始欄 + 各目標語言翻譯欄）。

---

## 一、本機先跑起來（可選，想先在自己電腦測就做）

1. 安裝相依套件：
   ```
   pip install -r requirements.txt
   ```
2. 設定 API key（本機用環境變數或 `.streamlit/secrets.toml`）：
   在專案下建一個檔案 `.streamlit/secrets.toml`，內容：
   ```
   OPENAI_API_KEY = "sk-你的key"
   ```
   （這個檔案不要上傳到 GitHub，見下方 .gitignore）
3. 啟動：
   ```
   streamlit run app.py
   ```

---

## 二、部署到 Streamlit Community Cloud（拿到公開 URL）

Streamlit 官方免費部署，交作業要的「可直接訪問的 URL」就從這裡來。

1. **把程式碼放上 GitHub**
   - 建一個 GitHub repo（可設 private）。
   - 上傳 `app.py`、`requirements.txt`、`README.md`、`.gitignore`。
   - **千萬不要**上傳含 key 的 `secrets.toml`。

2. **連到 Streamlit Cloud**
   - 到 https://share.streamlit.io，用 GitHub 帳號登入。
   - 點 **Create app** → 選你的 repo、branch、主程式檔 `app.py`。

3. **設定 API key（重點，key 放這裡最安全）**
   - 部署設定裡點 **Advanced settings → Secrets**。
   - 貼上：
     ```
     OPENAI_API_KEY = "sk-你的key"
     ```
   - 存檔。程式用 `st.secrets["OPENAI_API_KEY"]` 讀，不會出現在原始碼裡。

4. **Deploy**
   - 按下部署，等它 build 完，就會給你一個 `https://xxx.streamlit.app` 的公開網址。
   - 這個網址就是交給 CTW 的成果 URL。

---

## 三、.gitignore（避免 key 外流）

專案下建一個 `.gitignore`，至少包含：
```
.streamlit/secrets.toml
__pycache__/
*.pyc
```

---

## 四、目前 MVP 做到的範圍

- chat 式介面（上傳 → 指定語言 → 下載結果）
- 自動偵測中文欄（以「該欄含中文比例」判斷，避免誤判夾雜中文的欄）
- 支援自由輸入多個目標語言
- 去重後翻譯（省 token、加速）
- 分批翻譯 + 單筆 fallback（某批失敗不會整個掛掉）
- 支援 100 筆以上，輸出 UTF-8-SIG（Excel 開中文不亂碼）

## 五、之後可加的亮點（第二階段）

- 讓使用者手動勾選要翻的欄（覆蓋自動偵測）
- 翻譯品質抽查 / 讓使用者確認（human-in-the-loop）
- 顯示每種語言的成本與耗時估計
- 失敗列的清單與重試
- 大檔的分頁處理與進度保存
