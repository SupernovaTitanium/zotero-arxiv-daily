<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="assets/logo.svg" alt="logo"></a>
</p>

<h3 align="center">Zotero-arXiv-Daily</h3>

<div align="center">

  [![Status](https://img.shields.io/badge/status-active-success.svg)]()
  ![Stars](https://img.shields.io/github/stars/TideDra/zotero-arxiv-daily?style=flat)
  [![GitHub Issues](https://img.shields.io/github/issues/TideDra/zotero-arxiv-daily)](https://github.com/TideDra/zotero-arxiv-daily/issues)
  [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/TideDra/zotero-arxiv-daily)](https://github.com/TideDra/zotero-arxiv-daily/pulls)
  [![License](https://img.shields.io/github/license/TideDra/zotero-arxiv-daily)](/LICENSE)
  [<img src="https://api.gitsponsors.com/api/badge/img?id=893025857" height="20">](https://api.gitsponsors.com/api/badge/link?p=PKMtRut1dWWuC1oFdJweyDSvJg454/GkdIx4IinvBblaX2AY4rQ7FYKAK1ZjApoiNhYEeduIEhfeZVIwoIVlvcwdJXVFD2nV2EE5j6lYXaT/RHrcsQbFl3aKe1F3hliP26OMayXOoZVDidl05wj+yg==)

</div>

---

<p align="center"> Recommend new arxiv papers of your interest daily according to your Zotero library.
    <br> 
</p>

## 🧐 About <a name = "about"></a>

*Zotero-arXiv-Daily* finds arxiv papers that may attract you based on the context of your Zotero library, and then sends the result to your mailbox📮. This fork runs it every morning from a Windows PC, which hands the run to Light1 or Light2 (see *Windows 每日執行* below).

## ✨ Features
- Short LLM-written teasers (Traditional Chinese) for you to quickly pick up target papers.
- A link to each paper's PDF in the e-mail.
- Papers sorted by relevance to your Zotero library and grouped by topic.
- Ignore unwanted Zotero papers using a list of glob patterns.
- Support multiple sources of papers to retrieve:
  - arxiv
  - biorxiv
  - medrxiv
  - chemrxiv

## 📷 Screenshot
![screenshot](./assets/screenshot.png)

## 🚀 Usage
### Configuration
Credentials and endpoints come from environment variables; the Windows daily run below reads them from `config/local.env.json`.

| Key |Description | Example |
| :---  | :---  | :--- |
| ZOTERO_ID  | User ID of your Zotero account. **User ID is not your username, but a sequence of numbers**Get your ID from [here](https://www.zotero.org/settings/security). You can find it at the position shown in this [screenshot](https://github.com/TideDra/zotero-arxiv-daily/blob/main/assets/userid.png). | 12345678  |
| ZOTERO_KEY | An Zotero API key with read access. Get a key from [here](https://www.zotero.org/settings/security).  | AB5tZ877P2j7Sm2Mragq041H   |
| SENDER | The email account of the SMTP server that sends you email. | abc@qq.com |
| SENDER_PASSWORD | The password of the sender account. Note that it's not necessarily the password for logging in the e-mail client, but the authentication code for SMTP service. Ask your email provider for this.   | abcdefghijklmn |
| RECEIVER | The e-mail address that receives the paper list. | abc@outlook.com |
| OPENAI_API_KEY | API Key when using the API to access LLMs. You can get FREE API for using advanced open source LLMs in [SiliconFlow](https://cloud.siliconflow.cn/i/b3XhBRAm). | sk-xxx |
| OPENAI_API_BASE | API URL when using the API to access LLMs. | https://api.siliconflow.cn/v1 |
| EMAIL_SMTP_SERVER / EMAIL_SMTP_PORT | SMTP server and port of the sender account (the legacy names `SMTP_SERVER` / `SMTP_PORT` also work). | smtp.gmail.com / 465 |

`config/custom.yaml` (gitignored; start from `config/local.example.yaml`) is deep-merged on top of the full configuration below (`config/base.yaml`). `${VAR}` interpolates an environment variable and `${VAR:default}` provides a fallback.
```yaml
zotero:
  user_id: ${ZOTERO_ID} # User ID of your Zotero account.
  api_key: ${ZOTERO_KEY} # A Zotero API key with read access.
  include_path: null # Glob patterns of collections to include. Example: ["2026/survey/**"]
  ignore_path: null # Glob patterns of collections to exclude. Example: ["archive/**"]

email:
  sender: ${SENDER}
  receiver: ${RECEIVER}
  sender_password: ${SENDER_PASSWORD}
  subject_prefix: Daily Papers

llm:
  api_key: ${OPENAI_API_KEY}
  base_url: ${OPENAI_API_BASE}
  model: gpt-4o-mini
  max_tokens: 16384
  language: Traditional Chinese # Teaser output language
  requests_per_minute: 10 # Max chat-completion requests per minute; 0 disables throttling
  teaser_char_limit: 150 # Max characters per teaser
  teaser_batch_size: 10 # Papers per teaser LLM request
  max_concurrency: 4 # Teaser requests in flight at once (requests_per_minute still spaces their starts); 1 = one at a time

embedding:
  model: jinaai/jina-embeddings-v5-text-nano-retrieval # Hugging Face embedding model for ranking

executor:
  categories: ["cs.AI", "cs.CV", "cs.LG", "cs.CL"] # arXiv categories to follow
  include_cross_list: false
  send_empty: false # Send an email even when no new papers were found
  max_paper_num: 100 # Papers presented in the email
  lookback_days: 3 # Days back to retrieve, so a failed run is caught up next run
  state_file: state/recommended.json
  preferences_file: preferences.yaml # Weekly-review boost/mute keywords
```

### Local Running
Supported by [uv](https://github.com/astral-sh/uv), this workflow can easily run on your local device if uv is installed:
```bash
# set all the environment variables
# export ZOTERO_ID=xxxx
# ...
cd zotero-arxiv-daily
uv run python -m zotero_arxiv_daily.main
```

### Windows 每日執行（Light1 → Light2 → 本機）

入口 `scripts/run_digest.ps1` 在本機啟動，依序選擇執行主機：Light1、Light2，兩台都不可用時才在本機執行。每台遠端主機都在 `127.0.0.1:10100` 跑自己的 opencodex 代理，`config/custom.yaml` 的端點因此不必改。

- **選主機**：以 SSH（`C:\Windows\System32\OpenSSH\ssh.exe`，金鑰 `id_ed25519_light1` / `id_ed25519-remote-ssh`）探測代理健康（`/health` 回 200）、`uv` 存在、CPU 負載低於 90%、可用記憶體至少 3 GB；連線失敗（ssh 結束碼 255）重試一次。
- **推送**：把 `src`、`scripts/remote_entry.py`、`config/base.yaml`、`config/custom.yaml`、`pyproject.toml`、`uv.lock`、`.python-version`、`state/recommended.json`、`state/corpus_embeddings.npz`（及存在時的 `preferences.yaml`）打包複製到遠端 `D:\zotero-arxiv-daily`，清掉遠端舊的 `src`、`state`、`output` 後解開，只在 `uv.lock` 與上次同步時不同時執行 `uv sync --frozen --no-dev --no-install-project`（依 `uv.lock` 的 CPU 版 PyTorch）。Light1 的 CPU 常被其他卸載工作占滿（CPU 嵌入實測每篇約 5 秒），因此同步後改裝 CUDA 版 `torch==2.11.0+cu128` 用其 RTX 4060；Light2 維持 CPU 版。改變某台的 PyTorch 版本後須刪除其 `.venv`。推送或同步失敗時改試下一台，因為此時尚未執行。
- **憑證**：`config/local.env.json` 只經 SSH 標準輸入交給 `scripts/remote_entry.py`，載入該程序的環境後即執行，不寫入遠端磁碟。
- **執行**：遠端輸出即時寫入本機日誌。開始前寫入 `state/remote-run.json` 記下主機；成功後把遠端的 `state` 與 `output` 拉回（`recommended.json` 以原子取代），再刪除該記號。遠端執行失敗不改到別台重跑，因為可能已寄出郵件或失敗通知；拉回失敗時結束碼為 `3`，下次執行開始前會先從記號上的主機補拉。
- **本機備援**：使用 `C:\Users\light\venvs\zotero-arxiv-daily-cuda\Scripts\python.exe`（Python 3.13、CUDA 版 PyTorch `2.11.0+cu128`），需本機代理健康。若需重建：`uv venv --python C:\Python313\python.exe <路徑>`，先用 `--index-url https://download.pytorch.org/whl/cu128` 安裝 `torch==2.11.0`，再依 `uv.lock` 的版本安裝其餘依賴。
- 郵件頁尾與 `output/run_summary_*.json` 的 `host` 欄位記錄實際執行主機。

設定步驟：

1. 若 `config/custom.yaml` 不存在，將 `config/local.example.yaml` 複製為 `config/custom.yaml`；若已存在，先檢查內容，不要直接覆蓋。此覆寫與 `config/base.yaml` 深度合併，固定端點 `http://127.0.0.1:10100/v1`，每次最多推薦 100 篇。
2. 將 `config/local.env.example.json` 複製為 `config/local.env.json`，用本機編輯器填寫所有空字串。`OPENAI_API_KEY` 是代理的入站憑證，不是模型供應商金鑰；請透過本機安全方式取得及輸入，不要貼到聊天、命令列或紀錄。Gmail 使用 `EMAIL_SMTP_SERVER` = `smtp.gmail.com`、`EMAIL_SMTP_PORT` = `465`，`SENDER_PASSWORD` 填 Gmail 應用程式密碼。
3. 此 JSON 是明文私密檔案：放在非同步的本機磁碟，限制檔案權限為自己的 Windows 帳戶，不分享、不上傳。`.gitignore` 排除它與 `config/custom.yaml`，但忽略規則不是加密。入口只接受範例中的八個名稱與非空字串，不印出值或 JSON 解析錯誤內容；日誌寫入前遮蔽憑證值。
4. 遠端主機需要使用者層級的 `uv`（`C:\Users\light\.local\bin\uv.exe`）與執行中的 opencodex 代理；第一次執行會下載嵌入模型。

```powershell
# 只檢查設定、憑證格式、獨占鎖與各主機狀態；不抓文章、不呼叫模型、不寄信
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\light\sideproject\zotero-arxiv-daily\scripts\run_digest.ps1" -Check
# 真實執行（會寄信）
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\light\sideproject\zotero-arxiv-daily\scripts\run_digest.ps1"
# 在指定主機測試（會寄信）：不帶去重歷史，結果不拉回，日誌為 logs\testrun-*.log
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\light\sideproject\zotero-arxiv-daily\scripts\run_digest.ps1" -TestHost Light2
```

入口使用 `state/local-run.lock` 的獨占檔案鎖避免重疊執行，並傳回執行的結束碼；前置檢查失敗回傳 `1`。鎖檔存在不代表仍在執行。保留 `state/recommended.json`、`state/corpus_embeddings.npz` 與 `preferences.yaml`；不要為了重跑刪除歷史。`DEBUG`、`MAX_PAPER_NUM`、`LOOKBACK_DAYS` 等環境覆寫只在本機備援時有效。

日誌位於 `C:\Users\light\sideproject\zotero-arxiv-daily\logs\digest-yyyyMMdd-HHmmss-fff-PID.log`，每行使用本機時間，末行為 `runner finished, exit=N`。

### 排程

每日 08:00（UTC+3）由本機的 server-dashboard 觸發：Windows 任務 `Hub-Tick` 每分鐘執行 `hub_tick.py`，依 `C:\Users\light\.zcode\workspace\default\server-dashboard\services.toml` 的 `[job.arxiv-digest]` 以 `wscript.exe //B //NoLogo scripts\run-hidden.vbs` 啟動入口（六小時上限）。`run-hidden.vbs` 同步等待隱藏的 `powershell.exe -WindowStyle Hidden` 並傳回結束碼，不開前景視窗。停用：從 `services.toml` 移除 `[job.arxiv-digest]` 區塊。

## 📖 How it works
*Zotero-arXiv-Daily* firstly retrieves all the papers in your Zotero library and all the papers released in the previous `lookback_days` days, via corresponding API. Then it calculates the embedding of each paper's abstract via an embedding model (corpus embeddings are cached between runs). The score of a paper is its mean similarity to the 10 Zotero papers closest to it, so a paper close to one of your interests ranks above one halfway between several (replayed on the papers most recently added to the library, this put 85% of them in that day's top 100, against 66% for the former weighted average over the whole library). A short Traditional-Chinese teaser for each presented paper is generated by LLM from its title and abstract (one batched request per 10 papers, several requests in flight at once). If the LLM endpoint keeps failing for about 20 minutes, the email goes out with the abstracts in place of the missing teasers and a notice at the top; a run that fails sends a short failure notice instead of the digest. arXiv papers are retrieved by an OAI-PMH harvest of the papers announced in the lookback window, with the search API as a fallback when the harvest fails.

## 🗃️ State and caching
The pipeline keeps two state files in `state/` (a remote run copies them back to this PC after it succeeds):
- `state/recommended.json` — papers already processed, so nothing is recommended twice. Entries older than `history_days` are pruned.
- `state/corpus_embeddings.npz` — cached embeddings of your Zotero abstracts; only papers new to the library are embedded each run. It is rebuilt automatically when the embedding model changes.

Each run also writes the rendered email and a full ranking with scores to `output/` for debugging.

## 📌 Limitations
- The recommendation algorithm is very simple, it may not accurately reflect your interest. Welcome better ideas for improving the algorithm!


## 📃 License
Distributed under the AGPLv3 License. See `LICENSE` for detail.

## ❤️ Acknowledgement
- [pyzotero](https://github.com/urschrei/pyzotero)
- [arxiv](https://github.com/lukasschwab/arxiv.py)
- [sentence_transformers](https://github.com/UKPLab/sentence-transformers)

## ☕ Buy Me A Coffee
If you find this project helpful, welcome to sponsor me via WeChat or via [ko-fi](https://ko-fi.com/tidedra).
![wechat_qr](assets/wechat_sponsor.JPG)


## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=TideDra/zotero-arxiv-daily&type=Date)](https://star-history.com/#TideDra/zotero-arxiv-daily&Date)
