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

> [!IMPORTANT]
> Please keep an eye on this repo, and merge your forked repo in time when there is any update of this upstream, in order to enjoy new features and fix found bugs.

## 🧐 About <a name = "about"></a>

> Track new scientific researches of your interest by just forking (and staring) this repo!😊

*Zotero-arXiv-Daily* finds arxiv papers that may attract you based on the context of your Zotero library, and then sends the result to your mailbox📮. It can be deployed as Github Action Workflow with **zero cost**, **no installation**, and **few configuration** of Github Action environment variables for daily **automatic** delivery.

## ✨ Features
- Totally free! All the calculation can be done in the Github Action runner locally within its quota (for public repo).
- AI-generated TL;DR for you to quickly pick up target papers.
- Affiliations of the paper are resolved and presented.
- Links of PDF and code implementation (if any) presented in the e-mail.
- List of papers sorted by relevance with your recent research interest.
- Fast deployment via fork this repo and set environment variables in the Github Action Page.
- Support LLM API for generating TL;DR of papers.
- Ignore unwanted Zotero papers using a list of glob patterns.
- Support multiple sources of papers to retrieve:
  - arxiv
  - biorxiv
  - medrxiv
  - chemrxiv

## 📷 Screenshot
![screenshot](./assets/screenshot.png)

## 🚀 Usage
### Quick Start
1. Fork (and star😘) this repo.
![fork](./assets/fork.png)

2. Set Github Action environment variables.
![secrets](./assets/secrets.png)

Below are all the secrets you need to set. They are invisible to anyone including you once they are set, for security.

| Key |Description | Example |
| :---  | :---  | :--- |
| ZOTERO_ID  | User ID of your Zotero account. **User ID is not your username, but a sequence of numbers**Get your ID from [here](https://www.zotero.org/settings/security). You can find it at the position shown in this [screenshot](https://github.com/TideDra/zotero-arxiv-daily/blob/main/assets/userid.png). | 12345678  |
| ZOTERO_KEY | An Zotero API key with read access. Get a key from [here](https://www.zotero.org/settings/security).  | AB5tZ877P2j7Sm2Mragq041H   |
| SENDER | The email account of the SMTP server that sends you email. | abc@qq.com |
| SENDER_PASSWORD | The password of the sender account. Note that it's not necessarily the password for logging in the e-mail client, but the authentication code for SMTP service. Ask your email provider for this.   | abcdefghijklmn |
| RECEIVER | The e-mail address that receives the paper list. | abc@outlook.com |
| OPENAI_API_KEY | API Key when using the API to access LLMs. You can get FREE API for using advanced open source LLMs in [SiliconFlow](https://cloud.siliconflow.cn/i/b3XhBRAm). | sk-xxx |
| OPENAI_API_BASE | API URL when using the API to access LLMs. | https://api.siliconflow.cn/v1 |

Then you should also set a public variable `CUSTOM_CONFIG` for your custom configuration.
![vars](./assets/repo_var.png)
![custom_config](./assets/config_var.png)
Paste the following content into the value of `CUSTOM_CONFIG` variable:
```yaml
zotero:
  user_id: ${ZOTERO_ID}
  api_key: ${ZOTERO_KEY}
  include_path: null # Or e.g. ["2026/survey/**", "2026/reading-group/**"]

email:
  sender: ${SENDER}
  receiver: ${RECEIVER}
  smtp_server: smtp.gmail.com
  smtp_port: 465
  sender_password: ${SENDER_PASSWORD}

llm:
  api_key: ${OPENAI_API_KEY}
  base_url: ${OPENAI_API_BASE}
  model: gpt-4o-mini

executor:
  categories: ["cs.AI", "cs.CV", "cs.LG", "cs.CL"]
  include_cross_list: false # Set to true to include arXiv cross-list papers in these categories.
```
`smtp_server` / `smtp_port` may be omitted: they are then read from the `EMAIL_SMTP_SERVER` / `EMAIL_SMTP_PORT` variables (preferred) or the legacy `SMTP_SERVER` / `SMTP_PORT` secrets.
>[!NOTE]
> `${oc.env:XXX,yyy}` means the value of the environment variable `XXX`. If the variable is not set, the default value `yyy` will be used.

Here is the full configuration (`config/base.yaml`); anything set in `CUSTOM_CONFIG` is deep-merged on top of it, and `${VAR}` interpolates environment variables (`${VAR:default}` provides a fallback):
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
  fulltext_paper_num: 30 # Top N papers get full-text fetching after ranking


That's all! Now you can test the workflow by manually triggering it:
![test](./assets/test.png)

> [!NOTE]
> The Test-Workflow Action is the debug version of the main workflow (Send-emails-daily), which always retrieve 5 arxiv papers regardless of the date. While the main workflow will be automatically triggered everyday and retrieve new papers released yesterday. There is no new arxiv paper at weekends and holiday, in which case you may see "No new papers found" in the log of main workflow.

Then check the log and the receiver email after it finishes.

By default, the main workflow runs on 22:00 UTC everyday. You can change this time by editting the workflow config `.github/workflows/main.yml`.

### Local Running
Supported by [uv](https://github.com/astral-sh/uv), this workflow can easily run on your local device if uv is installed:
```bash
# set all the environment variables
# export ZOTERO_ID=xxxx
# ...
cd zotero-arxiv-daily
uv run main.py
```

## 🚀 Sync with the latest version
This project is in active development. You can subscribe this repo via `Watch` so that you can be notified once we publish new release.

![Watch](./assets/subscribe_release.png)


## 📖 How it works
*Zotero-arXiv-Daily* firstly retrieves all the papers in your Zotero library and all the papers released in the previous `lookback_days` days, via corresponding API. Then it calculates the embedding of each paper's abstract via an embedding model (corpus embeddings are cached between runs). The score of a paper is its weighted average similarity over all your Zotero papers (newer paper added to the library has higher weight). Full text is fetched only for the top ranked papers, and a short Traditional-Chinese teaser for each paper is generated by LLM (one batched request per 10 papers). arXiv retrieval falls back to an OAI-PMH harvest when the search API is rate-limited.

## 🗃️ State and caching
The pipeline keeps two state files (persisted between runs via [actions/cache](https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows) in the bundled workflow):
- `state/recommended.json` — papers already processed, so nothing is recommended twice. If the cache is evicted (10GB repo quota / 7 days unused), the worst case is a paper being recommended once more.
- `state/corpus_embeddings.npz` — cached embeddings of your Zotero abstracts; only papers new to the library are embedded each run. It is rebuilt automatically when the embedding model changes.

Each run also writes the rendered email and a full ranking with scores to `output/`, uploaded as a workflow artifact for debugging.

## 📌 Limitations
- The recommendation algorithm is very simple, it may not accurately reflect your interest. Welcome better ideas for improving the algorithm!
- High `MAX_PAPER_NUM` can lead the execution time exceed the limitation of Github Action runner (6h per execution for public repo, and 2000 mins per month for private repo). Commonly, the quota given to public repo is definitely enough for individual use. If you have special requirements, you can deploy the workflow in your own server, or use a self-hosted Github Action runner, or pay for the exceeded execution time.


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
