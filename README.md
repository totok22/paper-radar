# paper-radar：自动驾驶方向每日论文雷达

[![daily](https://github.com/totok22/paper-radar/actions/workflows/daily.yml/badge.svg)](https://github.com/totok22/paper-radar/actions/workflows/daily.yml)
[![tests](https://github.com/totok22/paper-radar/actions/workflows/ci.yml/badge.svg)](https://github.com/totok22/paper-radar/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11-blue)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> Daily autonomous-driving paper radar (motion planning / control / scenario generation): multi-source recall → embedding prefilter → LLM review → Feishu push.

每天从 **arXiv / OpenAlex / 目标期刊 / Semantic Scholar 推荐 / DeepSeek 网页搜索** 召回新论文，去重后按课题组子方向做嵌入预筛（**每个子方向单独打分，取最大**），只把道路车辆相关的 Top-K 交给 LLM 精审，再叠加刊会、代码、机构、Hugging Face、OpenReview 等质量信号，用 MMR 去冗余并留一个探索位，按「运动规划 / 运动控制 / 场景生成」推到飞书。

不依赖 Zotero。兴趣写在 `config.yaml`。以后要接 Zotero，打开配置即可，不必改流水线。

```
sources/    arxiv · openalex · journals(ISSN) · semantic_scholar · deepseek_search · rss
s2.py       Semantic Scholar 全端点共用 1.1s 限速，优先 paper/batch、author/batch
interests/  config · feedback · zotero(可选) · seeds
rank/       embed(API → 本地模型 → TF-IDF，API 向量缓存) · signals · llm · score
pdf.py      arXiv PDF，以及 OpenAlex / Unpaywall / S2 的开放获取 PDF
push/       飞书卡片（含 PDF 按钮）· 可选自建应用发送 PDF 文件 · Markdown/HTML 日报
```

## 本地运行

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m paper_radar validate
python -m paper_radar run --dry-run    # 生成 reports/日期.md .html 和飞书卡片 JSON，不推送、不上传文件
```

没有 `LLM_API_KEY` 时用嵌入 + 质量信号做启发式评分，卡片里会标明「启发式」，TL;DR 不是模型写的中文。配上 key 之后才有中文 TL;DR 和推荐理由。

```bash
export LLM_API_KEY=sk-...
export LLM_BASE_URL=https://api.deepseek.com    # 带不带 /v1 都可以
export LLM_MODEL=deepseek-flash                  # 也可用 deepseek-v4-pro
export EMBED_API_KEY=...
export EMBED_BASE_URL=https://api.siliconflow.cn/v1
export EMBED_MODEL=Qwen/Qwen3-Embedding-8B
python -m paper_radar run --dry-run
python -m paper_radar run                        # 真正推送，需要 FEISHU_WEBHOOK
```

## GitHub Actions

每天 **00:30 UTC（北京时间 08:30）** 跑 `.github/workflows/daily.yml`。仓库 Settings → Secrets and variables → Actions：

| Secret | 必需 | 说明 |
|---|---|---|
| `FEISHU_WEBHOOK` | 推送卡片时需要 | 群自定义机器人 Webhook |
| `FEISHU_SECRET` | 推荐 | 机器人签名校验密钥 |
| `LLM_API_KEY` | 推荐 | DeepSeek 等 OpenAI 兼容接口 |
| `LLM_BASE_URL` | 可选 | 默认 `https://api.deepseek.com` |
| `LLM_MODEL` | 可选 | 默认 `deepseek-flash` |
| `S2_API_KEY` | 推荐 | Semantic Scholar。所有端点合计 **1 次/秒**，代码里按 1.1 秒间隔共用一个限速器 |
| `OPENALEX_API_KEY` / `OPENALEX_EMAIL` | 可选 | OpenAlex；邮箱也用于 Unpaywall |
| `UNPAYWALL_EMAIL` | 可选 | 不填则用 `OPENALEX_EMAIL` |
| `EMBED_API_KEY` / `EMBED_BASE_URL` / `EMBED_MODEL` | 可选 | 不填则用 TF-IDF |
| `FEISHU_APP_ID` / `FEISHU_APP_SECRET` / `FEISHU_RECEIVE_ID` / `FEISHU_RECEIVE_ID_TYPE` | 可选 | 四个都有才会把 PDF 文件发到飞书；缺任何一个只发卡片 |

Settings → Actions → General → Workflow permissions 选 **Read and write**（把 `data/state.sqlite` 和 `reports/` 提交回去）。

手动验证：Actions → paper-radar daily → Run workflow，`dry_run` 填 **true**。这一步只写报告和卡片 JSON，**不会**调用飞书 webhook，也**不会**上传 PDF。日志里的 `stats` 应包含 `per_source`、`selected`、`with_pdf`、`s2_requests`。确认报告没问题后，再用 `dry_run=false` 正式推送。dry-run 的报告和卡片 JSON 作为 Actions artifact 上传（保留 14 天），**不会**提交 `data/state.sqlite`，所以不会把论文记成已推送。

`workflow_dispatch` 的 `days` 仍可临时覆盖 `lookback_days`。

## 飞书卡片

每个大方向一张卡片（太长会拆成多张，单条控制在约 18KB，低于自定义机器人 20KB 限制）。每篇有：

- 分数徽章、子方向、中文 TL;DR、为什么推荐
- **PDF 按钮**：arXiv 为 `https://arxiv.org/pdf/{id}`；期刊论文用 OpenAlex / Unpaywall（按 DOI）/ Semantic Scholar `openAccessPdf` 里真正的 PDF 链接。没有开放获取 PDF 时不编造链接，只保留 DOI
- arXiv、DOI、代码按钮（摘要或备注里出现 `github.com/owner/repo` 时）

### 自定义机器人（卡片）

1. 群设置 → 群机器人 → 添加 → 自定义机器人。
2. Webhook 存为 `FEISHU_WEBHOOK`。
3. 安全设置选签名校验，密钥存为 `FEISHU_SECRET`。
4. 自定义机器人**不能发文件**，所以 PDF 文件要走下面的自建应用。

### 自建应用（可选，把 PDF 文件发进会话）

不配置下面的环境变量时，这条路径完全关闭。

1. 打开 [飞书开放平台](https://open.feishu.cn/app) → 创建**企业自建应用**。
2. 「凭证与基础信息」复制 App ID、App Secret → `FEISHU_APP_ID`、`FEISHU_APP_SECRET`。
3. 「添加应用能力」打开**机器人**。
4. 「权限管理」开通并发布版本：
   - `im:message`
   - `im:message:send_as_bot`（以应用身份发消息）
   - `im:resource`（上传文件；控制台里也可能写成上传图片/文件资源）
5. 创建版本、让管理员审核通过，然后把这个机器人拉进目标群。
6. 拿接收方 ID（开放平台 → API 调试台，使用该应用的 tenant token）：
   - 群：`GET /open-apis/im/v1/chats`，用返回里 `oc_` 开头的 `chat_id`。`FEISHU_RECEIVE_ID_TYPE=chat_id`
   - 个人：通过手机号或邮箱获取用户 ID，用 `open_id`。`FEISHU_RECEIVE_ID_TYPE=open_id`（这也是缺省值）
7. `FEISHU_RECEIVE_ID` 填上一步的 ID。

运行时会取 `tenant_access_token`，下载综合分最高的 `pdf.push_top_n` 篇（默认 5）直接 PDF，单文件超过 `pdf.max_bytes`（默认 20MB；飞书上传上限 30MB）就跳过，再 `POST /open-apis/im/v1/files`（`file_type=pdf`）并作为文件消息发送。`--dry-run` 只在日志里列出将要发送的文件，不下载、不上传。

国际版 Lark 可设 `FEISHU_OPEN_BASE=https://open.larksuite.com`。

## 打分（每天大约十几次 S2 请求）

1. **召回**：arXiv 全量小类 + 大类关键词；OpenAlex 按子方向检索；ISSN 期刊；S2 用正/负种子做推荐；DeepSeek 网页搜索（见下）。某个源失败只记日志，其他源继续。
2. **去重**：arXiv id（去掉版本号）、DOI（含 `10.48550/arXiv.…`）、规范化标题、标题模糊匹配。同一篇在多个源出现时合并摘要、PDF 和代码链接。
3. **道路车辆过滤**：标题和摘要里要能看出是道路上的车（driving / vehicle / traffic / lane / nuPlan / CARLA / Waymo 等）。仓储机器人、机械臂、足式、无人机、没有写明可迁移到驾驶的多机器人，会在预筛前丢掉，也不会拿来填某个方向的空位。`min_keep` 只在剩下的车辆论文里保底，不会把这些论文补回来。
4. **子方向画像**：描述文本，加上最多几条正种子摘要（缓存过的不再请求）。分数是「画像相似度」和「最接近的种子」的加权，再减去负例（每个子方向都带一段机械臂/仓储/足式负例），然后**取各子方向的最大值**。低于 `ranking.min_emb` 的丢掉，但至少保留 `min_keep` 篇领域内论文，避免空日报。
5. **精审**：只评 Top-K。LLM 按批（默认 4 篇）返回 JSON；坏 JSON 会重试，再退回单篇。模型默认 `deepseek-flash`。提示词要求：对象不是道路车辆时相关度给 1–3。相关度低于 `min_relevance`（默认 5）的丢掉。
6. **质量信号**：arXiv comments 里的录用信息、基准/真车关键词、代码链接、OpenAlex 机构、S2 h-index（`paper/batch`，缺 h-index 才 `author/batch`）、引用数、开放获取 PDF、HF 日榜、OpenReview 公开评分（失败则跳过）。
7. **多样性**：MMR，外加 `exploration_slots`（默认 1）留给还没出现的子方向。空位只从还在池子里的车辆论文里补，不会为了凑方向塞进机械臂或仓储论文。

### DeepSeek 网页搜索

`sources.deepseek_search` 调用 `POST https://api.deepseek.com/v1/tools/search`，鉴权是 `Bearer LLM_API_KEY`（和审稿用同一把密钥，没有单独的 secret）。每个子方向每天大约 3 条查询：`site:arxiv.org`、`site:ieeexplore.ieee.org OR site:sciencedirect.com`，以及一条 OpenReview / arXiv 的英文变体，都带 `after:`（默认今天往前 7 天）和最多 50 条结果。`rerank_score` 低于 `min_rerank_score`（默认 1.0；官方把 ≥1.5 视为高质量）的丢掉。

只保留 `allow_domains` 里的学术站点。CSDN、GitHub、博客和个人主页不在表里，直接丢掉。arXiv 链接会认 `/abs`、`/pdf`、`/html`、版本号和 `#fragment`，然后用 arXiv API 把摘要补全，方便后面的嵌入和 LLM。期刊链接能从 URL 抽出 DOI 就用 DOI；否则按标题问 OpenAlex，不放心再问 Crossref，对不上就只留标题，交给后面的模糊去重。返回字段里的 `date` 可能是抓取时间，不当作发表日期。

搜索按次计费（约 $0.75 / 1000 次成功请求），不按返回条数。`max_requests` 默认 40（11 个子方向 × 3 ≈ 33）。没配 `LLM_API_KEY`、单条查询失败、或额度用完，都只记日志，不影响其他源。

Semantic Scholar 的限速在 `paper_radar/s2.py`：全进程一个时钟，间隔 `min_interval`（1.1 秒），429 读 `Retry-After` 或指数退避，单次运行超过 `max_requests`（300）就停止再请求并继续用已经拿到的数据。种子摘要写在 SQLite 缓存里，第二天不会重复拉。

## 反馈和周报

卡片底部有 paper key（如 `arxiv:2401.00001`）：

```bash
python -m paper_radar feedback like arxiv:2401.00001
python -m paper_radar feedback dislike arxiv:2401.00001
python -m paper_radar feedback like 2401.00001 --topic scen_diffusion
python -m paper_radar recent
python -m paper_radar weekly --dry-run
```

点赞会变成该子方向的 S2 正种子和嵌入文本，点踩变成负种子。也可以用 Actions → **paper-radar feedback** 手动触发。

`weekly.weekday: 0` 表示周一（北京时间）附带一份近 7 天周报。`-1` 关闭。`weekly` 子命令不看星期几，随时可以生成。

## 测试

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

单测不访问外网（LLM、嵌入、HTTP、飞书、S2 都是假的）。CI 在 `.github/workflows/ci.yml`。

## 以后接 Zotero

1. `pip install pyzotero`。
2. 在 [zotero.org/settings/keys](https://www.zotero.org/settings/keys) 拿 user id 和只读 key，存成 `ZOTERO_ID`、`ZOTERO_KEY`，并加到 workflow 的 env。
3. `config.yaml`：

```yaml
zotero:
  enabled: true
  library_type: user
  collection_map:
    "规划-LLM基元": plan_llm_primitive
    "场景-扩散": scen_diffusion
```

分类里的论文会成为对应子方向的正种子。别的来源实现 `InterestSource.contribute(topics)`，在 `interests/base.py` 的 `build_topics` 里注册即可。

## 调配置

- `topics[].description` 用英文写清对象和方法；`positive` 用 S2 认的 id（`ARXIV:2310.03026`、`DOI:10.…`）。
- `ranking.top_k` 决定 LLM 成本，`top_n` 是每天推送篇数，`per_topic_max` 防止一个子方向占满。
- 期刊分级在 `paper_radar/rank/signals.py` 的 `TIER1/2/3`。

## 已知限制

- 没配 LLM key 就没有模型写的中文 TL;DR，DeepSeek 网页搜索也会跳过。
- 领域过滤看的是标题和摘要里的道路车辆用语。一篇确实在做驾驶、但通篇不写 vehicle / driving / traffic / 常见基准名字的论文会被丢掉。
- 付费墙论文如果 OpenAlex / Unpaywall / S2 都没有开放获取 PDF，卡片只给 DOI，不会给一个打不开的「PDF」按钮。
- arXiv 建议 3 秒一次请求；被限流时改抓 `arxiv.org/list/<cat>/new`（只有最近一次公告）。
- 不配 `S2_API_KEY` 时推荐接口经常 429。限速和退避之后仍失败，就跳过剩余子方向，不影响 arXiv / OpenAlex。
- OpenAlex 收期刊有几天延迟，`journals.lookback_days` 默认 10 天。

## License

MIT，见 [LICENSE](LICENSE)。
