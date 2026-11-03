# 📡 paper-radar：自动驾驶方向每日高质量论文雷达 + 飞书推送

每天自动从 **arXiv / OpenAlex / 目标期刊 / Semantic Scholar 推荐** 召回新论文 → 去重 → 按课题组 11 个子方向做嵌入预筛（**取各子方向最大值**）→ 只对 Top-K 调 LLM 精审打分（相关度/新颖性/实验扎实度 + 中文 TL;DR）→ 叠加质量信号（录用刊会、作者 h-index、HF 点赞、基准/真车实验）→ MMR 去冗余 → 按「运动规划 / 运动控制 / 场景生成」分组推送到**飞书群机器人卡片**，同时生成 Markdown/HTML 日报。

**不依赖 Zotero**：兴趣画像写在 `config.yaml`（每个子方向 = 一段描述 + 正/负种子论文）。兴趣来源是可插拔的，以后接 Zotero 只需打开配置（见文末）。

```
sources/   arxiv(API+列表页兜底) · openalex(按子方向检索) · journals(OpenAlex 按 ISSN: T-ITS/T-IV/RA-L/T-RO/TVT/TCST) · semantic_scholar(种子推荐) · rss(通用插槽)
interests/ config(config.yaml) · feedback(👍👎 回灌) · zotero(可选) · seeds(种子摘要补全)
rank/      embed(API → sentence-transformers → TF-IDF) · signals(质量信号) · llm(DeepSeek 等精审) · score(MAX 预筛 + 加权 + MMR)
push/      feishu(签名 + 交互卡片) · report(Markdown/HTML)
store.py   SQLite：已推送历史（不重复推送）、反馈、缓存、运行日志
```

## 一、快速开始（本地）

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export LLM_API_KEY=sk-...                    # 可选；不设则用启发式评分
export LLM_BASE_URL=https://api.deepseek.com # 默认就是 DeepSeek
export LLM_MODEL=deepseek-chat
python -m paper_radar topics                 # 查看子方向
python -m paper_radar run --dry-run          # 跑一遍：生成 reports/日期.md/.html 和 飞书卡片 JSON，不推送
python -m paper_radar run                    # 正式推送（需 FEISHU_WEBHOOK）
```

## 二、部署到 GitHub Actions（每天北京时间 08:30 自动推送）

1. 把本项目推到你自己的 GitHub 仓库（建议 **Private**，因为 `data/state.sqlite` 会记录你的阅读反馈）。
2. 仓库 **Settings → Secrets and variables → Actions → New repository secret**，添加：

| Secret | 必需 | 说明 |
|---|---|---|
| `FEISHU_WEBHOOK` | ✅ | 飞书自定义机器人 Webhook 地址 |
| `FEISHU_SECRET` | 推荐 | 机器人「签名校验」密钥 |
| `LLM_API_KEY` | 推荐 | DeepSeek / 通义 / 硅基流动等 OpenAI 兼容 key |
| `LLM_BASE_URL` | 可选 | 默认 `https://api.deepseek.com/v1`；Qwen: `https://dashscope.aliyuncs.com/compatible-mode/v1`；硅基流动: `https://api.siliconflow.cn/v1` |
| `LLM_MODEL` | 可选 | 默认 `deepseek-chat` |
| `S2_API_KEY` | 可选 | [Semantic Scholar API key](https://www.semanticscholar.org/product/api#api-key-form)（免费申请，强烈建议，否则常被限流） |
| `OPENALEX_API_KEY` / `OPENALEX_EMAIL` | 可选 | [OpenAlex key](https://openalex.org/settings/api)（免费；无 key 时共享 IP 的每日额度） |
| `EMBED_API_KEY` / `EMBED_BASE_URL` / `EMBED_MODEL` | 可选 | 嵌入 API，如硅基流动 `BAAI/bge-m3`；不填用 TF-IDF |

3. **Settings → Actions → General → Workflow permissions** 选 *Read and write*（用于把 SQLite 状态提交回仓库）。
4. 打开 **Actions** 页启用工作流，先手动 *Run workflow* 一次（`dry_run=true` 可只生成报告）。
5. 之后每天 00:30 UTC（北京时间 08:30）自动运行；GitHub 定时任务可能延迟 5–30 分钟。

### 创建飞书群机器人
1. 飞书里新建（或选一个）群 → 右上角 **设置 → 群机器人 → 添加机器人 → 自定义机器人**。
2. 起名（如「论文雷达」），复制 **Webhook 地址** → 存为 `FEISHU_WEBHOOK`。
3. 安全设置勾选 **签名校验**，复制密钥 → 存为 `FEISHU_SECRET`（也可用「自定义关键词」，填 `Paper Radar` 即可）。
4. 只想自己看：建一个只有自己的群即可。
> 限制：自定义机器人 100 次/分钟、5 次/秒，单条消息 ≤ 20KB；本项目每个大方向发一张卡片（每天 ≤3 条）。

## 三、调优指南（`config.yaml`）

- **子方向 `topics`**：`description` 用英文写清楚研究对象/方法/关键词（检索与嵌入都是英文语料）；`positive` 放 S2 可识别的种子（`ARXIV:2310.03026` / `DOI:10.xxx`），**种子越准推荐越准**，每个方向建议 3–10 篇；`negative` 放不想看的内容描述。
- **召回范围**：`sources.arxiv.full_categories`（全量）、`keyword_categories + keywords`（大类只取含关键词的）、`lookback_days`。
- **数量**：`ranking.top_k`（进 LLM 精审的数量，决定 token 成本）、`top_n`（每天推送数）、`per_topic_max`、`mmr_lambda`（越小越多样）、`min_relevance`。
- **权重**：`ranking.weights`（LLM 相关度/新颖性/实验 + 嵌入 + 质量信号）。质量信号的刊会分级在 `rank/signals.py` 的 `TIER1/2/3`。
- **成本**：DeepSeek 每天 40 篇 × ~1.5k token ≈ 几分钱人民币。

## 四、反馈闭环（越用越准）

每篇卡片底部有 paper key（如 `arxiv:2610.12345`）：

```bash
python -m paper_radar feedback like arxiv:2610.12345            # 👍，沿用推送时的子方向
python -m paper_radar feedback dislike arxiv:2610.12345
python -m paper_radar feedback like 2405.01234 --topic scen_diffusion   # 没推过的论文也能当正样本
python -m paper_radar recent                                     # 最近推送
```
👍 会成为该子方向的 S2 推荐正种子、并加入嵌入画像；👎 成为负种子。在 GitHub 上也可以直接用 **Actions → paper-radar feedback → Run workflow** 提交反馈（手机 GitHub App 也行）。

## 五、以后接入 Zotero

1. `pip install pyzotero`（并加到 `requirements.txt`）。
2. 在 [zotero.org/settings/keys](https://www.zotero.org/settings/keys) 获取 userID 和只读 API key，设为 Secrets `ZOTERO_ID`、`ZOTERO_KEY`（并在 workflow 的 env 中添加这两项）。
3. 在 `config.yaml` 中：
```yaml
zotero:
  enabled: true
  library_type: user
  collection_map:            # Zotero 分类名 -> 子方向 id
    "规划-LLM基元": plan_llm_primitive
    "场景-扩散": scen_diffusion
```
对应分类里的每篇论文会作为该子方向的正种子（arXiv/DOI）+ 画像文本。要加其他来源（BibTeX、Notion…），在 `paper_radar/interests/` 里实现一个 `InterestSource.contribute(topics)` 并在 `base.build_topics` 注册即可。

## 六、已知限制
- arXiv API 对同一 IP 有限流（429）；被限流时自动降级为抓取 `arxiv.org/list/<cat>/new` 列表页（只含最近一次公告日）。
- Semantic Scholar 无 key 时经常 429；推荐/h-index 失败会自动跳过，不影响主流程。建议申请 `S2_API_KEY`。
- OpenAlex 对正式期刊论文的收录有数天延迟，所以 `journals` 默认回看 10 天。
- 不设 LLM key 时为启发式评分（嵌入相似度 + 实验/刊会信号），没有中文 TL;DR。
- 这里给的期刊/会议分级是经验性的，可以按组里的偏好改 `signals.py`。
