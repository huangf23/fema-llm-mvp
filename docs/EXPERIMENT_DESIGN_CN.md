# FEMA 问卷数据的 LLM 受灾人口决策模拟实验设计

本文档整理当前项目中已经形成的实验构想，用于研究大语言模型能否基于有限受访者信息，模拟 FEMA National Household Survey 中的灾害准备状态判断。

## 1. 研究目标

核心研究问题：

> 在应急管理场景中，LLM 能否根据受灾人口或潜在受灾人口的有限基础信息，预测其灾害准备状态，并复现真实问卷结果中的个体差异与总体分布？

进一步的问题包括：

- 最少需要披露哪些信息，才能让 LLM 预测结果接近真实问卷？
- LLM 的预测能力是否优于简单统计基线？
- LLM 是否存在系统性偏差，例如过度预测 Prepared 或 Unprepared？
- 传统统计模型的信息是否可以帮助校准 LLM 模拟？
- 不同模型、温度、是否思考、重复采样等模型因素是否影响结果稳定性？
- 跨年份数据合并后，结论是否仍然稳健？

## 2. 数据来源

当前 MVP 使用 FEMA National Household Survey 公开数据。

当前主实验数据：

- 年份：2023
- 表：`Core Survey`
- 原始样本量：7604
- 目标变量：`dis_2_prepstages`
- 有效二分类样本量：7217
- Prepared Individuals：3724
- Unprepared Individuals：3493
- Unknown：387，主实验中排除

可扩展数据：

- 2017-2023 年总有效二分类目标样本约 40334
- 2022-2023 字段对齐最好
- 2021-2023 可以作为较稳健的跨年 pooled data
- 2017-2020 需要较多变量 harmonization

## 3. 目标变量

当前目标是预测受访者是否属于：

- `Prepared Individuals`
- `Unprepared Individuals`

2023 年直接使用 FEMA 已计算好的二分类字段：

```text
dis_2_prepstages
```

跨年份时可以由以下变量折算：

- 2017：`ST_STG1`
- 2018：`st_stg1`
- 2019：`st_stg1`
- 2020：`st_stg1`
- 2021：`ST_STG1`
- 2022：`_2_prep_stages_general`
- 2023：`dis_2_prepstages`

折算原则：

- 已经准备一年以内、超过一年、持续准备：Prepared
- 未准备但打算准备、未准备且不打算准备：Unprepared
- Don't know、Prefer not to answer、Unknown：排除或作为单独敏感性分析

## 4. 主实验：渐进式信息披露

主实验采用 progressive disclosure，让 LLM 在不同信息量下预测同一个受访者的灾害准备状态。

### C0：无个体信息

输入内容：

- 灾害准备问题描述
- 两个可选标签
- 不提供任何受访者信息

目的：

- 检查 LLM 的先验倾向
- 观察模型在没有个体信息时是否偏向 Prepared 或 Unprepared
- 对应 majority baseline 或总体先验

### C1：基础人口学信息

在 C0 基础上加入：

- 年龄
- 性别
- 教育
- 种族
- 族裔
- 收入

目的：

- 测试人口学画像是否足以模拟部分灾害准备差异
- 观察 LLM 是否依赖社会经济刻板规律

### C2：地理、风险感知与灾害经历

在 C1 基础上加入：

- 州
- 地理分区
- Census region
- 城乡属性
- 感知灾害影响可能性
- 是否经历过灾害影响

目的：

- 测试环境暴露、风险感知和经验是否提高预测准确性
- 判断 LLM 是否能够利用灾害情境变量

### C3：准备效能、信心、家庭约束与资源

在 C2 基础上加入：

- 认为准备措施是否有帮助
- 对采取准备措施的信心
- 是否有残障或健康限制
- 是否承担照护责任
- 成年人数
- 儿童人数
- 租房或自有住房
- 是否有住房保险

目的：

- 测试最完整的个体与家庭画像能否接近真实问卷结果
- 识别心理变量和家庭资源变量对 LLM 模拟的边际贡献

## 5. 当前 n=1600 实验结果

当前已完成一次 2023 年 stratified sample：

- 受访者数：1600
- 披露条件：C0-C3
- prompt 数量：6400
- 模型：DeepSeek V4 Flash
- 并发：10

结果路径：

```text
work/outputs/n1600/
```

LLM 结果：

| 条件 | Accuracy | Macro-F1 | 真实 Prepared 比例 | 预测 Prepared 比例 |
|---|---:|---:|---:|---:|
| C0 | 0.481 | 0.335 | 0.516 | 0.015 |
| C1 | 0.556 | 0.555 | 0.516 | 0.429 |
| C2 | 0.616 | 0.590 | 0.516 | 0.736 |
| C3 | 0.633 | 0.612 | 0.516 | 0.719 |

初步解释：

- LLM 预测表现随信息披露增加而提升，说明 progressive disclosure 是有效实验设计。
- C0 几乎不预测 Prepared，说明模型先验与样本分布不一致。
- C2 和 C3 明显高估 Prepared，说明模型存在分布校准问题。
- C3 的个体分类能力提高，但总体分布复现仍不理想。

## 6. 传统基线实验

为避免只报告 LLM 结果，需要加入传统机器学习基线。

当前 baseline：

- Majority baseline
- Logistic regression
- Random forest

评估方式：

- 使用与 LLM 相同的样本
- 使用与 C1-C3 对应的同一组字段
- Stratified cross-validation
- C0 只能使用 majority baseline

n=1600 baseline 结果：

| 方法 | 条件 | Accuracy | Macro-F1 | 预测 Prepared 比例 |
|---|---|---:|---:|---:|
| Majority | C0-C3 | 0.516 | 0.340 | 1.000 |
| Logistic regression | C1 | 0.581 | 0.581 | 0.494 |
| Random forest | C1 | 0.543 | 0.543 | 0.499 |
| Logistic regression | C2 | 0.618 | 0.617 | 0.505 |
| Random forest | C2 | 0.623 | 0.623 | 0.503 |
| Logistic regression | C3 | 0.663 | 0.663 | 0.523 |
| Random forest | C3 | 0.657 | 0.657 | 0.509 |

解释：

- 传统模型在 C3 条件下略优于当前 DeepSeek V4 Flash。
- Logistic regression 和 random forest 的 Prepared 比例更接近真实分布。
- LLM 的价值不一定是单纯分类准确率，而是作为可解释、可提示、可模拟的行为生成模型。

## 7. 回归信息增强 LLM 实验

可以设计一个扩展实验：将传统统计模型学到的群体规律作为额外信息输入 LLM。

核心问题：

> LLM 在个体信息之外，如果获得统计模型总结出的群体规律，是否能更接近真实问卷结果？

需要避免信息泄露：

- 先划分 train/test
- 在 train 上训练 logistic regression 或 random forest
- 只把 train 得到的规律输入 LLM
- 在 test 上评估 LLM 预测

### C4a：回归方向摘要

输入 C3 信息，再加入由训练集得到的简短统计规律，例如：

```text
In the training data, respondents are more likely to be prepared when they:
- believe preparedness steps help a great deal;
- feel confident about taking preparedness steps;
- have experienced disaster impacts;
- have higher household income;
- own their home or have residence insurance.

Use these tendencies only as background information. Make the final prediction for this specific respondent.
```

目的：

- 测试统计规律是否能校准 LLM 判断
- 观察 LLM 是否能合理整合群体层面的信息

### C4b：加入 logistic regression 预测概率

输入 C3 信息，再加入：

```text
The logistic regression model trained on other respondents estimates:
P(Prepared Individuals) = 0.67.
```

目的：

- 测试 LLM 是否能利用数值概率
- 比较 LLM 是否只是复述概率，还是会结合个体叙述重新判断

### C4c：概率加特征贡献

输入 C3 信息，再加入：

```text
The logistic regression model trained on other respondents estimates:
P(Prepared Individuals) = 0.67.

Main factors increasing this probability:
- confidence in preparedness steps;
- belief that preparedness helps;
- residence insurance.

Main factors decreasing this probability:
- low household income.
```

目的：

- 将传统模型的可解释性转化为 LLM prompt
- 测试 LLM 作为“统计模型解释器 + 个体模拟器”的潜力

## 8. 模型因素实验

LLM 结果可能受到模型端设置影响，因此需要显式记录和比较。

可测试因素：

- 不同模型：DeepSeek、Qwen、Llama、Gemini、GPT 等
- 是否启用 thinking / reasoning
- temperature：0、0.3、0.7
- top_p
- max_tokens
- JSON mode
- seed
- repeat 次数
- prompt language：英文、中文、双语
- prompt 风格：简洁型、角色型、统计型、严格分类型

建议顺序：

1. 固定 DeepSeek V4 Flash，temperature=0，跑主实验。
2. 加入 1-2 个开源模型，比较模型差异。
3. 对最有希望的模型做 temperature 和 repeat 稳定性实验。
4. 最后再测试 thinking 是否改善结果。

## 9. 跨年份合并实验

当前最稳妥的扩展路径：

### 阶段一：2023 单年主实验

优点：

- 变量最完整
- 字段名清晰
- 数据口径最干净

用途：

- 作为论文主实验
- 建立 progressive disclosure 的基本证据

### 阶段二：2021-2023 pooled data

优点：

- 样本量显著增加
- 变量结构相对接近
- 可以测试跨年份稳健性

需要做：

- 统一 ID 为 `year_id`
- 加入 `survey_year`
- 将 `ST_STG1`、`_2_prep_stages_general`、`dis_2_prepstages` 统一为 `prepared_binary`
- 对人口学、地理、风险感知、灾害经历、家庭变量做字段映射

### 阶段三：2017-2023 全年份 harmonization

优点：

- 样本量最大，约 40334 个有效二分类目标样本
- 可以做跨时间稳健性分析

风险：

- 早期年份变量名旧，如 `D2`、`D3`、`D4A`、`D5`、`D8`、`D13`
- 部分变量选项编码不同
- 2022-2023 有派生变量，早期年份需要手工重建

建议：

- 不作为第一篇论文主实验
- 可以作为 robustness check 或 appendix

## 10. 评估指标

个体层面：

- Accuracy
- Macro-F1
- Per-class precision / recall / F1
- Confusion matrix

总体分布层面：

- 真实 Prepared rate
- 预测 Prepared rate
- Prepared rate error
- Absolute prepared rate error

分组公平性与异质性：

- 年龄组
- 教育
- 收入
- 残障状态
- 城乡
- 州或区域
- 灾害经历

稳定性：

- 同一模型多次重复的一致率
- 不同 temperature 下的方差
- 不同模型之间的一致率

校准：

- 如果输出概率，可计算 Brier score
- 比较 predicted prepared rate 和 true prepared rate
- 比较 C3 与 C4 的分布误差变化

## 11. 论文中的主要解释路径

可以围绕三条线展开。

第一，信息披露效应：

- C0 到 C3 性能逐步提升
- 说明 LLM 能利用受访者信息改善个体模拟
- 哪些信息层最有贡献，可以对应应急管理理论

第二，LLM 与传统模型的差异：

- 传统模型在准确率和总体分布校准上可能更强
- LLM 的优势在于可提示、可解释、可进行情景模拟
- LLM 不应被简单视为传统分类器替代品

第三，分布偏差：

- LLM 可能系统性高估或低估 Prepared
- 这种偏差本身是研究对象
- 应急管理中的 AI 模拟不能只看个体 accuracy，还要看总体分布是否接近真实人群

## 12. 推荐的论文实验结构

主实验：

```text
2023 data
C0 / C1 / C2 / C3
DeepSeek V4 Flash
Majority / Logistic regression / Random forest baselines
```

扩展实验一：

```text
C3 vs C4a / C4b / C4c
检验统计模型信息是否能校准 LLM
```

扩展实验二：

```text
不同模型
不同 temperature
thinking on/off
repeat stability
```

扩展实验三：

```text
2021-2023 pooled data
检验跨年份稳健性
```

附录或后续研究：

```text
2017-2023 full harmonized data
更大样本下的时间稳健性和变量口径敏感性
```

## 13. 当前代码对应关系

数据与 prompt：

```bash
python prepare_data.py --n 1600 --seed 42 --output-dir work/outputs/n1600
```

LLM 推理：

```bash
python run_llm.py \
  --profile deepseek_v4_flash \
  --prompts work/outputs/n1600/prompts.jsonl \
  --output work/outputs/n1600/predictions_deepseek_v4_flash.jsonl \
  --concurrency 10
```

评估：

```bash
python evaluate.py \
  --predictions work/outputs/n1600/predictions_deepseek_v4_flash.jsonl \
  --output-prefix deepseek_v4_flash \
  --sample work/outputs/n1600/sample.csv
```

传统基线：

```bash
python run_baselines.py \
  --sample work/outputs/n1600/sample.csv \
  --output-dir work/outputs/n1600
```

## 14. 下一步实现建议

优先级 1：

- 固化 2023 主实验
- 跑全量 7217 个有效样本
- 输出主表和图

优先级 2：

- 实现 C4 回归增强 LLM
- 避免 train/test 信息泄露
- 比较 C3 与 C4 的提升

优先级 3：

- 实现 `year_harmonizer.py`
- 先支持 2021-2023
- 再扩展到 2017-2020

优先级 4：

- 增加多模型配置
- 加入 repeat stability
- 增加 subgroup error 分析

## 15. 可能的投稿定位

这项研究可以定位为：

- AI for emergency management
- computational social science
- survey simulation using LLM agents
- human behavior modeling under disaster preparedness contexts
- hybrid statistical and LLM-based simulation

论文贡献可以写成：

1. 提出一个用于灾害准备行为问卷模拟的渐进式信息披露框架。
2. 比较 LLM、majority baseline、logistic regression 和 random forest 在个体预测和总体分布复现上的差异。
3. 发现 LLM 能随信息增加改善预测，但可能存在明显分布校准偏差。
4. 探索统计模型信息增强 LLM 模拟的可能性。
5. 为应急管理中的 AI 人群模拟提供可复现的公开数据实验范式。
