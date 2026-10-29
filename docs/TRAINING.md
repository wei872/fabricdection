# ViT-S：已知检测、正常学习与未知疵点

## 1. 数据合同与工况划分

模型只有 `configs/jingwei_dinov2.yaml`，数据路径在 `configs/fabric19.yaml`。
路径相对仓库根目录或为绝对路径，不自动拼接数据 YAML 的 `path` 字段。

```yaml
train: D:/fabric/images/train
val: D:/fabric/images/val
normaltrain: D:/fabric/normal/train
normalcal: D:/fabric/normal/cal
normalval: D:/fabric/normal/test
unknownval: D:/fabric/images/unknown_test
```

缺陷图对应 `images/train → labels/train`、`images/val → labels/val` 等目录，标签一行
`class_id cx cy width height`（归一化），类别 `0..18`。类别名字／顺序以模型配置 `data.names` 为准。
读者将缺失标签当背景，因此务必检查漏标和标签路径；缺失标签的图不会被自动当作可信正常纹理。
`normaltrain` 仅放人工确认无疵点的图；程序自动产生空 boxes／labels，并标记正常图更新原型。

`train.normal_ratio: 0.5` 表示每轮抽取与缺陷图同等数量的正常图，正常池内随机抽样。
正常图同时贡献背景检测监督和正常原型更新；带标注的缺陷图只训练检测头，不更新正常原型。
正常池与缺陷池共享 RNG，支持 worker 独立种子和 checkpoint 恢复。

**必须按物理原图／布卷分组，不可按增广图片随机切分。** 至少记录：

| 分组变量 | 训练与校准应覆盖 | 最终独立验收 |
|---|---|---|
| 经纬密度 | 多种真实密度／纱线粗细 | 留出密度区间／布卷 |
| 光照 | 正常曝光、亮暗、不同方向、反光 | 留出灯光组合／采集批次 |
| 布种／颜色 | 多种干净正常图和缺陷图 | 留出布种或新卷 |
| 已知缺陷 | 每类都有标注 | 每类 P/R 与小缺陷召回 |
| 未知疵点 | 不用于检测训练或阈值选取 | 留出整类缺陷作为 unknownval |

训练增强包含独立经纬缩放（0.75–1.4）、同步框变换、曝光、gamma、色温扰动、阴影、反光、轻微模糊／锐化。
经纬缩放发生在固定尺寸 resize **之后**，避免被整图 resize 抵消；缩小时使用常量填充，不镜像复制疵点。
这只是工况模拟：过强光照增强可能让类似污点的变化变成“正常”，参数应以真实验证结果调整。
真实光照／密度多样性不能靠合成增强保证。

## 2. 训练

```bat
python tools/train_detector.py --data configs/fabric19.yaml --epochs 50 --batch 2 --val-batch 1 --num-workers 4 --amp off --out runs/dinov2s
```

默认官方预训练 `dinov2_vits14`，冻结 encoder，训练适配层／PAN／检测头。
仅可信 `normaltrain` 特征以在线聚类更新 32 个正常原型。未知模块不需要独立 teacher、蒸馏或第二次 ViT。
启用未知模块时要求冻结主干，避免更新 encoder 后旧原型空间失效。

- GTX1660：**不支持 BF16**，默认 FP32；OOM 时减 `--batch 1`／`--val-batch 1`。
- `--amp fp16` 可单独比较，但不能假定它在 1660 上更快；训练混合精度不意味着最终推理必须 FP16。
- `--num-workers 0` 用于排查 Windows／读图问题；`--no-augment` 用于小样本过拟合检查。
- `--smoke N` 是合成数据测试，仍加载配置的 ViT；不是收敛或性能证明。
- 训练时未知阈值尚未校准，验证日志中的 FPR 是已有类别路径，不是最终开放集误报率。
- `best.pt` 按已知检测 `best_f1` 选择；未宣称按未知召回挑选最佳 epoch。

离线模式修改模型配置已有字段：

```yaml
dino_repo: C:/models/dinov2
dino_weights: C:/models/dinov2_vits14_pretrain.pth
```

本地源码须有 `hubconf.py`；权重须为匹配的官方 raw backbone state_dict。下载或严格加载失败会报错，
不会退回随机 ViT。`xFormers is not available` 一般表示标准 attention 回退，不是权重加载失败。

## 3. 恢复与格式变更

```bat
python tools/train_detector.py --resume runs/dinov2s/last.pt --epochs 50 --batch 2 --out runs/dinov2s
```

保持结构、数据和总 schedule 不变。恢复包括模型、正常原型、AdamW、scheduler、主进程随机状态、
可用时的 FP16 scaler；多 worker 预取／进程内部状态不保证逐 batch 位级一致。
不完整的 `--resume` 报错，不静默 warm start。

`--pretrained CHECKPOINT --lr ... --out NEW_DIR` 表示仅恢复完整检测模型权重、重新建立 optimizer／schedule。
原型再更新会使校准标志失效，结束后重新校准。
`best.pt` 和 `calibrated.pt` 是权重产物，只能用于推理或 `--pretrained`，不是完整 resume 文件。

**当前默认已从 ViT-B／1280／448 改为 ViT-S／640／224，并增加正常原型 buffers，需重新训练。**
旧 checkpoint 文件不删除，但不能直接按新配置 `--resume` 或 `--pretrained`；不使用 `strict=False` 掩盖不匹配。

## 4. 独立校准

```bat
python tools/calibrate_anomaly.py --data configs/fabric19.yaml --ckpt runs/dinov2s/best.pt --out runs/dinov2s/calibrated.pt --target-fpr 0.01
```

- `normalcal` 必须与训练／最终验证目录互不包含，程序检查目录重叠；不同目录里的内容重复仍需用户按来源排查。
- 不更新原型、不使用异常验证图。按**每张图最大 patch 距离**取有限样本顺序统计量。
- 默认 1% 目标至少需 99 张独立正常图，建议 200–500 张以上并覆盖实际工况；不是 99 张同图增广副本。
- 同分布／可交换性成立时才有该阈值的统计意义；新布种或新照明可能破坏假设，必须重测误报并重新校准。
- 目标只针对未知分支，已有类别检测也可能误报，因此组合系统 FPR 可能更高。
- `min_patches`、得分阈值、输入大小或精度变动后重新验证，输入／原型空间改变后重新校准。

FP16 推理要单独 `calibrate_anomaly.py --half`，并在 `predict.py`／`evaluate.py`／`bench_device.py` 同时传 `--half`。
程序拒绝校准和推理精度不一致。正常原型／阈值保留 FP32；DINO 特征精度仍会影响分数。
不把未知距离分数解释为缺陷类别置信概率。

## 5. 泛化与开放集验收

```bat
python tools/evaluate.py --data configs/fabric19.yaml --ckpt runs/dinov2s/calibrated.pt --batch 1 --diag
python tools/predict.py --ckpt runs/dinov2s/calibrated.pt --source D:/fabric_test/
```

输出：已知类别自定义 mAP50、P/R/F1，正常图组合 FPR、已知 FPR、未知 FPR；
若配置 `unknownval`，再输出未知框一对一 IoU 匹配的 precision／recall 和正样本图定位召回。
未知标签类别不参与匹配，但框不能缺失；没有未知标注集时报告 N/A，不能据此声称未知召回已验证。

`unknownval` 只放留出的新类型缺陷，不混入训练缺陷后称为“未知”。评价按布卷／密度／照明分组分别运行，
记录每组指标、最差组和整体指标。没有用户确认的召回率与误报上限，不宣称质量验收合格。
完整推理入口要求已校准；`--known-only` 明确关闭未知提示，不能用它的速度代表完整系统。

已知检测先受 `model.score_thresh`、`inference.final_score_thresh` 两层过滤。
`--val-conf` 只调整统计阈值，不能恢复已滤掉的框。mAP50 不是未截断的 COCO 全量指标；
`best_f1` 对应阈值不会自动写回推理配置。

## 6. 1660 速度验收

```bat
python tools/bench_device.py --ckpt runs/dinov2s/calibrated.pt --source D:/fabric_acceptance/ --device cuda --warmup 30 --iters 300 --target-fps 30 --require-target --out runs/1660_acceptance.json
```

代表帧需要同时覆盖正常、已有缺陷、未知疵点及不同密度／光照，包括多候选／多异常区域场景。
内存中循环使用已解码真实帧，计时包含 resize／RGB／tensor、H2D、ViT、检测解码／NMS、
未知距离和连通域、原图坐标、D2H。**不包含相机等待、磁盘解码、绘制／显示、写盘**。
报告 GPU 型号、精度、输入大小、候选数、平均／P50／P95／P99 延迟。

平均 ≥30 FPS 且 P95≤33.3ms 才能通过延迟检查；真实 1660、完整未知路径、真实输入且至少 10 张代表帧、10 次预热、100 次采样
才可能标记 `gtx1660_full_acceptance=true`，失败时 `--require-target` 返回 2。
质量另行验收；温度、功耗、其他 GPU 负载会影响复测结果。FP16 可另开完整校准／验证／测速实验，不假定加速。

若仍达不到目标，先根据实测瓶颈调整；增加 ViT 分辨率／切片通常提高开销，缩图可能漏微小疵点。
不能通过跳帧、关闭未知路径、改计时范围或牺牲召回来冒充完整 30 FPS。

## 7. 导出与测试边界

```bat
python -m unittest discover -s tests -v
python tools/export.py --ckpt runs/dinov2s/calibrated.pt --known-only --out runs/export/known.onnx --check
```

ONNX 目前仅实验性导出已有类别六张量，外部解码／NMS；不包含正常原型、校准阈值或未知区域处理。
真实模型的 ONNX/TensorRT/INT8 全流程没有验证。当前实现以 PyTorch/OpenCV 为准。
回归测试不下载权重；另做过官方源码随机 ViT-S 前后向接口检查，但没有官方权重／真实数据／1660 实测结果。


## 8. Loss 下降但召回低／部分轮次零出框

不要仅凭前几轮 loss 下降认定已收敛，也不能凭早期低召回断定预训练没有加载。
检测适配器／分类／定位头从随机权重开始，且分类、objectness、定位误差是不同因素。
当前分数是 `sigmoid(objectness) × max(sigmoid(class_logits))`。
模型先通过 `model.score_thresh` 和 `inference.final_score_thresh` 过滤后才进入指标计算；
因此仅传 `--val-conf 0.01` 无法找回被 0.15 最终过滤的候选。

使用已保存的 `best.pt` 运行独立诊断，不需要未知阈值校准：

```bat
python tools/evaluate.py --data configs/fabric19.yaml --ckpt runs/dinov2s/best.pt --known-only --candidate-conf 0.001 --val-conf 0.15 --score-stats --diag --batch 1 --workers 0
```

- `--candidate-conf`：只在本次评估中同时覆盖两层候选过滤，不写配置／checkpoint。
- `--val-conf`：仍为主要 P/R 的报告阈值；不指定时保留覆盖之前的配置值。
- `--score-stats`：按 val／normalval 分开报告**过滤前**的类别、objectness、联合分数的每图最大值分位数，
  以及多个阈值下 top-k 前的原始网格候选数。两个分量的各自最大值不一定来自同一个网格，不能直接相乘。
- 额外统计会同步／传输少量数据，不应用来测 FPS。

判断路径：

1. 若低阈值下出现 TP、`best_f1` 的阈值低于 0.15，说明置信过滤是因素之一，
   但是否改部署阈值仍要看独立验证的误报／召回权衡。
2. 若原始分数足够高、降低过滤仍没有 TP，先看 `FP breakdown`（错类、定位、背景、重复），
   再检查标签、缩放后的框、类别顺序和目标尺度；不能把全部问题归因于阈值。
3. 若各类诊断仍差，用 8–16 张真实已标注图＋少量正常图单独建立过拟合诊断集，关闭增广，
   训练／验证用相同图片仅排查实现与数据一致性；这不是泛化评测，不能拿其指标作最终结果。
4. 224 的 ViT 缩图可能损失细小纹理，但仅从汇总日志无法确定。增大输入需要独立速度／召回对比，
   不建议在 30 FPS 约束下直接换大模型或盲目增大分辨率。

`--candidate-conf` 会影响 mAP、PR 曲线和出框 FPR；诊断得到的提升不是训练收敛提升，
也不能与原过滤设置的 mAP 当作同一实验直接比较。
训练阶段未知模块尚未校准，`normal_fpr` 是当前已知检测路径的出框率；
`unknown_active=False` 明确表示没有评估最终未知疵点能力。零误报也可能只是几乎不出框。
