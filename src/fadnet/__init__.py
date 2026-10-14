"""FADNet-RT：实时布匹缺陷检测（设计文档：docs/FADNet-RT_design.md）。

架构 = 预处理层 P + S 分支(YOLO27s 风格监督检测) + U 分支(Dinomaly 重构
+ PatchCore 记忆库 双流无监督) + 布种条件化 FC + 融合裁决 F-Head。
"""

__version__ = "0.1.0"
