# Adaptive_Reasoning_Research

## 🧭 项目总体结构
Adaptive Reasoning in Small LMs
│
├── phase_1_baseline/        ← 阶段一：TinyLlama 复现与基线测评
│   ├── notebooks/
│   │   ├── tinyllama_inference.ipynb
│   │   └── flops_profile.ipynb
│   ├── logs/
│   │   └── performance_metrics.csv
│   ├── reports/
│   │   └── replication_report.md
│   └── slides/
│       └── nov_meeting_talk.pptx
│
├── phase_2_dynamic_control/ ← 阶段二：置信度动态停止控制模块
│   ├── src/
│   │   ├── controller.py         # 实现基于熵的早停逻辑
│   │   ├── utils.py
│   │   └── visualization.py
│   ├── experiments/
│   │   ├── run_entropy_thresholds.py
│   │   ├── config_high.yaml
│   │   └── config_low.yaml
│   └── results/
│       ├── layer_usage_plot.png
│       └── confidence_histogram.png
│
├── phase_3_evaluation/      ← 阶段三：系统实验与能效分析
│   ├── datasets/
│   │   ├── gsm8k/
│   │   ├── boolq/
│   │   └── arc_easy/
│   ├── run_experiments.py
│   ├── analysis/
│   │   ├── compare_baselines.ipynb
│   │   └── plot_accuracy_flops.py
│   └── results/
│       └── efficiency_curve.png
│
├── phase_4_paper/           ← 阶段四：论文撰写与投稿
│   ├── draft/
│   │   ├── 00_abstract.md
│   │   ├── 01_intro.md
│   │   ├── 02_method.md
│   │   ├── 03_experiments.md
│   │   └── 04_conclusion.md
│   ├── bib/
│   │   └── references.bib
│   └── figures/
│       └── flops_tradeoff.png
│
└── tools/
    ├── measure_latency.py       # 通用性能测试工具
    ├── entropy_analyzer.py      # 计算输出分布的熵
    ├── flops_estimator.py       # 层级 FLOPs 统计
    └── wandb_logger.py
