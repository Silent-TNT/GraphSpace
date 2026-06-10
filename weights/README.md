# weights/ — 本地模型权重与推理输出

此目录**不会**上传至 GitHub。

## 推荐结构

```text
weights/
├── spatial_modal_cvae_v4_88x88x24.pth      # 训练权重（已被 *.pth 规则忽略）
├── spatial_modal_cvae_v3_88x88x24_probe_best.pth
├── outputs/                                 # CLI 推理输出（png/json/html）
├── outputs_web/                             # Web 演示用输出
├── eval_reports/                            # 权重对比与探针评估报告
└── *.log                                    # Gradio 运行日志
```

## 说明

- `.pth` / `.pt` 文件由全局 `*.pth` 规则忽略
- `outputs*/`、`eval_reports/`、`.log` 等推理产物由 `weights/*` 规则忽略
- 推理代码见 `scripts/spatial_modal_infer/`，配置模板见同目录下的 `model_config_*.json`
