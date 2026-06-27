# data/ - 本地数据集目录

此目录**不会**上传至 GitHub，请在本地自行维护。

## 推荐结构

```text
data/
├── raw/              # 原始 Rhino 导出或未清洗 JSON（可选）
├── processed/        # 通过 QC 的 house_*.json，供校验与训练使用
└── phase1/           # V5 训练前审计、关系、固定划分和切割上限产物
```

## 使用方式

1. 使用 `scripts/rhino_export/260524rhino-json-v14.py` 在 Rhino 中导出 JSON，
   保存至 `processed/`。
2. 运行离线校验：

   ```bash
   python scripts/offline_check/validate_json_dataset.py --data-dir data/processed
   ```

3. 训练 notebook（`notebooks/train/`）默认从本目录读取 JSON；预处理生成的
   `.pt` 缓存也建议放在本地 `data/` 下（已被 `.gitignore` 排除）。
4. V5 训练前第一阶段审计：

   ```bash
   python scripts/data_phase1/run_phase1.py
   ```

   结果见 `data/phase1/REPORT.md` 和 `data/phase1/split_v1.json`。

5. 构建 V5 双层统一画布并执行真值往返检查：

   ```bash
   python scripts/data_phase2/build_v5_representation.py
   ```

   结果见 `data/phase2_v5/summary.json`。每套样本包含：

   - `site_mask`：用户输入长宽对应的矩形可建边界；
   - `class_grid`：两层的边界内空值与 11 类功能空间；
   - `building_mask`：每层所有功能体块的平面并集；
   - `instance_grid`：房间实例监督；
   - `cross_floor_mask`：楼梯及其他跨层空间；
   - `double_height_void_mask`：非楼梯挑空空间；
   - `floor_overlap_mask`：一二层平面重叠区域。

6. 校准结构多样性指标：

   ```bash
   python scripts/data_phase3/evaluate_structural_diversity.py
   ```

   结果见 `data/phase3_diversity/calibration.json`。该评估会区分真正的轮廓、
   分割和上下层结构变化，与仅交换功能标签的伪多样性。

7. 运行统一候选评估器回归：

   ```bash
   python scripts/data_phase4/validate_unified_evaluator.py
   ```

   结果见 `data/phase4_evaluation/validation_summary.json`。评估器统一检查
   P0 硬约束、P1 功能体块空间组织、P2 质量通过率、实例恢复和同条件多 seed
   结构多样性。

8. 构建 V5 房间实例监督：

   ```bash
   python scripts/data_phase5/build_instance_supervision.py
   ```

   结果见 `data/phase5_instances/summary.json` 和
   `data/phase5_instances/samples/`。每套样本增加：

   - `center_heatmap`：每层房间中心热力图；
   - `center_offset`：已占用格子指向所属房间中心的二维偏移；
   - `center_valid_mask`：实例偏移损失的有效区域；
   - `boundary_mask`：房间外边界和房间之间的分界；
   - `floor_instance_counts`：每层实例总数；
   - `class_instance_counts`：每层每类实例数量。

9. 运行 V5 最小训练链路：

   ```bash
   python scripts/train_v5/train.py --smoke-test --device cpu
   ```

   完整参数和本地/Colab运行方法见 `docs/V5_MINIMAL_TRAINING.md`。

10. 将 V5 权重输出转换为标准房间 JSON 并统一评估：

   ```bash
   ./.venv-v5-cuda/Scripts/python.exe \
     scripts/train_v5/evaluate_standard_pipeline.py \
     --checkpoint outputs/v5_instance_overfit_8/best.pt \
     --split train --max-samples 8 --device cuda \
     --output-dir outputs/v5_standard_pipeline_overfit_8_final
   ```

## 说明

原始户型数据涉及学术研究与建模规范，在论文正式发表前不对外公开。
