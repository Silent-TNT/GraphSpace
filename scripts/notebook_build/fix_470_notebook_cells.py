# -*- coding: utf-8 -*-
import json
import os
import textwrap

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "notebooks", "train", "260523-74条件生成与量化评估（MVP）.ipynb")
DST = os.path.join(ROOT, "notebooks", "train", "260607-470条件生成与量化评估（训练优化v1）.ipynb")
BUILD = os.path.join(os.path.dirname(__file__), "create_470_train_opt_v1.py")


def lines(code: str) -> list[str]:
    return [line + "\n" for line in textwrap.dedent(code).strip("\n").splitlines()]


def main():
    orig = json.load(open(SRC, encoding="utf-8"))
    nb = json.load(open(DST, encoding="utf-8"))

    build_src = open(BUILD, encoding="utf-8").read()
    step7b = build_src.split('STEP7B = """')[1].split('"""')[0]

    nb["cells"][13]["source"] = lines(
        """\
        ## Step 7: 量化评估（重建 vs 条件生成 best-of-K）

        - **重建模式**：GT 异构图（上界）
        - **生成模式**：仅用户约束 → 合成拓扑，**K 次采样取最优**（真实使用路径）
        - 验证集与训练共用 `train_val_split_v470.json`

        导出命名：`{报告名}_json{总数}_train{训练数}_{时间}.csv`
        """
    )

    step7a = "".join(orig["cells"][13]["source"])
    step7a = step7a.replace("print('Step 7a 就绪')\n", "").rstrip()
    step7a += """

@torch.no_grad()
def eval_conditional_generation_best_of_k(user_req, gt_voxel, model, k=None, base_seed=42):
    k = k or GEN_EVAL_K
    best = None
    best_score = (-1.0, -1.0, -1.0)
    trials = []
    for i in range(k):
        seed = base_seed + i * 17
        gen = eval_conditional_generation(user_req, gt_voxel, model, seed=seed)
        trials.append(gen)
        score = (gen['miou'], gen['program_acc'], float((gen['pred'] > 0).sum()))
        if score > best_score:
            best_score = score
            best = gen
    best = best or trials[0]
    best['eval_k'] = k
    best['trials'] = trials
    return best


print(f'Step 7a 就绪 | 条件生成 best-of-K={GEN_EVAL_K}')
"""
    nb["cells"][14]["source"] = lines(step7a)
    nb["cells"][15]["source"] = lines(step7b)

    # 删除误插入的重复 Step 7b，恢复 Step 8
    if len(nb["cells"]) > 16 and "Step 7b" in "".join(nb["cells"][16]["source"]):
        nb["cells"][16] = orig["cells"][15]
        nb["cells"][17] = orig["cells"][16]
        if len(orig["cells"]) > 17:
            nb["cells"][18] = orig["cells"][17]
        if len(orig["cells"]) > 18:
            nb["cells"][19] = orig["cells"][18]
        nb["cells"] = nb["cells"][:20]

    json.dump(nb, open(DST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("Fixed:", DST)
    for i, c in enumerate(nb["cells"]):
        print(f"{i:2d}", "".join(c["source"])[:70].replace("\n", " "))


if __name__ == "__main__":
    main()
