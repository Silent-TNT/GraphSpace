# -*- coding: utf-8 -*-
"""V4-a 诊断脚本：在投入 V4 大改前，先量出 V3 的真实失败模式。

只需一个权重文件即可运行（语义/解码器敏感性不依赖训练集）。
若提供 --data-dir（含 house_*.json），坍缩指标会用真实数据；否则用合成请求做代理统计。

用法:
    python diagnose_v4a.py --weights /path/to/spatial_modal_cvae_v3_88x88x24.pth
    python diagnose_v4a.py --weights w.pth --data-dir ../../data/processed --out report.json

三组指标:
  A 后验坍缩   mu_norm / sigma_mean / kl_per_dim / dead_dims_ratio / active_units_ratio
  B 编码器语义 改卧室数(3↔5)、改场地后 μ 的余弦距离
  C 解码器条件 固定 z 只改 cond 的体素变化率 vs 改 z 的变化率（关键：判断是否无视条件）
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .config import LATENT_DIM, NUM_CHANNELS
    from .graph_utils import graph_batch_dict, json_to_sample, prepare_graph_batch
    from .layout import build_user_request, layout_rooms_from_program, request_to_house_json
    from .pipeline import load_model
except ImportError:
    from config import LATENT_DIM, NUM_CHANNELS
    from graph_utils import graph_batch_dict, json_to_sample, prepare_graph_batch
    from layout import build_user_request, layout_rooms_from_program, request_to_house_json
    from pipeline import load_model

DEAD_KL_THRESH = 0.01
ACTIVE_VAR_THRESH = 0.01

BASE_COUNTS = {
    "entryway": 1, "living_room": 1, "dining_room": 1, "kitchen": 1,
    "bedroom": 3, "bathroom": 2, "corridor": 2, "stairs": 1, "balcony": 1,
}
SEMANTIC_SEED = 22

# 合成请求集（坍缩代理统计用，覆盖不同规模/卧室数/场地）
SYNTH_SPECS = [
    (15000, 12000, {"entryway": 1, "living_room": 1, "kitchen": 1, "bedroom": 2, "bathroom": 1, "corridor": 1, "stairs": 1}, 11),
    (18000, 15000, BASE_COUNTS, 22),
    (21000, 18000, {**BASE_COUNTS, "bedroom": 4, "utility": 1}, 33),
    (24000, 20000, {**BASE_COUNTS, "bedroom": 5, "bathroom": 3}, 44),
    (12000, 9000, {"living_room": 1, "kitchen": 1, "bedroom": 1, "bathroom": 1, "stairs": 1}, 55),
    (28000, 22000, {**BASE_COUNTS, "bedroom": 6, "multi_purpose": 1}, 66),
]


@torch.no_grad()
def encode_mu_logvar(model, graph, cond, device):
    batch = prepare_graph_batch(graph, condition=cond).to(device)
    mu, logvar = model.encoder(batch.x_dict, batch.edge_index_dict, graph_batch_dict(batch))
    return mu.detach(), logvar.detach()


@torch.no_grad()
def decode_cls(model, z, cond, device):
    c = cond.unsqueeze(0) if cond.dim() == 1 else cond
    logits = model.decoder(z.to(device), c.to(device))
    return torch.argmax(logits[0], dim=0).cpu().numpy()


def change_fraction(a, b) -> float:
    return float(np.mean(a != b))


def occ_by_type(cls) -> dict:
    return {int(c): int((cls == c).sum()) for c in range(1, NUM_CHANNELS) if (cls == c).any()}


def sample_from_request(site_x, site_y, counts, seed):
    req = build_user_request(site_x, site_y, counts)
    rooms, G, pos, et = layout_rooms_from_program(req, seed=seed)
    data = request_to_house_json(req, rooms)
    sample = json_to_sample(data, G, et)
    return sample


def collapse_from_mulogvar(mu_all, logvar_all) -> dict:
    sigma_all = torch.exp(0.5 * logvar_all)
    kl_dim = 0.5 * (mu_all.pow(2) + sigma_all.pow(2) - 1.0 - logvar_all)
    kl_per_dim = kl_dim.mean(0)
    mu_norm = mu_all.norm(dim=1)
    return {
        "n_samples": int(mu_all.size(0)),
        "latent_dim": int(mu_all.size(1)),
        "mu_norm_mean": float(mu_norm.mean()),
        "mu_norm_std": float(mu_norm.std()),
        "sigma_mean": float(sigma_all.mean()),
        "sigma_std": float(sigma_all.std()),
        "kl_total_mean": float(kl_per_dim.sum()),
        "kl_per_dim_max": float(kl_per_dim.max()),
        "kl_per_dim_top10": [round(v, 4) for v in torch.sort(kl_per_dim, descending=True).values[:10].tolist()],
        "dead_dims_ratio": float((kl_per_dim < DEAD_KL_THRESH).float().mean()),
        "active_units_ratio": float((mu_all.var(0) > ACTIVE_VAR_THRESH).float().mean()),
        "kl_per_dim": [round(v, 6) for v in kl_per_dim.tolist()],
    }


def run_collapse(model, device, data_dir=None) -> dict:
    mus, logvars, source = [], [], "synthetic_proxy"
    if data_dir and os.path.isdir(data_dir):
        files = [f for f in glob.glob(os.path.join(data_dir, "**", "house_*.json"), recursive=True)
                 if not f.endswith("_topology.json")]
        for fp in files:
            try:
                with open(fp, encoding="utf-8") as f:
                    data = json.load(f)
                sample = json_to_sample(data)
                if sample is None:
                    continue
                mu, logvar = encode_mu_logvar(model, sample["graph"], sample["condition"], device)
                mus.append(mu.cpu()); logvars.append(logvar.cpu())
            except Exception as exc:
                print(f"[skip] {os.path.basename(fp)}: {exc}")
        if mus:
            source = f"dataset({len(mus)})"
    if not mus:
        for sx, sy, counts, seed in SYNTH_SPECS:
            sample = sample_from_request(sx, sy, counts, seed)
            mu, logvar = encode_mu_logvar(model, sample["graph"], sample["condition"], device)
            mus.append(mu.cpu()); logvars.append(logvar.cpu())
    rep = collapse_from_mulogvar(torch.cat(mus), torch.cat(logvars))
    rep["source"] = source
    return rep


def run_semantic(model, device) -> dict:
    def enc_for(sx, sy, counts):
        s = sample_from_request(sx, sy, counts, SEMANTIC_SEED)
        mu, logvar = encode_mu_logvar(model, s["graph"], s["condition"], device)
        return mu.detach(), s["condition"]

    def cos_dist(a, b):
        return float(1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

    mu_b3, cond_b3 = enc_for(18000, 15000, {**BASE_COUNTS, "bedroom": 3})
    mu_b5, cond_b5 = enc_for(18000, 15000, {**BASE_COUNTS, "bedroom": 5})
    mu_small, _ = enc_for(12000, 9000, BASE_COUNTS)
    mu_large, _ = enc_for(28000, 22000, BASE_COUNTS)
    mu_b3c, mu_b5c = mu_b3.squeeze(0).cpu(), mu_b5.squeeze(0).cpu()
    return {
        "bedroom_3_vs_5_cos_dist": cos_dist(mu_b3c, mu_b5c),
        "bedroom_3_vs_5_l2": float((mu_b3c - mu_b5c).norm()),
        "site_small_vs_large_cos_dist": cos_dist(mu_small.squeeze(0).cpu(), mu_large.squeeze(0).cpu()),
        "_mu_b3": mu_b3, "_mu_b5": mu_b5,
        "_cond_b3": cond_b3, "_cond_b5": cond_b5,
    }


def run_decoder(model, device, mu_b3, mu_b5, cond_b3, cond_b5) -> dict:
    """以真实 encoder 的 μ 为基准 z（在训练流形上），并报 n_occ。"""
    def dec_occ(z, cond):
        cls = decode_cls(model, z, cond, device)
        return cls, int((cls > 0).sum())

    cls_mu3_c3, occ_mu3_c3 = dec_occ(mu_b3, cond_b3)
    cls_mu3_c5, occ_mu3_c5 = dec_occ(mu_b3, cond_b5)
    cls_mu5_c3, occ_mu5_c3 = dec_occ(mu_b5, cond_b3)
    cls_mu5_c5, occ_mu5_c5 = dec_occ(mu_b5, cond_b5)
    cls_mu3_c0, occ_mu3_c0 = dec_occ(mu_b3, torch.zeros_like(cond_b3))
    torch.manual_seed(0)
    _, occ_z0 = dec_occ(torch.zeros(1, LATENT_DIM, device=device), cond_b3)
    _, occ_zr = dec_occ(torch.randn(1, LATENT_DIM, device=device), cond_b3)
    frac_cond = change_fraction(cls_mu3_c3, cls_mu3_c5)
    frac_z = change_fraction(cls_mu3_c3, cls_mu5_c3)
    return {
        "occ_mu3_cond3": occ_mu3_c3, "occ_mu3_cond5": occ_mu3_c5,
        "occ_mu5_cond3": occ_mu5_c3, "occ_mu5_cond5": occ_mu5_c5,
        "occ_mu3_cond_zeroed": occ_mu3_c0, "occ_z0_cond3": occ_z0, "occ_zrandn_cond3": occ_zr,
        "voxel_change_frac_cond(bed3->5)@mu": round(frac_cond, 5),
        "voxel_change_frac_z(mu3->mu5)": round(frac_z, 5),
        "voxel_change_frac_cond_zeroed@mu": round(change_fraction(cls_mu3_c3, cls_mu3_c0), 5),
        "cond_over_z_ratio": round(frac_cond / max(frac_z, 1e-6), 4),
        "occ_by_type_mu3_cond3": occ_by_type(cls_mu3_c3),
        "occ_by_type_mu5_cond5": occ_by_type(cls_mu5_c5),
    }


def make_verdict(collapse, semantic, decoder) -> list[str]:
    v = []
    if collapse.get("dead_dims_ratio", 0) > 0.9 and collapse.get("kl_total_mean", 1e9) < 1.0:
        v.append("CONFIRMED_POSTERIOR_COLLAPSE: z≈先验，应上 KL退火/FreeBits（V4 P0 成立）")
    elif collapse.get("sigma_mean", 1.0) < 0.3 and collapse.get("kl_total_mean", 0) > 20:
        v.append("AE_LIKE: z 携带过多信息（β 太小），FreeBits 可能加剧条件被忽略")
    if semantic.get("bedroom_3_vs_5_cos_dist", 1.0) < 0.02:
        v.append("ENCODER_INSENSITIVE: 编码器 μ 对卧室数变化几乎不响应（疑似 mean-pool 稀释）")
    real_occ = max(decoder.get("occ_mu3_cond3", 0), decoder.get("occ_mu5_cond5", 0))
    if real_occ == 0:
        v.append("EMPTY_COLLAPSE: 真实 μ 下 argmax 全 empty → 主因是稀疏占用塌空，V4 损失须对抗（类别加权/Focal/提高Dice）")
    elif decoder.get("cond_over_z_ratio", 1.0) < 0.2 or decoder.get("voxel_change_frac_cond(bed3->5)@mu", 1.0) < 0.01:
        v.append("DECODER_IGNORES_CONDITION: 解码器主要靠 z → 优先 V4-b（绝对数量+FiLM+升维）而非 FreeBits")
    if not v:
        v.append("未触发明显告警阈值，请人工对照 A/B/C 数值判断")
    return v


def main():
    ap = argparse.ArgumentParser(description="V4-a 诊断：后验坍缩 / 语义敏感性 / 解码器条件敏感性")
    ap.add_argument("--weights", required=True, help="V3 权重 .pth 路径")
    ap.add_argument("--data-dir", default=None, help="可选：含 house_*.json 的目录，用于真实坍缩统计")
    ap.add_argument("--device", default=None, help="cuda / cpu，默认自动")
    ap.add_argument("--out", default=None, help="报告输出路径，默认 ./diagnostics_v4a.json")
    args = ap.parse_args()

    if not Path(args.weights).exists():
        raise FileNotFoundError(f"未找到权重: {args.weights}")

    model, device = load_model(args.weights, args.device)
    model.eval()
    print(f"[诊断] 权重={args.weights} | 设备={device} | 潜维={LATENT_DIM} | 通道={NUM_CHANNELS}")

    print("\n========== A. 后验坍缩 ==========")
    collapse = run_collapse(model, device, args.data_dir)
    print(f"来源={collapse['source']} | 样本={collapse['n_samples']}")
    print(f"mu_norm={collapse['mu_norm_mean']:.3f}±{collapse['mu_norm_std']:.3f} | "
          f"sigma_mean={collapse['sigma_mean']:.4f} | KL_total={collapse['kl_total_mean']:.3f}")
    print(f"死维占比={collapse['dead_dims_ratio']*100:.1f}% | 活跃单元={collapse['active_units_ratio']*100:.1f}%")
    print(f"KL_per_dim Top10={collapse['kl_per_dim_top10']}")

    print("\n========== B. 编码器语义敏感性 ==========")
    semantic = run_semantic(model, device)
    mu_b3 = semantic.pop("_mu_b3"); mu_b5 = semantic.pop("_mu_b5")
    cond_b3 = semantic.pop("_cond_b3"); cond_b5 = semantic.pop("_cond_b5")
    print(f"卧室 3↔5  μ 余弦距离={semantic['bedroom_3_vs_5_cos_dist']:.4f} (L2={semantic['bedroom_3_vs_5_l2']:.3f})")
    print(f"场地 小↔大 μ 余弦距离={semantic['site_small_vs_large_cos_dist']:.4f}")

    print("\n========== C. 解码器条件敏感性（真实 μ 基准）==========")
    decoder = run_decoder(model, device, mu_b3, mu_b5, cond_b3, cond_b5)
    print(f"n_occ:  μ3/c3={decoder['occ_mu3_cond3']}  μ3/c5={decoder['occ_mu3_cond5']}  "
          f"μ5/c3={decoder['occ_mu5_cond3']}  μ5/c5={decoder['occ_mu5_cond5']}  "
          f"| z0={decoder['occ_z0_cond3']} z~N={decoder['occ_zrandn_cond3']}")
    print(f"改 cond(卧室3→5)@μ 体素变化率={decoder['voxel_change_frac_cond(bed3->5)@mu']*100:.3f}%")
    print(f"改 z(μ3→μ5)        体素变化率={decoder['voxel_change_frac_z(mu3->mu5)']*100:.3f}%")
    print(f"cond/z 影响比={decoder['cond_over_z_ratio']:.3f}（越小越无视条件）")
    if max(decoder['occ_mu3_cond3'], decoder['occ_mu5_cond5']) == 0:
        print("⚠ 真实 μ 下 n_occ=0：argmax 全 empty（稀疏塌空）")

    verdict = make_verdict(collapse, semantic, decoder)
    print("\n========== 判读 ==========")
    for x in verdict:
        print("  •", x)

    report = {
        "phase": "V4-a_diagnostics",
        "weights": str(args.weights),
        "thresholds": {"dead_kl": DEAD_KL_THRESH, "active_var": ACTIVE_VAR_THRESH},
        "A_collapse": collapse,
        "B_semantic": semantic,
        "C_decoder": decoder,
        "verdict": verdict,
    }
    out = args.out or os.path.join(os.getcwd(), "diagnostics_v4a.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n诊断报告已保存: {out}")


if __name__ == "__main__":
    main()
