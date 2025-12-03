import torch, os
import numpy as np
from PIL import Image
import matplotlib.cm as cm
from .attention import flash_attention


# ========== 模块 1：计算 entropy / weight ==========
def compute_entropy_and_weight(q, k, block_size, eps, mode):
    """计算每个 token 的 entropy 与权重总和"""
    B, L, H, D = q.shape
    device = q.device
    dtype = torch.float32
    k_all = k.to(dtype)

    need_entropy = mode in ("high", "low", "weight_entropy_ratio_high", "weight_entropy_ratio_low")
    need_weight = mode in ("weight_high", "weight_low", "weight_entropy_ratio_high", "weight_entropy_ratio_low")

    entropy = torch.zeros((B, H, L), dtype=dtype, device=device) if need_entropy else None
    weight_sum = torch.zeros((B, H, L), dtype=dtype, device=device) if need_weight else None

    if need_entropy or need_weight:
        for qi in range(0, L, block_size):
            qi_end = min(qi + block_size, L)
            q_chunk = q[:, qi:qi_end, :, :].to(dtype)
            scores = torch.einsum("blhd,bmhd->bhlm", q_chunk, k_all) / (D ** 0.5)
            attn = torch.softmax(scores, dim=-1).clamp(min=eps)

            if need_entropy:
                entropy[:, :, qi:qi_end] = (-(attn * torch.log(attn))).sum(dim=-1)
            if need_weight:
                weight_sum += attn.sum(dim=2)

            del scores, attn, q_chunk
            torch.cuda.empty_cache()

    if "weight_entropy_ratio" in mode:
        ratio_metric = weight_sum / (entropy + eps)
    else:
        ratio_metric = None

    return entropy, weight_sum, ratio_metric


# ========== 模块 2：选择扰动的 token ==========
def select_tokens(B, H, L, entropy, weight_sum, ratio_metric, mode, select_fraction, device):
    """根据模式选出要扰动的 token"""
    num_select = max(1, int(L * select_fraction))
    mask = torch.zeros((B, H, L), dtype=torch.bool, device=device)

    for b in range(B):
        for h in range(H):
            if mode == "high":
                idx = torch.topk(entropy[b, h], num_select, largest=True).indices
            elif mode == "low":
                idx = torch.topk(entropy[b, h], num_select, largest=False).indices
            elif mode == "weight_high":
                idx = torch.topk(weight_sum[b, h], num_select, largest=True).indices
            elif mode == "weight_low":
                idx = torch.topk(weight_sum[b, h], num_select, largest=False).indices
            elif mode == "weight_entropy_ratio_high":
                idx = torch.topk(ratio_metric[b, h], num_select, largest=True).indices
            elif mode == "weight_entropy_ratio_low":
                idx = torch.topk(ratio_metric[b, h], num_select, largest=False).indices
            elif mode == "random":
                idx = torch.randperm(L, device=device)[:num_select]
            else:
                raise ValueError(f"未知 mode: {mode}")
            mask[b, h, idx] = True

    return mask


# ========== 模块 3：扰动计算 ==========
def apply_attention_perturbation(q, k_all, v_all, mask, block_size, perturb_scale):
    """执行扰动并计算扰动差异"""
    B, L, H, D = q.shape
    compute_dtype = torch.float32
    device = q.device

    out = torch.zeros((B, L, H, D), dtype=compute_dtype, device=device)
    diff_map = torch.zeros((B, H, L), dtype=compute_dtype, device=device)

    for qi in range(0, L, block_size):
        qi_end = min(qi + block_size, L)
        q_chunk = q[:, qi:qi_end, :, :].to(compute_dtype)

        scores_b = torch.einsum("blhd,bmhd->bhlm", q_chunk, k_all) / (D ** 0.5)
        attn_b = torch.softmax(scores_b, dim=-1)
        scores_a = scores_b.clone()

        mask_block = mask[:, :, qi:qi_end].unsqueeze(-1)
        if mask_block.any():
            scores_a += torch.randn_like(scores_a) * perturb_scale * mask_block

        attn_a = torch.softmax(scores_a, dim=-1)
        diff_map[:, :, qi:qi_end] = ((attn_a - attn_b) ** 2).sum(dim=-1).sqrt()
        out[:, qi:qi_end, :, :] = torch.einsum("bhlm,bmhd->blhd", attn_a, v_all)

        del q_chunk, scores_b, scores_a, attn_b, attn_a
        torch.cuda.empty_cache()

    return out, diff_map


# ========== 模块 4：保存 delta 和 entropy ==========
def save_metrics(diff_map, entropy, save_root, exp_idx, mode, timestep, block_idx, batch, cond_uncond):
    """保存每个 head 的 delta 与 entropy"""
    text_root = os.path.join(save_root, "textdata", f"exp{exp_idx}")
    os.makedirs(text_root, exist_ok=True)

    delta_path = os.path.join(text_root, f"attn_dist_summary-mode_{mode}-step{timestep}-{cond_uncond}.txt")
    entropy_path = os.path.join(text_root, f"entropy_summary-mode_{mode}-step{timestep}-{cond_uncond}.txt")

    B, H, _ = diff_map.shape

    with open(delta_path, "a") as f_d, open(entropy_path, "a") as f_e:
        per_head_deltas = []
        per_head_entropy = []
        for h in range(H):
            delta_sum = diff_map[batch, h].sum().item()
            per_head_deltas.append(delta_sum)
            f_d.write(f"timestep_{timestep}-block_{block_idx}-head_{h}-delta:{delta_sum:.6f}\n")

            if entropy is not None:
                entropy_sum = entropy[batch, h].sum().item()
                per_head_entropy.append(entropy_sum)
                f_e.write(f"timestep_{timestep}-block_{block_idx}-head_{h}-entropy:{entropy_sum:.6f}\n")

        f_d.write(f"timestep_{timestep}-block_{block_idx}-mean_delta:{np.mean(per_head_deltas):.6f}\n")
        if entropy is not None:
            f_e.write(f"timestep_{timestep}-block_{block_idx}-mean_entropy:{np.mean(per_head_entropy):.6f}\n")

# ========== 模块 5：主函数 ==========
def perturb_flash_attention_scores(
    q, k, v,
    k_lens=None, window_size=None, grid_sizes=None,
    mode="high", perturb_scale=1.5, block_size=512,
    select_fraction=0.3, eps=1e-8,
    cond_uncond=None,
    save_root="Exps", block_idx=0,
    timestep=0, batch=0, patch_size=(1, 2, 2),
    frame_stride=1, 
    save_mode="none",
    apply_perturb=True,
    save_attn_diff=False,
    save_delta_sum=False,
    exp_idx=17, 
):
    """主入口：保持逻辑完全一致，仅结构化整理"""
    if not apply_perturb:
        result = flash_attention(q=q, k=k, v=v, k_lens=k_lens, window_size=window_size)

    if mode == "low":
        perturb_scale = 1.0

    B, L, H, D = q.shape
    device = q.device
    dtype_orig = q.dtype
    compute_dtype = torch.float32
    exp_root = os.path.join(save_root, f"exp_{exp_idx}")

    # ===== 1. 计算 entropy / weight / ratio =====
    entropy, weight_sum, ratio_metric = compute_entropy_and_weight(q, k, block_size, eps, mode)

    # ===== 2. 选择扰动 token =====
    mask = select_tokens(B, H, L, entropy, weight_sum, ratio_metric, mode, select_fraction, device)

    # ===== 3. 执行扰动 =====
    k_all = k.to(compute_dtype)
    v_all = v.to(compute_dtype)
    out, diff_map = apply_attention_perturbation(q, k_all, v_all, mask, block_size, perturb_scale)

    # ===== 4. 保存结果 =====
    if save_delta_sum:
        save_metrics(diff_map, entropy, save_root, exp_idx, mode, timestep, block_idx, batch, cond_uncond)

    # ===== 5. 可选可视化（逻辑不变） =====
    if save_attn_diff:
        os.makedirs(exp_root, exist_ok=True)
        Fp, Hp, Wp = grid_sizes[batch].tolist() if grid_sizes is not None else (1, 1, L)
        tokens_per_frame = Hp * Wp

        entropy_np = entropy[batch].float().cpu().numpy() if entropy is not None else None
        diff_np = diff_map[batch].float().cpu().numpy()
        ratio_np = ratio_metric[batch].float().cpu().numpy() if ratio_metric is not None else None

        cmap_ent = cm.get_cmap("viridis")
        cmap_diff = cm.get_cmap("plasma")
        cmap_ratio = cm.get_cmap("inferno")

        for h in range(H):
            head_dir = os.path.join(exp_root, f"mode{mode}", f"head{h}", f"timestep{timestep}", f"block{block_idx}")
            os.makedirs(head_dir, exist_ok=True)

            for f in range(0, Fp, frame_stride):
                s = f * tokens_per_frame
                e = min((f + 1) * tokens_per_frame, L)
                if e - s != Hp * Wp:
                    continue

                if entropy_np is not None:
                    ent_frame = entropy_np[h, s:e].reshape(Hp, Wp)
                    ent_norm = (ent_frame - ent_frame.min()) / (ent_frame.ptp() + 1e-8)
                    ent_rgb = (cmap_ent(ent_norm)[:, :, :3] * 255).astype(np.uint8)

                diff_frame = diff_np[h, s:e].reshape(Hp, Wp)
                diff_norm = (diff_frame - diff_frame.min()) / (diff_frame.ptp() + 1e-8)
                diff_rgb = (cmap_diff(diff_norm)[:, :, :3] * 255).astype(np.uint8)

                if ratio_np is not None:
                    ratio_frame = ratio_np[h, s:e].reshape(Hp, Wp)
                    ratio_norm = (ratio_frame - ratio_frame.min()) / (ratio_frame.ptp() + 1e-8)
                    ratio_rgb = (cmap_ratio(ratio_norm)[:, :, :3] * 255).astype(np.uint8)
                else:
                    ratio_rgb = None

                if save_mode in ("image", "both"):
                    if entropy_np is not None:
                        Image.fromarray(ent_rgb).save(os.path.join(head_dir, f"entropy_f{f}.jpg"), quality=95)
                    Image.fromarray(diff_rgb).save(os.path.join(head_dir, f"diff_f{f}.jpg"), quality=95)
                    if ratio_rgb is not None:
                        Image.fromarray(ratio_rgb).save(os.path.join(head_dir, f"ratio_f{f}.jpg"), quality=95)

    # ===== 6. 输出结果 =====
    return out.to(dtype_orig) if apply_perturb else result
