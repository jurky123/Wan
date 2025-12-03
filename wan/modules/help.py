import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import lpips
exp_idx=17


def save_attention_heatmap(q, k, v, grid_sizes, save_dir="attn_maps",
                           block_idx=0, timestep=0, batch=0, heads=(0,), max_frames=None,
                           patch_size=(1,8,8)):
    """
    计算注意力权重并保存热力图 (unpatchify 到帧空间)
    q, k, v: [B, L, H, D]
    grid_sizes: [B,3], (F,H_patches,W_patches)
    patch_size: patch 大小 (F_patch, H_patch, W_patch)
    """

    if timestep >= 35 and block_idx == 39:
        os.makedirs(save_dir, exist_ok=True)

        B, L, H, D = q.shape
        Fp, Hp, Wp = grid_sizes[batch].tolist()  # patch grid
        tokens_per_frame = Hp * Wp
        if max_frames is None:
            max_frames = Fp

        # 计算注意力分数 [B,H,L,L]
        scores = torch.einsum("blhd,bmhd->bhlm", q, k) / (D ** 0.5)
        attn = torch.softmax(scores, dim=-1).detach().cpu().numpy()
        
        # 每个 patch 还原到空间分辨率
        H_img, W_img = Hp * patch_size[1], Wp * patch_size[2]

        for head in heads:
            for f in range(max_frames):
                s, e = f * tokens_per_frame, (f + 1) * tokens_per_frame
                # 取该帧 self-attn [tokens,tokens]
                attn_map = attn[batch, head, s:e, s:e]

                # 将每个 query token 的注意力分布 reshape 回 patch grid
                # 这里我们简单地对所有 query 取平均 => [Hp,Wp]
                frame_attn = attn_map.mean(axis=0).reshape(Hp, Wp)

                # unpatchify 到像素级 (H_img,W_img)
                frame_attn_up = np.kron(frame_attn, np.ones((patch_size[1], patch_size[2])))

                plt.figure(figsize=(6,6))
                plt.imshow(frame_attn_up, cmap="hot", interpolation="nearest")
                plt.colorbar()
                plt.title(f"Block {block_idx}, Head {head}, Timestep {timestep}, Frame {f}")
                plt.xlabel("Width")
                plt.ylabel("Height")

                fname = os.path.join(
                    save_dir,
                    f"attn_block{block_idx}_head{head}_t{timestep}_frame{f}.png"
                )
                plt.savefig(fname, dpi=300)
                plt.close()
    else:
        print(f"Skipping saving attention map at timestep {timestep} < 35")
def compute_lpips(img1, img2):
    loss_fn_vgg = lpips.LPIPS(net='vgg', version='0.1')  # 也可以使用 net='alex'

    # 将模型移到 GPU (如果可用)
    if torch.cuda.is_available():
        loss_fn_vgg.cuda()
    """
    计算两张图像之间的 LPIPS 感知相似度
    img1, img2: [B, C, H, W], 像素值范围 [-1, 1]
    lpips_model: 预加载的 LPIPS 模型
    返回: [B] 的感知相似度分数，值越小表示越相似
    """
    d = loss_fn_vgg(img1.float(), img2.float())
    
    # d 的形状通常是 (N, 1, 1, 1)，取其平均值并转换为 Python float
    return d.mean().item()