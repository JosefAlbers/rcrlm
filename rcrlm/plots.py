import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

def plt_svd(A, fraction=None, show=True):
    if A.ndim != 2:
        raise ValueError("Input A must be a 2D matrix.")
    
    A = A.astype(mx.float32) if isinstance(A, mx.array) else A
    A = np.array(A).astype(np.float32)

    M, N = A.shape
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    K_full = len(S)
    
    if fraction is None:
        k = K_full
    else:
        k = min(K_full, int(K_full*fraction))

    Uk = U[:, :k]
    Sk = S[:k]
    Sigmak = np.diag(Sk)
    Vtk = Vt[:k, :]
    Ak = Uk @ Sigmak @ Vtk
    
    Sigma = np.zeros((K_full, K_full))
    np.fill_diagonal(Sigma, S)

    def create_composite(matrix, mask, cmap_kept='viridis', cmap_discarded='gray'):
        norm = mcolors.Normalize(vmin=matrix.min(), vmax=matrix.max())
        img_kept = plt.get_cmap(cmap_kept)(norm(matrix))
        img_discarded = plt.get_cmap(cmap_discarded)(norm(matrix))
        mask_rgba = np.stack([mask]*4, axis=-1)
        return img_kept * mask_rgba + img_discarded * (1 - mask_rgba)

    mask_U = np.zeros_like(U, dtype=bool)
    mask_U[:, :k] = True
    img_U = create_composite(U, mask_U, cmap_kept='viridis')

    mask_S = np.zeros_like(Sigma, dtype=bool)
    mask_S[:k, :k] = True
    img_S = create_composite(Sigma, mask_S, cmap_kept='viridis', cmap_discarded='gist_gray')

    mask_Vt = np.zeros_like(Vt, dtype=bool)
    mask_Vt[:k, :] = True
    img_Vt = create_composite(Vt, mask_Vt, cmap_kept='viridis')
    
    max_h = M 
    
    def pad_and_center(img, target_h):
        h, w, c = img.shape
        if h >= target_h:
            return img, 0, h
            
        total_pad = target_h - h
        pad_top = total_pad // 2
        pad_bot = total_pad - pad_top
        
        top_padding = np.zeros((pad_top, w, c))
        bot_padding = np.zeros((pad_bot, w, c))
        
        padded = np.vstack([top_padding, img, bot_padding])
        return padded, pad_top, h

    img_U_padded, U_top, U_h = pad_and_center(img_U, max_h)
    img_S_padded, S_top, S_h = pad_and_center(img_S, max_h)
    img_Vt_padded, Vt_top, Vt_h = pad_and_center(img_Vt, max_h)
    
    fig = plt.figure(figsize=(20, 8))
    
    spacer_w = max(N, K_full) / 10 
    w_ratios = [N, spacer_w, K_full, spacer_w, K_full, spacer_w, N, spacer_w, N]
    gs = gridspec.GridSpec(1, 9, width_ratios=w_ratios)
    
    def clean_axis(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


    ax_A = plt.subplot(gs[0])
    ax_A.imshow(A, cmap='gray', aspect='equal')
    ax_A.set_title(f'A (Original)\n{M}x{N}')
    clean_axis(ax_A)

    ax_eq1 = plt.subplot(gs[1])
    ax_eq1.text(0.5, 0.5, '=', fontsize=30, ha='center', va='center')
    clean_axis(ax_eq1)

    ax_U = plt.subplot(gs[2])
    ax_U.imshow(img_U_padded, aspect='equal')
    ax_U.vlines(x=k-0.5, ymin=U_top-0.5, ymax=U_top+U_h-0.5, color='red', linewidth=2)
    ax_U.set_title(f'U\n{M}x{K_full}')
    clean_axis(ax_U)

    ax_dot1 = plt.subplot(gs[3])
    ax_dot1.text(0.5, 0.5, r'$\cdot$', fontsize=30, ha='center', va='center')
    clean_axis(ax_dot1)

    ax_S = plt.subplot(gs[4])
    ax_S.imshow(img_S_padded, aspect='equal')
    ax_S.vlines(x=k-0.5, ymin=S_top-0.5, ymax=S_top+S_h-0.5, color='red', linewidth=2)
    ax_S.hlines(y=S_top+k-0.5, xmin=-0.5, xmax=K_full-0.5, color='red', linewidth=2)
    ax_S.set_title(rf'$\Sigma$'+f'\n{K_full}x{K_full}')
    clean_axis(ax_S)

    ax_dot2 = plt.subplot(gs[5])
    ax_dot2.text(0.5, 0.5, r'$\cdot$', fontsize=30, ha='center', va='center')
    clean_axis(ax_dot2)

    ax_Vt = plt.subplot(gs[6])
    ax_Vt.imshow(img_Vt_padded, aspect='equal')
    ax_Vt.hlines(y=Vt_top+k-0.5, xmin=-0.5, xmax=N-0.5, color='red', linewidth=2)
    ax_Vt.set_title(f'$V^T$\n{K_full}x{N}')
    clean_axis(ax_Vt)

    ax_eq2 = plt.subplot(gs[7])
    ax_eq2.text(0.5, 0.5, r'$\approx$', fontsize=30, ha='center', va='center')
    clean_axis(ax_eq2)

    ax_Ak = plt.subplot(gs[8])
    ax_Ak.imshow(Ak, cmap='gray', aspect='equal')
    
    ax_Ak.set_title(f"$A_k$ (k={k})\n{M}x{N}")
    clean_axis(ax_Ak)

    plt.suptitle(f'Truncated SVD Visualization', fontsize=16)
    plt.tight_layout()
    plt.savefig('out.png')
    if show:
        plt.show()
    return mx.array(Uk), mx.array(Sigmak), mx.array(Vtk)


def get_2d():
    from skimage import data
    from skimage.color import rgb2gray
    from skimage.transform import resize
    image_bw = rgb2gray(data.astronaut())
    image_bw = resize(image_bw, (512, 512))
    return image_bw

if __name__ == '__main__':
    A = get_2d()
    plt_svd(A, fraction=0.125)

