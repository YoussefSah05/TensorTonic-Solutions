import numpy as np

def patch_embed(image: np.ndarray, patch_size: int,
                W_proj: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Returns float64 patch embeddings with shape (B, N, D).
    """
    B, H, W, C = image.shape
    N = (H//patch_size) * (W//patch_size)
    image = image[:, :H//patch_size * patch_size, :W//patch_size * patch_size, :]
    image = image.reshape(B, H//patch_size, patch_size, W//patch_size, patch_size, C).transpose(0, 1, 3, 2, 4, 5)
    image = image.reshape(B, N, -1) # shape now is (B, N, *)
    return (image @ W_proj + bias).astype(np.float64)