import numpy as np
from PIL import Image
import Dataloaders.surreal_human_seg_dl as sh_seg_dl


def save_model_pred_ims(original_im: np.ndarray, pred_mask: np.ndarray,
                        gt_mask: np.ndarray, save_fp: str) -> np.ndarray:
    seg_pred_color_im = sh_seg_dl.mask_to_color_im(pred_mask)
    seg_gt_color_im = sh_seg_dl.mask_to_color_im(gt_mask)
    if original_im.shape[-1] != 3:
        original_im_ = np.transpose(original_im, (1, 2, 0))
    else:
        original_im_ = original_im.copy()
    overall_im = np.hstack((original_im_, seg_gt_color_im, seg_pred_color_im))
    im = Image.fromarray(overall_im)
    # im.show()
    im.save(save_fp)