import scipy.io
import numpy as np


def load_mat(mat_fp: str) -> dict:
    mat = scipy.io.loadmat(mat_fp)
    return mat


def get_person_bb(arr:np.ndarray) -> np.ndarray:
    bb = np.empty((4,), dtype=int)
    non_zero_indices = np.nonzero(arr)
    bb[0] = np.min(non_zero_indices[1])
    bb[2] = np.max(non_zero_indices[1])
    bb[1] = np.min(non_zero_indices[0])
    bb[3] = np.max(non_zero_indices[0])
    return bb


def get_im_num(seg_entry):
    return int(seg_entry.split("_")[-1])


def get_seg_masks(mat: dict) -> np.ndarray:
    keys = list(mat.keys())
    for k in ["__header__", "__version__", "__globals__"]:
        keys.remove(k)
    keys.sort(key=lambda x: get_im_num(x))
    n_masks = len(keys)
    h, w = mat[keys[0]].shape
    seg_masks = np.empty((n_masks, h , w), dtype=int)
    for im_num in range(n_masks):
        seg_masks[im_num] = mat[keys[im_num]]
    return seg_masks


def get_seg_mask_at_frame(mat_fp: str, frame_num: int) -> np.ndarray:
    m = load_mat(mat_fp)
    key = "segm_" + str(frame_num+1)
    return m[key]


def get_depth_ims(mat: dict) -> np.ndarray:
    keys = list(mat.keys())
    for k in ["__header__", "__version__", "__globals__"]:
        keys.remove(k)
    keys.sort(key=lambda x: get_im_num(x))
    n_masks = len(keys)
    h, w = mat[keys[0]].shape
    depth_ims = np.empty((n_masks, h , w), dtype=float)
    for im_num in range(n_masks):
        depth_ims[im_num] = mat[keys[im_num]]
    return depth_ims


def get_person_bbs(seg_masks:np.ndarray) -> np.ndarray:
    nf = seg_masks.shape[0]
    person_bbs = np.empty((nf, 4), dtype=int)
    for fn in range(nf):
        person_bbs[fn] = get_person_bb(seg_masks[fn])
    return person_bbs
