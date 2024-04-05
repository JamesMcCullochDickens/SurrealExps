import data
import vid_utils as vid_utils
import os
import numpy as np
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from mat_proc import get_seg_mask_at_frame, get_person_bb
from typing import List, Tuple, Optional
import torch.nn.functional as F
import random as r
from PIL import Image
import math

dataset_outer_fp = data.dataset_outer_fp
cwd = os.environ["cwd"]

sorted_parts = ['hips', 'leftUpLeg', 'rightUpLeg', 'spine', 'leftLeg', 'rightLeg',
                'spine1', 'leftFoot', 'rightFoot', 'spine2', 'leftToeBase', 'rightToeBase',
                'neck', 'leftShoulder', 'rightShoulder', 'head', 'leftArm', 'rightArm',
                'leftForeArm', 'rightForeArm', 'leftHand', 'rightHand', 'leftHandIndex1', 'rightHandIndex1']
part_d_str_to_num = {k: v for v, k in enumerate(sorted_parts)}
part_d_num_to_str = {v: k for (k, v) in part_d_str_to_num.items()}


def get_all_vid_mp4s(split_str: str) -> list[str]:
    outer_folder_fp = os.path.join(dataset_outer_fp, split_str)
    fps = list(glob.iglob(outer_folder_fp + "/**/*.mp4", recursive=True))
    fps.sort()
    return fps


def get_part_color_map() -> dict:
    global part_d_num_to_str
    keys = list(range(25))
    color_d = {}
    biases, gain, pow = [80, 160, 240], 2, 3
    for key in keys:
        r = int(math.pow((key+biases[0])*gain, pow)) % 255.0
        g = int(math.pow((key+biases[1])*gain, pow)) % 255.0
        b = int(math.pow((key+biases[2])*gain, pow)) % 255.0
        color_d[key] = np.array([r, g, b], dtype=np.uint8)
    return color_d


def visualize(im: np.ndarray, mask: np.ndarray) -> None:
    rgb_im = Image.fromarray(im)
    rgb_im.show()
    color_mask = np.zeros_like(rgb_im, dtype=np.uint8)
    unique_vals = np.unique(mask)
    color_d = get_part_color_map()
    for val in unique_vals:
        if val == 0:
            continue
        spatial_mask = mask == val
        color_mask[spatial_mask, :] = color_d[val]
    color_mask_im = Image.fromarray(color_mask)
    color_mask_im.show()
    rgb_mask = (0.65*im + 0.35*color_mask).astype(np.uint8)
    rgb_mask_im = Image.fromarray(rgb_mask)
    rgb_mask_im.show()



def get_vid_frame_pairs(split: str, ratio: float = (1.0 / 15.0)) -> list[str]:
    vid_fps = get_all_vid_mp4s(split)
    vid_frame_pairs = []
    for i, vid_fp in enumerate(vid_fps):
        num_frames = vid_utils.get_vid_len_decord(vid_fp)
        if num_frames < 10:  # take all frames
            frame_nums = np.arange(0, num_frames, 1)
        else:
            step_size = num_frames // max((int(ratio * num_frames)), 1)
            frame_nums = np.arange(0, num_frames, step_size)
        vid_fp_inner = vid_fp.replace(dataset_outer_fp, "")
        vid_frame_pairs.extend([vid_fp_inner + "," + str(int(frame_nums[i])) for i in range(frame_nums.shape[0])])
    return vid_frame_pairs


def write_vid_frame_pairs() -> None:
    outer_save_fp = os.path.join(cwd, "Dataloaders/vid_fp_frame_pairs")
    if not os.path.exists(outer_save_fp):
        os.mkdir(outer_save_fp)
    for split in ["train", "val", "test"]:
        vf_pairs = get_vid_frame_pairs(split)
        save_txt_fp = os.path.join(outer_save_fp, split + ".txt")
        with open(save_txt_fp, "w") as f:
            f.writelines([vf_pair + "\n" for vf_pair in vf_pairs])


def read_vid_frame_pairs(split: str) -> List[Tuple[str, int]]:
    pairs_fp = os.path.join(cwd, "Dataloaders/vid_fp_frame_pairs", split + ".txt")
    with open(pairs_fp, "r") as f:
        pairs = [(line.strip().split(",")[0], int(line.strip().split(",")[1])) for line in f]
    return pairs


def get_seg_mask(vid_fp: str, frame_num: int) -> np.ndarray:
    seg_mask_fp = vid_fp.replace(".mp4", "_segm.mat")
    seg_mask = get_seg_mask_at_frame(seg_mask_fp, frame_num)
    return seg_mask


def rgb_channel_normalize(f: torch.Tensor) -> torch.Tensor:
    imagenet_means = [0.485, 0.456, 0.406]
    imagenet_stds = [0.229, 0.224, 0.225]
    for i in range(3):
        f[i, :, :] = (f[i, :, :]-imagenet_means[i])/(imagenet_stds[i])
    return f


def interp_im_and_mask(im: torch.Tensor, mask: torch.Tensor,
                       inter_shape: Tuple[int, int] = (300, 200)):
    im = torch.unsqueeze(im, dim=0)
    im = F.interpolate(im, size=inter_shape, mode="bilinear")[0]
    mask = torch.unsqueeze(torch.unsqueeze(mask, 0), 0).float()
    mask = F.interpolate(mask, size=inter_shape, mode="nearest")[0][0].long()
    return im, mask


class SurrealHumanSegDataset(Dataset):
    def __init__(self, split: str,
                 with_masking: bool = False,
                 with_cropping: bool = False,
                 cropping_type: str = "bb",
                 interp_size: Optional[Tuple[int, int]] = None,
                 debug: bool = False
                 ):
        super().__init__()
        assert split in ["train", "val", "test"], "invalid split"
        if cropping_type is not None:
            assert cropping_type in ["bb", "random"]
        self.is_train = split == "test"
        self.with_masking = with_masking
        self.with_cropping = with_cropping
        self.cropping_type = cropping_type
        self.interp_size = interp_size
        self.vid_frame_fps = read_vid_frame_pairs(split)
        if debug:
            r.shuffle(self.vid_frame_fps)
            self.vid_frame_fps = self.vid_frame_fps[:int(0.01 * len(self.vid_frame_fps))]

    def __len__(self):
        return len(self.vid_frame_fps)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        vid_fp, frame_num = self.vid_frame_fps[index]
        vid_fp = os.path.join(dataset_outer_fp, vid_fp[1:])
        vid_frame = vid_utils.get_frame_from_rgb_vid(vid_fp, frame_num)[0]  # shape H, W, 3
        seg_mask = get_seg_mask(vid_fp, frame_num)

        # sanity check
        # visualize(vid_frame, seg_mask)

        # preprocess rgb
        vid_frame = torch.tensor(vid_frame, dtype=torch.float).permute((-1, 0, 1)) / 255.0
        vid_frame = rgb_channel_normalize(vid_frame)

        if self.with_masking:
            zero_mask = torch.unsqueeze(seg_mask == 0, dim=0).repeat((3, 1, 1))
            vid_frame = zero_mask * vid_frame
        if self.with_cropping:
            if self.cropping_type == "bb":
                bb = get_person_bb(seg_mask)
                vid_frame = vid_frame[:, bb[1]:bb[3] + 1, bb[0]:bb[2] + 1]
                seg_mask = seg_mask[bb[1]:bb[3] + 1, bb[0]:bb[2] + 1]
            else:
                h, w = vid_frame.shape[1:]
                x1 = r.randint(0, w // 2)
                x2 = r.randint(w // 2, w - 1)
                y1 = r.randint(0, h // 2)
                y2 = r.randint(h // 2, h - 1)
                bb = [x1, y1, x2, y2]
                vid_frame = vid_frame[:, bb[1]:bb[3] + 1, bb[0]:bb[2] + 1]
                seg_mask = seg_mask[bb[1]:bb[3] + 1, bb[0]:bb[2] + 1]
        seg_mask = torch.tensor(seg_mask, dtype=torch.long)
        if self.interp_size:
            vid_frame, seg_mask = interp_im_and_mask(vid_frame, seg_mask, self.interp_size)
        return vid_frame, seg_mask


def get_surreal_human_seg_dl(dl_kwargs, dataset_kwargs):
    dataset = SurrealHumanSegDataset(**dataset_kwargs)
    dl = DataLoader(dataset=dataset, **dl_kwargs)
    return dl


"""
if __name__ == "__main__":
    dl_kwargs = {"num_workers": 0, "batch_size": 1,
                 "shuffle": False, "drop_last": True,
                 "pin_memory": False}
    dataset_kwargs = {"split": "train"}
    dl = get_surreal_human_seg_dl(dl_kwargs, dataset_kwargs)
    for data in dl:
        debug = "debug"
"""