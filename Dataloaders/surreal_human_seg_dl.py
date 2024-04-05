import data
import vid_utils as vid_utils
import os
import numpy as np
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from mat_proc import get_seg_mask_at_frame, get_person_bb
from typing import List, Tuple
import torch.nn.functional as F
import random as r

dataset_outer_fp = data.dataset_outer_fp
cwd = os.environ["cwd"]


def get_all_vid_mp4s(split_str: str) -> list[str]:
    outer_folder_fp = os.path.join(dataset_outer_fp, split_str)
    fps = list(glob.iglob(outer_folder_fp + "/**/*.mp4", recursive=True))
    fps.sort()
    return fps


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
            f.writelines([vf_pair+"\n" for vf_pair in vf_pairs])


def read_vid_frame_pairs(split: str) -> List[Tuple[str, int]]:
    pairs_fp = os.path.join(cwd, "Dataloaders/vid_fp_frame_pairs", split + ".txt")
    with open(pairs_fp, "r") as f:
        pairs = [(line.strip().split(",")[0], int(line.strip().split(",")[1])) for line in f]
    return pairs


def get_seg_mask(vid_fp:str, frame_num:int) -> np.ndarray:
    seg_mask_fp = vid_fp.replace(".mp4", "_segm.mat")
    seg_mask = get_seg_mask_at_frame(seg_mask_fp, frame_num)
    return seg_mask


def rgb_channel_normalize(frame: torch.Tensor)-> torch.Tensor:
    f = torch.unsqueeze(frame, dim=0)
    imagenet_means = torch.tensor([0.485, 0.456, 0.406], device=frame.device)
    imagenet_stds = torch.tensor([0.229, 0.224, 0.225], device=frame.device)
    f = (f - imagenet_means.view(1, -1, 1, 1)) / imagenet_stds.view(1, -1, 1, 1)
    return f[0]


def interp_im_and_mask(im: torch.Tensor, mask: torch.Tensor,
                       inter_shape: Tuple[int, int] = (300, 200)):
    im = torch.unsqueeze(im, dim=1)
    im = F.interpolate(im, size=inter_shape, mode="bilinear")[0]
    mask = torch.unsqueeze(torch.unsqueeze(mask, 1), 1).float()
    mask = F.interpolate(mask, size=inter_shape, mode="nearest")[0][0].int()
    return im, mask


class SurrealHumanSegDataset(Dataset):
    def __init__(self, split: str,
                 with_masking: bool = False,
                 with_cropping: bool = True,
                 cropping_type: str = "bb",
                 interp_size: Tuple[int, int] = (300, 200)
                 ):
        super().__init__()
        assert split in ["train", "val", "test"], "invalid split"
        assert cropping_type in ["bb", "random"]
        self.is_train = split == "test"
        self.with_masking = with_masking
        self.with_cropping = with_cropping
        self.cropping_type = cropping_type
        self.interp_size = interp_size
        self.vid_frame_fps = read_vid_frame_pairs(split)

    def __len__(self):
        return len(self.vid_frame_fps)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        vid_fp, frame_num = self.vid_frame_fps[index]
        vid_fp = os.path.join(dataset_outer_fp, vid_fp[1:])
        vid_frame = vid_utils.get_frame_from_rgb_vid(vid_fp, frame_num)[0] # shape H, W, 3
        vid_frame = torch.tensor(vid_frame, dtype=torch.float).permute((-1, 0, 1))/255.0
        vid_frame = rgb_channel_normalize(vid_frame)
        seg_mask = get_seg_mask(vid_fp, frame_num)
        if self.with_masking:
            zero_mask = torch.unsqueeze(seg_mask == 0, dim=0).repeat((3, 1, 1))
            vid_frame = zero_mask * vid_frame
        if self.with_cropping:
            if self.cropping_type == "bb":
                bb = get_person_bb(seg_mask)
                vid_frame = vid_frame[:, bb[1]:bb[3]+1, bb[0]:bb[2]+1]
                seg_mask = seg_mask[bb[1]:bb[3]+1, bb[0]:bb[2]+1]
            else:
                h, w = vid_frame.shape[1:]
                x1 = r.randint(0, w//2)
                x2 = r.randint(w // 2, w-1)
                y1 = r.randint(0, h//2)
                y2 = r.randint(h//2, h-1)
                bb = [x1, y1, x2, y2]
                vid_frame = vid_frame[:, bb[1]:bb[3] + 1, bb[0]:bb[2] + 1]
                seg_mask = seg_mask[bb[1]:bb[3] + 1, bb[0]:bb[2] + 1]
        seg_mask = torch.tensor(seg_mask, dtype=torch.int)
        vid_frame, seg_mask = interp_im_and_mask(vid_frame, seg_mask)
        return vid_frame, seg_mask


def get_surreal_human_seg_dl(dl_kwargs, dataset_kwargs):
    dataset = SurrealHumanSegDataset(**dataset_kwargs)
    dl = DataLoader(dataset=dataset, **dl_kwargs)
    return dl
