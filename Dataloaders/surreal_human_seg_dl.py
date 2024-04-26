import data
import os
from PIL import Image
import numpy as np
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from mat_proc import get_seg_mask_at_frame, get_person_bb
from typing import List, Tuple, Optional
import torch.nn.functional as F
import random as r
import math
import General_Utils.cached_dict_utils as cached_dict_utils
import multiprocessing
import General_Utils.RAM_utils as RAM_utils
import time

dataset_outer_fp = data.dataset_outer_fp
cwd = os.environ["cwd"]
frame_outer_fp = "/media/H/Surreal_Dataset_Frames"

sorted_parts = ['background', 'hips', 'leftUpLeg', 'rightUpLeg', 'spine', 'leftLeg', 'rightLeg',
                'spine1', 'leftFoot', 'rightFoot', 'spine2', 'leftToeBase', 'rightToeBase',
                'neck', 'leftShoulder', 'rightShoulder', 'head', 'leftArm', 'rightArm',
                'leftForeArm', 'rightForeArm', 'leftHand', 'rightHand', 'leftHandIndex1', 'rightHandIndex1']
class_mapping_num_to_str = {i: k for (i, k) in enumerate(sorted_parts)}
class_mapping_str_to_num = {v: k for (k, v) in class_mapping_num_to_str.items()}


def get_all_vid_mp4s(split_str: str) -> list[str]:
    outer_folder_fp = os.path.join(dataset_outer_fp, split_str)
    fps = list(glob.iglob(outer_folder_fp + "/**/*.mp4", recursive=True))
    fps.sort()
    return fps


def load_image(vid_fp, split, frame_num, frame_outer_fp_):
    frame_inner_fp = (vid_fp.replace("/", "_").replace(".mp4", "")
                      + "_" + str(frame_num) + ".png")
    frame_fp = os.path.join(frame_outer_fp_, split, frame_inner_fp)
    vid_frame_original = np.asarray(Image.open(frame_fp))
    return vid_fp+"_"+str(frame_num), vid_frame_original


def load_ims_to_RAM(vid_fp_pairs: list, split: str,
                    max_load: float = 0.9, ram_checking_interval: int = 1000):
    ims_d = {}
    total_RAM = RAM_utils.get_total_ram()
    RAM_in_use = RAM_utils.get_ram_in_use()
    load = RAM_in_use / total_RAM
    if load > max_load:
        print(f"No images loaded in RAM, you have already exceeded the load factor.")
        return ims_d

    print(f"Loading frames to RAM.")
    t1 = time.time()
    with multiprocessing.Pool(processes=12) as pool:
        results = []
        for vid_fp, frame_num in vid_fp_pairs:
            results.append(pool.apply_async(load_image, args=(vid_fp, split, frame_num, frame_outer_fp)))

        for i, res in tqdm.tqdm(enumerate(results), total=len(vid_fp_pairs)):
            key, vid_frame_original = res.get()
            ims_d[key] = vid_frame_original

            if i and i % ram_checking_interval == 0:
                RAM_in_use = RAM_utils.get_ram_in_use()
                load = RAM_in_use / total_RAM
                if load > max_load:
                    print(f"Load factor exceeded at {len(ims_d.keys())} images loaded in RAM.")
                    break

        """
        for res in results[i + 1:]:
            res.get()
        """
    t2 = time.time()
    print(f"Loaded {len(ims_d.keys())} images in {round(t2-t1, 3)} seconds.")
    return ims_d


def get_class_imbalance_ratios(with_normalize: bool = True) -> torch.Tensor:
    d = cached_dict_utils.read_cached_dict(
        os.path.join(cwd, "Dataloaders/imbalance_ratios"))
    keys = list(d.keys())
    keys.sort()
    arr = [d[k] for k in keys]
    arr = torch.tensor(np.array(arr), dtype=torch.float32)
    if with_normalize:
        arr = arr / arr.max()
    return arr


def get_part_color_map() -> dict:
    global part_d_num_to_str
    keys = list(range(25))
    color_d = {}
    biases, gain, power = [80, 160, 240], 2, 3
    for key in keys:
        r = int(math.pow((key+biases[0])*gain, power)) % 255.0
        g = int(math.pow((key+biases[1])*gain, power)) % 255.0
        b = int(math.pow((key+biases[2])*gain, power)) % 255.0
        color_d[key] = np.array([r, g, b], dtype=np.uint8)
    return color_d


def mask_to_color_im(mask: np.ndarray, is_binary: bool) -> np.ndarray:
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    if not is_binary:
        color_d = get_part_color_map()
        unique_vals = np.unique(mask)
        for val in unique_vals:
            if val == 0:
                continue
            spatial_mask = mask == val
            color_mask[spatial_mask, :] = color_d[val]
    else:
        color_mask[mask == 1] = 255
    return color_mask


def visualize(im: np.ndarray, mask_: np.ndarray) -> None:
    im_ = im.copy()
    mask = mask_.copy()
    if im.shape[0] == 3:
        im_ = np.transpose(im, (1, 2, 0))
    rgb_im = Image.fromarray(im_)
    rgb_im.show()
    color_mask = mask_to_color_im(mask)
    color_mask_im = Image.fromarray(color_mask)
    color_mask_im.show()
    rgb_mask = (0.65*im_ + 0.35*color_mask).astype(np.uint8)
    rgb_mask_im = Image.fromarray(rgb_mask)
    rgb_mask_im.show()


def get_visualizations(im: np.ndarray, mask_: np.ndarray) -> None:
    im_ = im.copy()
    mask = mask_.copy()
    if im.shape[0] == 3:
        im_ = np.transpose(im, (1, 2, 0))
    color_mask = mask_to_color_im(mask)
    color_mask_im = Image.fromarray(color_mask)
    rgb_mask = (0.65*im_ + 0.35*color_mask).astype(np.uint8)
    rgb_mask_im = Image.fromarray(rgb_mask)
    return color_mask_im, rgb_mask_im


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


def rgb_channel_denormalize(f: torch.Tensor) -> torch.Tensor:
    imagenet_means = [0.485, 0.456, 0.406]
    imagenet_stds = [0.229, 0.224, 0.225]
    f_ = f.clone()
    for i in range(3):
        f_[i, :, :] = (imagenet_stds[i]*f[i, :, :]) + imagenet_means[i]
    f_ *= 255.0
    return f_


def interp_im_and_mask(im: torch.Tensor, mask: Optional[torch.Tensor],
                       inter_shape: Tuple[int, int]):
    im = torch.unsqueeze(im, dim=0)
    im = F.interpolate(im, size=inter_shape, mode="bilinear")[0]
    if mask is not None:
        mask = torch.unsqueeze(torch.unsqueeze(mask, 0), 0).float()
        mask = F.interpolate(mask, size=inter_shape, mode="nearest")[0][0].long()
    return im, mask


def preprocess_im(im_fp: str, crop_bb: list,
                  interp_size: tuple = (300, 200)) -> torch.Tensor:
    im_arr = np.asarray(Image.open(im_fp))
    im_arr = np.transpose(im_arr, (-1, 0, 1))
    im_arr = torch.tensor(im_arr, dtype=torch.float)
    im_arr = im_arr[:, crop_bb[1]:crop_bb[3]+1, crop_bb[0]:crop_bb[2]+1]
    im_arr *= (1.0/255.0)
    im_arr = rgb_channel_normalize(im_arr)
    im_arr = torch.unsqueeze(im_arr, dim=0)
    im_arr = F.interpolate(im_arr, size=(interp_size), mode="bilinear")
    return im_arr


def get_loose_bb(bb: np.ndarray, h: int, w: int, padding: int = 5) -> np.ndarray:
    x_min = max(bb[0]-padding, 0)
    y_min = max(bb[1]-padding, 0)
    x_max = min(bb[2]+padding, w-1)
    y_max = min(bb[3]+padding, h-1)
    return np.array([x_min, y_min, x_max, y_max], dtype=int)


def pad_seg_mask(mask: torch.Tensor, pad_h: int, pad_w:int) -> torch.Tensor:
    padded_mask = torch.zeros(size=(pad_h, pad_w), device=mask.device, dtype=mask.dtype)
    original_h, original_w = mask.shape[:]
    padded_mask[:original_h+1, :original_w+1] = mask
    return mask


class SurrealHumanSegDataset(Dataset):
    def __init__(self, split: str,
                 with_rgb_masking: bool = False,
                 with_bb_cropping: bool = False,
                 interp_size: Optional[Tuple[int, int]] = None,
                 debug: bool = False,
                 load_to_RAM: bool = False,
                 ):
        super().__init__()
        assert split in ["train", "val", "test"], "invalid split"
        self.is_test = split not in ["train", "val"]
        self.split = split
        self.with_rgb_masking = with_rgb_masking
        self.with_bb_cropping = with_bb_cropping
        self.interp_size = interp_size
        # A consistent size of segmentation masks and images is required at train time
        if not self.interp_size and self.with_bb_cropping:
            self.interp_size = (300, 200)
        self.vid_frame_fps = read_vid_frame_pairs(split)
        if debug:
            r.shuffle(self.vid_frame_fps)
            self.vid_frame_fps = self.vid_frame_fps[:int(0.01 * len(self.vid_frame_fps))]
        self.load_to_RAM = load_to_RAM
        if self.load_to_RAM:
            self.loaded_ims = load_ims_to_RAM(self.vid_frame_fps, self.split)

    def __len__(self):
        return len(self.vid_frame_fps)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        is_loaded = False
        vid_fp, frame_num = self.vid_frame_fps[index]
        if self.load_to_RAM:
            key = vid_fp+"_"+str(frame_num)
            if key in self.loaded_ims.keys():
                vid_frame_original = self.loaded_ims[key]
                is_loaded = True

        # read from videos
        """
        vid_fp = os.path.join(dataset_outer_fp, vid_fp[1:])
        vid_frame_original = vid_utils.get_frame_from_rgb_vid(vid_fp, frame_num)[0]  # shape H, W, 3
        """

        # read from frames
        if not is_loaded:
            frame_inner_fp = (vid_fp.replace("/", "_").replace(".mp4", "")
                              + "_" + str(frame_num) + ".png")
            frame_fp = os.path.join(frame_outer_fp, self.split, frame_inner_fp)
            vid_frame_original = np.asarray(Image.open(frame_fp))
        vid_fp = os.path.join(dataset_outer_fp, vid_fp[1:])

        # seg mask
        seg_mask = get_seg_mask(vid_fp, frame_num)

        # sanity check, visualize the original data
        #visualize(vid_frame_original, seg_mask)

        # preprocess rgb
        vid_frame = torch.tensor(vid_frame_original, dtype=torch.float).permute((-1, 0, 1)) / 255.0
        vid_frame = rgb_channel_normalize(vid_frame)

        if self.with_rgb_masking:
            zero_mask = torch.unsqueeze(torch.tensor(seg_mask, dtype=torch.long) != 0, dim=0).repeat((3, 1, 1))
            vid_frame = zero_mask * vid_frame
        if self.with_bb_cropping:
            h, w = vid_frame.shape[1:]
            if self.with_bb_cropping:
                bb = get_person_bb(seg_mask)
                if bb is None: # No person in the frame, take entire image
                    bb = [0, 0, w-1, h-1]
                else:
                    if not self.is_test:
                        bb = get_loose_bb(bb, h, w)

                vid_frame = vid_frame[:, bb[1]:bb[3] + 1, bb[0]:bb[2] + 1]
                seg_mask = seg_mask[bb[1]:bb[3] + 1, bb[0]:bb[2] + 1]

                # sanity check, visualize the cropped images/mask and overlay
                #vid_frame_original_cropped = vid_frame_original[bb[1]:bb[3] + 1, bb[0]:bb[2] + 1, :]
                #visualize(vid_frame_original_cropped, seg_mask)

        seg_mask = torch.tensor(seg_mask, dtype=torch.long)
        if self.interp_size:
            if vid_frame.shape[1] != self.interp_size[0] or vid_frame.shape[2] != self.interp_size[1]:
                if not self.is_test:
                    vid_frame, seg_mask = interp_im_and_mask(vid_frame, seg_mask, self.interp_size)
                else: # do not interpolate the ground truth seg mask
                    vid_frame, _ = interp_im_and_mask(vid_frame, None, self.interp_size)


        # final sanity check, after all transforms, denormalize the rgb data and view the (possibly)
        # interpolated image/mask pairs and overlays
        #vid_frame_denormalized = rgb_channel_denormalize(vid_frame)
        #visualize(vid_frame_denormalized.numpy().astype(np.uint8), seg_mask.numpy().astype(np.uint8))

        return vid_frame, seg_mask


def get_train_class_ratios() -> None:
    import pprint
    total = 0
    class_counts = {}
    save_fp = "./human_seg_class_counts.txt"
    dl_kwargs = {"num_workers": 16, "batch_size": 16,
                 "shuffle": False, "drop_last": False,
                 "pin_memory": False, "persistent_workers": False}
    dataset_kwargs = {"split": "train", "with_cropping": True,
                      "cropping_type": "bb", "interp_size": (240, 320)}
    dl = get_surreal_human_seg_dl(dl_kwargs, dataset_kwargs)
    for (_, seg_mask) in tqdm.tqdm(dl):
        seg_mask = seg_mask.to(0)
        unique = torch.unique(seg_mask).cpu().numpy().tolist()
        for val in unique:
            if val not in class_counts.keys():
                class_counts[val] = 0
            val = int(val)
            count = torch.sum(seg_mask == val).cpu().item()
            class_counts[val] += count
            total += count
    pprint.pprint(class_counts)
    keys = list(class_counts.keys())
    keys.sort()
    with open(save_fp, "w") as f:
        for key in keys:
            count = class_counts[key]
            ratio = round(count/total, 2)
            s = str(key) + ":" + str(count) + ":" + str(ratio)
            f.write(s+"\n")
        f.write("total:" + str(total))
    total = 0
    for k in class_counts.keys():
        total += class_counts[k]
    save_fp = "./human_seg_class_counts.txt"
    keys = list(class_counts.keys())
    keys.sort()
    with open(save_fp, "w") as f:
        for key in keys:
            count = class_counts[key]
            ratio = round(count / total, 5)
            s = str(key) + ":" + str(count) + ":" + str(ratio)
            f.write(s + "\n")
        f.write("total:" + str(total))
    imbalance_ratios = {}
    for k in keys:
        imbalance_ratios[k] = class_counts[0]/class_counts[k]
    save_fp2 = "./imbalance_ratios"
    cached_dict_utils.write_readable_cached_dict(save_fp2, imbalance_ratios)


def get_surreal_human_seg_dl(dl_kwargs, dataset_kwargs):
    dataset = SurrealHumanSegDataset(**dataset_kwargs)
    dl = DataLoader(dataset=dataset, **dl_kwargs)
    return dl


if __name__ == "__main__":
    import tqdm
    time_d = {}
    dl_kwargs = {"num_workers": 0, "batch_size": 1,
                 "shuffle": False, "drop_last": True,
                 "pin_memory": False, "persistent_workers": False}
    dataset_kwargs = {"split": "train", "with_bb_cropping": True,
                      "interp_size": None, "with_rgb_masking": True}
    dl = get_surreal_human_seg_dl(dl_kwargs, dataset_kwargs)
    for data in tqdm.tqdm(dl):
        pass
