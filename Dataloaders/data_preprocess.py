import data
import os
from multiprocessing.pool import ThreadPool

import numpy as np
from PIL import Image
from tqdm import tqdm

from Video_Utils import vid_utils
import Dataloaders.surreal_human_seg_dl as sh_seg_dl

dataset_outer_fp = sh_seg_dl.dataset_outer_fp
cwd = data.cwd
frame_outer_save_fp = data.frames_outer_fp
splits = ["train", "test", "val"]
for split in splits:
    outer_fp = os.path.join(frame_outer_save_fp, split)
    if not os.path.exists(outer_fp):
        os.mkdir(outer_fp)


def get_vid_frame_pairs(split: str, ratio: float = (1.0 / 15.0)) -> list[str]:
    vid_fps = sh_seg_dl.get_all_vid_mp4s(split)
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


def determine_split(fp):
    if "test" in fp:
        return "test"
    elif "val" in fp:
        return "val"
    else:
        return "train"


def write_frames(vid_frame_pair: tuple) -> None:
    vid_fp, frame_num = vid_frame_pair
    split = determine_split(vid_fp)
    save_vid_fp = os.path.join(frame_outer_save_fp, split,
                               vid_fp.replace("/", "_").replace(".mp4", "")+"_"
                               +str(frame_num) + ".png")
    if os.path.exists(save_vid_fp):
        return None
    vid_fp_ = os.path.join(dataset_outer_fp, vid_fp[1:])
    vid_frame = vid_utils.get_frame_from_rgb_vid(vid_fp_, frame_num)[0]
    vid_frame_im = Image.fromarray(vid_frame)
    vid_frame_im.save(save_vid_fp)


def write_surreal_dataset_frames() -> None:
    all_frame_pairs = []
    for split in ["train", "test", "val"]:
        all_frame_pairs.extend(sh_seg_dl.read_vid_frame_pairs(split))
    all_frame_pairs.sort()

    # Use tqdm to create a progress bar
    with tqdm(total=len(all_frame_pairs), desc="Processing frames") as pbar:
        def update_progress(*_):
            pbar.update()
        pool = ThreadPool(processes=18)
        for _ in pool.imap_unordered(write_frames, all_frame_pairs):
            update_progress()
        pool.close()
        pool.join()


if __name__ == "__main__":
    write_surreal_dataset_frames()
