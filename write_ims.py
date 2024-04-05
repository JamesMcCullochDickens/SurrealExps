import data
import vid_utils as vid_utils
import os
import numpy as np
import glob
from PIL import Image

dataset_outer_fp = data.dataset_outer_fp


def get_all_vid_mp4s(split_str: str) -> list[str]:
    outer_folder_fp = os.path.join(dataset_outer_fp, split_str)
    fps = list(glob.iglob(outer_folder_fp+"/**/*.mp4", recursive=True))
    fps.sort()
    return fps


def write_all_ims():
    for split in ["train", "val", "test"]:
        print(f"\n\nProcessing videos in the {split} split.\n\n")
        frames_dir = os.path.join(dataset_outer_fp, split + "_frames")
        if not os.path.exists(frames_dir):
            os.mkdir(frames_dir)
        vid_fps = get_all_vid_mp4s(split)
        for i, vid_fp in enumerate(vid_fps):
            frames = vid_utils.rgb_vid_to_frames(vid_fp)
            vid_name = vid_fp.split("/")[-1][:-4]
            ims_save_outer_fp = os.path.join(frames_dir, vid_name)
            if not os.path.exists(ims_save_outer_fp):
                os.mkdir(ims_save_outer_fp)
            for j, frame in enumerate(frames):
                save_fp = os.path.join(ims_save_outer_fp, str(j+1)+".png")
                im = Image.fromarray(frame)
                im.save(save_fp)
            print(f"Processed {i+1} videos out of {len(vid_fps)}.")


if __name__ == "__main__":
    write_all_ims()