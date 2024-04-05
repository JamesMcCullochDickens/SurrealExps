import numpy as np
from decord import VideoReader, cpu
import cv2


def rgb_vid_to_frames(rgb_vid_fp: str) -> np.ndarray:
    with open(rgb_vid_fp, 'rb') as f:
        vr = VideoReader(f, ctx=cpu(0))
    vid_len = len(vr)
    frames = vr.get_batch(list(range(0, vid_len))).asnumpy()
    return frames


def get_frame_from_rgb_vid(rgb_vid_fp: str, frame_num: int) -> np.ndarray:
    with open(rgb_vid_fp, 'rb') as f:
        vr = VideoReader(f, ctx=cpu(0))
    frame = vr.get_batch([frame_num]).asnumpy()
    return frame


def get_vid_len_decord(vid_fp: str) -> int:
    with open(vid_fp, 'rb') as f:
        vr = VideoReader(f, ctx=cpu(0))
    return len(vr)


def get_vid_len_opencv(vid_fp: str) -> int:
    cap = cv2.VideoCapture(vid_fp)
    if not cap.isOpened():
        print("Error: Could not open the video.")
        exit(-1)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
