import os
from typing import Optional

import torch
import torch.nn.functional as F
import tqdm
import pprint
import numpy as np

import Training.model_utils as model_utils
import General_Utils.cached_dict_utils as cached_dict_utils
from Dataloaders.surreal_human_seg_dl import rgb_channel_denormalize
import Visualizations.vis_utils as vis_utils

NUM_VISUALIZATION_IMAGES = 1000


def to_pct(f):
    return round(100 * f, 1)


@torch.no_grad()
def compute_model_inference(ims: torch.Tensor, gt: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    output = model(ims)
    if isinstance(output, dict):
        output = output["out"]  # shape (Batch_Size, Num_Classes, Height, Width)

    eval_h, eval_w = gt.shape[1:]
    h, w, = output.shape[2:]
    if not (h == eval_h and w == eval_w):
        output = F.interpolate(output, size=(eval_h, eval_w), mode="bilinear")
    return output


@torch.no_grad()
def seg_eval(eval_d: dict, is_val: bool, dl: Optional[torch.utils.data.DataLoader]) -> None:
    save_fps = eval_d["save_fps"]
    model = eval_d["model"]

    if not is_val:
        model_name = eval_d["model_name"]
        model_save_fp = os.path.join(save_fps["trained_models_fp"], model_name)
        model = model_utils.load_model_from_save(model_save_fp, model)

    model.eval()
    model.to(0)

    if is_val and not dl:
        test_dl = eval_d["val_dl"]
    elif not is_val and not dl:
        test_dl = eval_d["test_dl"]
    else:  # train accuracy
        test_dl = dl

    cm_nts = eval_d["class_mapping_num_to_str"]
    eval_save_path = save_fps["eval_results_fp"]

    eval_results = {}

    test_bgd = eval_d["test_background"]
    with_visualization = eval_d.get("with_visualization", True)

    cat_keys = list(cm_nts.keys())
    if not test_bgd:
        cat_keys.remove(0)
    for key in cm_nts.keys():
        eval_results[cm_nts[key]] = {"tp": 0, "fp": 0, "fn": 0}

    num_ims = 0
    """
    for data in tqdm.tqdm(test_dl):
        ims = data[0].to(0)
        gt = data[1].to(0)  # (Batch_Size, Height, Width)
        batch_size = ims.shape[0]
        num_ims += batch_size

        # model prediction
        output = compute_model_inference(ims, gt, model)
        pred = torch.argmax(output, dim=1)  # (Batch_Size, Height, Width)

        # class-wise eval
        unique_pred = torch.unique(pred).cpu().numpy().tolist()
        unique_gt = torch.unique(gt).cpu().numpy().tolist()

        if not test_bgd:
            if 0 in unique_gt:
                unique_gt.remove(0)
            if 0 in unique_pred:
                unique_pred.remove(0)
        unique_pred_s = set(unique_pred)
        unique_gt_s = set(unique_gt)
        all_vals = unique_pred_s.union(unique_gt_s)

        for val in all_vals:
            gt_val = int(val)

            gt = gt.reshape(-1)
            pred = pred.reshape(-1)

            if not test_bgd:
                non_zero_idx = gt != 0
                gt = gt[non_zero_idx]
                pred = pred[non_zero_idx]

            pred_eq_mask = (pred == val)
            pred_neq_mask = ~ pred_eq_mask
            gt_eq_mask = (gt == val)
            gt_neq_mask = ~ gt_eq_mask

            tp = torch.sum(torch.logical_and(gt_eq_mask, pred_eq_mask)).item()
            fn = torch.sum(torch.logical_and(gt_eq_mask, pred_neq_mask)).item()
            fp = torch.sum(torch.logical_and(gt_neq_mask, pred_eq_mask)).item()
            eval_results[cm_nts[gt_val]]["tp"] += tp
            eval_results[cm_nts[gt_val]]["fn"] += fn
            eval_results[cm_nts[gt_val]]["fp"] += fp

    # Prediction results
    eval_results["num_images"] = num_ims
    valid_cats = 0  # categories with at least 1 ground truth pixel
    all_pixels = 0
    correct = 0
    for key in cat_keys:
        fp, tp, fn = (eval_results[cm_nts[key]]["fp"], eval_results[cm_nts[key]]["tp"],
                      eval_results[cm_nts[key]]["fn"])
        union = fp + tp + fn
        num_pix = tp + fn
        if num_pix == 0:
            eval_results[cm_nts[key]]["IoU"] = "None"
        else:
            valid_cats += 1
            eval_results[cm_nts[key]]["IoU"] = to_pct(tp / union)
            eval_results[cm_nts[key]]["num_pixels"] = num_pix
            all_pixels += num_pix
        correct += tp

    eval_results["acc"] = to_pct(correct / all_pixels)
    iou_sum = 0
    for key in cat_keys:
        fp, tp, fn = (eval_results[cm_nts[key]]["fp"], eval_results[cm_nts[key]]["tp"],
                      eval_results[cm_nts[key]]["fn"])
        if eval_results[cm_nts[key]]["IoU"] != "None":
            union = fp + tp + fn
            iou_sum += tp / union

    mIoU = iou_sum / valid_cats
    eval_results["mIoU"] = to_pct(mIoU)

    if is_val:
        return to_pct(mIoU)

    else:
        print(f"Evaluation complete...\n")
        pprint.pprint(eval_results)
        results_fp = os.path.join(eval_save_path, "class_results.txt")
        cached_dict_utils.write_readable_cached_dict(results_fp, eval_results)

    """
    if with_visualization:
        num_ims_visualized = 0
        print(f"Writing visualization images of model output")
        vis_save_path = os.path.join(eval_save_path, "visualizations")
        if not os.path.exists(vis_save_path):
            os.mkdir(vis_save_path)
        test_dl.num_workers = 0
        for data in tqdm.tqdm(test_dl):
            ims = data[0].to(0)
            gt = data[1].to(0)  # (Batch_Size, Height, Width)

            # model prediction
            output = compute_model_inference(ims, gt, model)
            pred_mask = torch.argmax(output, dim=1).cpu()  # (Batch_Size, Height, Width)
            ims = ims.to("cpu")
            for i, im in enumerate(ims):
                save_fp = os.path.join(vis_save_path, str(num_ims_visualized+1)+".png")
                eval_h, eval_w = gt[i].shape[:]
                im = F.interpolate(torch.unsqueeze(im, dim=0), size=(eval_h, eval_w), mode="bilinear")[0]
                im_denormalized = rgb_channel_denormalize(im)
                im_denormalized = im_denormalized.numpy().astype(np.uint8)
                gt_seg_mask = gt[i].cpu()
                h, w, = gt_seg_mask.shape[:]
                gt_seg_mask = gt_seg_mask.reshape(-1)
                pred_mask = pred_mask.reshape(-1)
                zero_mask = torch.tensor(gt_seg_mask != 0, dtype=torch.int)
                pred_mask *= zero_mask
                gt_seg_mask = gt_seg_mask.reshape((h, w)).numpy().astype(np.uint8)
                pred_mask = pred_mask.reshape((h, w)).numpy().astype(np.uint8)
                vis_utils.save_model_pred_ims(im_denormalized, pred_mask, gt_seg_mask, save_fp)
                num_ims_visualized += 1

            if num_ims_visualized >= NUM_VISUALIZATION_IMAGES:
                del test_dl
                break