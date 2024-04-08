import os
import torch
import Training.model_utils as model_utils
import General_Utils.cached_dict_utils as cached_dict_utils
import tqdm
import torch.nn.functional as F


def to_pct(f):
    return round(100*f, 1)


@torch.no_grad()
def seg_eval(eval_d: dict, is_val: bool) -> None:
    save_fps = eval_d["save_fps"]
    model = eval_d["model"]


    if not is_val:
        model_name = eval_d["model_name"]
        model_save_fp = os.path.join(save_fps["trained_models_fp"], model_name)
        model = model_utils.load_model_from_save(model_save_fp, model)

    model.eval()
    model.to(0)

    test_dl = eval_d["test_dl"]
    cm_nts = eval_d["class_mapping_num_to_str"]

    eval_results = {}

    test_bgd = eval_d["test_background"]

    cat_keys = list(cm_nts.keys())
    if not test_bgd:
        cat_keys.remove(0)
    for key in cm_nts.keys():
        eval_results[cm_nts[key]] = {"tp": 0, "fp": 0, "fn": 0}

    for data in tqdm.tqdm(test_dl):
        ims = data[0].to(0)
        gt = data[1].to(0)  # (Batch_Size, Height, Width)

        # model prediction
        output = model(ims)
        if isinstance(output, dict):
            output = output["out"]  # shape (Batch_Size, Num_Classes, Height, Width)

        eval_h, eval_w = gt.shape[1:]
        h, w, = output.shape[2:]
        if not (h == eval_h and w == eval_w):
            output = F.interpolate(output, size=(eval_h, eval_w),
                                   mode="bilinear")

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
            gt_eq_mask = (gt == val)
            if test_bgd:
                gt_neq_mask = gt != val
            else:
                gt_neq_mask = torch.logical_and(gt != 0, gt != val)
            pred_eq_mask = (pred == val)
            pred_neq_mask = (pred != val)
            tp = torch.sum(torch.logical_and(gt_eq_mask, pred_eq_mask)).item()
            fn = torch.sum(torch.logical_and(gt_eq_mask, pred_neq_mask)).item()
            fp = torch.sum(torch.logical_and(gt_neq_mask, pred_eq_mask)).item()
            gt_val = int(val)
            eval_results[cm_nts[gt_val]]["tp"] += tp
            eval_results[cm_nts[gt_val]]["fn"] += fn
            eval_results[cm_nts[gt_val]]["fp"] += fp

    # Prediction results
    valid_cats = 0 # categories with at least 1 ground truth pixel
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
            eval_results[cm_nts[key]]["IoU"] = to_pct(tp/union)
            eval_results[cm_nts[key]]["num_pixels"] = num_pix
            all_pixels += num_pix
        correct += tp

    eval_results["acc"] = to_pct(correct/all_pixels)
    iou_sum = 0
    for key in cat_keys:
        fp, tp, fn = (eval_results[cm_nts[key]]["fp"], eval_results[cm_nts[key]]["tp"],
                      eval_results[cm_nts[key]]["fn"])
        if eval_results[cm_nts[key]]["IoU"] != "None":
            union = fp + tp + fn
            iou_sum += tp/union

    mIoU = iou_sum/valid_cats
    eval_results["mIoU"] = to_pct(mIoU)

    if is_val:
        return to_pct(mIoU)

    else:
        eval_save_path = save_fps["eval_results_fp"]
        results_fp = os.path.join(eval_save_path, "class_results.txt")
        cached_dict_utils.write_readable_cached_dict(results_fp, eval_results)
