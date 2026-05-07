# Yuqi Yang

import os
import cv2
import imageio
import numpy as np
import json
import torch
import scipy.io as sio
from cofai.engine.run_eval import task_decode_pred
# from detection_toolbox.det_tools import bbox2json, bbox2fig

class PerformanceMeter(object):
    """ A general performance meter which shows performance across one or more tasks """
    def __init__(self, p, tasks):
        self.database = p['train_db_name']
        if 'db_name' in p.keys():
            self.database = p['db_name']
        self.tasks = tasks
        self.meters = {t: get_single_task_meter(p, self.database, t) for t in self.tasks}

    def reset(self):
        for t in self.tasks:
            self.meters[t].reset()

    def update(self, pred, gt):
        for t in self.tasks:
            self.meters[t].update(pred[t], gt[t])

    def get_score(self, verbose=True):
        eval_dict = {}
        for t in self.tasks:
            m = self.meters[t]
            eval_dict[t] = m.compute() if hasattr(m, "compute") else m.get_score(verbose)

        return eval_dict


def _ignore_index(p):
    return p["ignore_index"] if isinstance(p, dict) else p.ignore_index


def get_single_task_meter(p, database, task):
    """Retrieve a meter to measure the single-task performance (``cofai.metrics``)."""

    ignore_index = _ignore_index(p)

    if task == "semseg":
        from cofai.metrics import SemanticSegmentationMeter

        return SemanticSegmentationMeter(database, ignore_idx=ignore_index)

    if task == "scene":
        from cofai.metrics import SceneClassificationMeter

        return SceneClassificationMeter(database)

    if task == "human_parts":
        from cofai.metrics import HumanPartSegmentationMeter

        return HumanPartSegmentationMeter(database, ignore_idx=ignore_index)

    if task == "normals":
        from cofai.metrics import SurfaceNormalsEstimationMeter

        return SurfaceNormalsEstimationMeter(ignore_index=ignore_index)

    if task == "sal":
        from cofai.metrics import SaliencyDetectionMeter

        return SaliencyDetectionMeter(
            ignore_index=ignore_index, threshold_step=0.05, beta_squared=0.3
        )

    if task == "depth":
        from cofai.metrics import DepthEstimationMeter

        if isinstance(p, dict):
            tc = p.get("TASKS", {}) or {}
            max_depth = tc.get("depth_max", 10.0)
            min_depth = tc.get("depth_min", 0.001)
        else:
            max_depth = p.TASKS.depth_max
            min_depth = p.TASKS.depth_min
        return DepthEstimationMeter(max_depth=max_depth, min_depth=min_depth)

    if task == "edge":
        from cofai.metrics import EdgeDetectionMeter

        ew = (
            p.get("edge_w", 0.95)
            if isinstance(p, dict)
            else float(getattr(p, "edge_w", 0.95))
        )
        return EdgeDetectionMeter(pos_weight=ew, ignore_index=ignore_index)

    raise NotImplementedError

@torch.no_grad()
def save_model_pred_for_one_task(p, batch_idx, sample, output, save_dirs, task=None, epoch=None):
    """ Save model predictions for one task"""

    inputs, meta = sample['image'].cuda(non_blocking=True), sample['meta']

    if task == 'semseg':
        output_task = task_decode_pred(output[task], task).cpu().data.numpy()
    else:
        output_task = task_decode_pred(output[task], task)#.cpu().data.numpy()

    for jj in range(int(inputs.size()[0])):
        if len(sample[task][jj].unique()) == 1 and sample[task][jj].unique() == p.ignore_index:
            continue
        fname = meta['img_name'][jj]

        im_height = meta['img_size'][jj][0]
        im_width = meta['img_size'][jj][1]
        pred = output_task[jj] # (H, W) or (H, W, C)
        # if we used padding on the input, we crop the prediction accordingly
        if (im_height, im_width) != pred.shape[:2]:
            delta_height = max(pred.shape[0] - im_height, 0)
            delta_width = max(pred.shape[1] - im_width, 0)
            if delta_height > 0 or delta_width > 0:
                height_begin = torch.div(delta_height, 2, rounding_mode="trunc")
                height_location = [height_begin, height_begin + im_height]
                width_begin =torch.div(delta_width, 2, rounding_mode="trunc")
                width_location = [width_begin, width_begin + im_width]
                pred = pred[height_location[0]:height_location[1],
                            width_location[0]:width_location[1]]
        assert pred.shape[:2] == (im_height, im_width)
        if pred.ndim == 3:
            raise
        result = pred.cpu().numpy()
        if task == 'depth':
            sio.savemat(os.path.join(save_dirs[task], fname + '.mat'), {'depth': result})
        else:
            imageio.imwrite(os.path.join(save_dirs[task], fname + '.png'), result.astype(np.uint8))
