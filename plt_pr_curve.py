import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset


def plot_pr_curve(config_file, result_file, save_root, metric="bbox"):
    '''plot precison-recall curve based on testing results of pkl file.
    :param config_file: config file path.
    :param result_file: pkl file of testing results path.
    :param save_root: save_root
    :param metric: Metrics to be evaluated. Options are 'bbox', 'segm'.
    :return:
    '''

    cfg = Config.fromfile(config_file)
    # turn on test mode of dataset
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    # build dataset
    dataset = build_dataset(cfg.data.test)
    # load result file in pkl format
    pkl_results = mmcv.load(result_file)
    # convert pkl file (list[list | tuple | ndarray]) to json
    json_results, _ = dataset.format_results(pkl_results)
    # initialize COCO instance
    coco_gt = COCO(annotation_file=cfg.data.test.ann_file)
    coco_dt = coco_gt.loadRes(json_results[metric])
    # initialize COCOeval instance
    coco_eval = COCOeval(coco_gt, coco_dt, metric)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # extract eval data
    precisions = coco_eval.eval["precision"]
    '''
    precisions[T, R, K, A, M]
    T: iou thresholds [0.5 : 0.05 : 0.95], idx from 0 to 9
    R: recall thresholds [0 : 0.01 : 1], idx from 0 to 100
    K: category, idx from 0 to ...
    A: area range, (all, small, medium, large), idx from 0 to 3
    M: max dets, (1, 10, 100), idx from 0 to 2
    '''
    #                      T, R, K, A, M
    pr_arrays = precisions[0, :, 1:, 0, 2]  # 取所有类别的miou=0.5
    px = np.arange(0.0, 1.01, 0.01)
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    for i, y in enumerate(pr_arrays.T):
        ax.plot(px, y, linewidth=1, label='calss%d' % (i))  # plot(recall, precision)
    ax.plot(px, pr_arrays.mean(1), linewidth=3, color='blue', label='all classes')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.grid(alpha=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    save_path = os.path.join(save_root, 'pr_curve_mAP@0.5.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='plt_pr_curve')
    parser.add_argument('--config', default='configs/mask_rcnn/mask_rcnn_r101_fpn_1x_coco.py',
                        help='config file path')
    parser.add_argument('--pkl', default='work_dirs/mask_rcnn_r101_fpn_1x_coco/latest.pkl',
                        help='test latest.pkl')
    parser.add_argument('--save_root', default='results/pr_curve', help='save path')
    parser.add_argument('--metric', default='segm', help='Metrics to be evaluated. Options are "bbox“ "segm" ')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    plot_pr_curve(config_file=args.config, result_file=args.pkl, save_root=args.save_root, metric=args.metric)
