#  MMDetection 部分tools 使用脚本

## 画pr曲线
```
python plt_pr_curve.py --config configs/mask_rcnn/mask_rcnn_r101_fpn_1x_coco.py --pkl work_dirs/mask_rcnn_r101_fpn_1x_coco/latest.pkl --save_root pr/results/pr_curve --metric bbox 
```

[//]: # ('lr', 'memory', 'data_time', 'loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'acc', 'loss_bbox', 'loss', 'time', 
[//]: # ('bbox_mAP', 'bbox_mAP_50', 'bbox_mAP_75', 'bbox_mAP_s', 'bbox_mAP_m', 'bbox_mAP_l', 'bbox_mAP_copypaste', 
[//]: # ('segm_mAP', 'segm_mAP_50', 'segm_mAP_75', 'segm_mAP_s', 'segm_mAP_m', 'segm_mAP_l', 'segm_mAP_copypaste')
## 画log中的曲线
```
python plt_logs_curve.py --json_dir work_dirs
```

## 测试Gfloats和参数量
```
python tools/analysis_tools/get_flops.py configs/mask_rcnn/mask_rcnn_r101_fpn_1x_coco.py
```

## 测试FPS
```测试FPS
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500   tools/analysis_tools/benchmark.py  configs/mask_rcnn/mask_rcnn_r101_fpn_1x_coco.py  work_dirs/mask_rcnn_r101_fpn_1x_coco/latest.pth --launcher pytorch
```
