#  MMDetection 部分tools 使用脚本

## 画pr曲线
```
python plt_pr_curve.py --config configs/mask_rcnn/mask_rcnn_r101_fpn_1x_coco.py --pkl work_dirs/mask_rcnn_r101_fpn_1x_coco/latest.pkl --save_root pr/results/pr_curve --metric bbox 
```

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

