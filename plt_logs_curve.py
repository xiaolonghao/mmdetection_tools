import argparse
import glob
import json
import os.path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_curve(args, log_dicts, json_names):
    if args.backend is not None:
        plt.switch_backend(args.backend)
    if args.style is not None:
        sns.set_style(args.style)
    # if legend is None, use {filename}_{key} as legend
    legend = args.legend
    if args.legend is None:
        legend = []
        for json_log in json_names:
            for metric in args.keys:
                legend.append(f'{json_log}_{metric}')
    metrics = args.keys
    num_metrics = len(metrics)

    # ['iter', 'lr', 'memory', 'data_time', 'loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'acc', 'loss_bbox', 'loss', 'time',
    # 'bbox_mAP', 'bbox_mAP_50', 'bbox_mAP_75', 'bbox_mAP_s', 'bbox_mAP_m', 'bbox_mAP_l', 'bbox_mAP_copypaste',
    # 'segm_mAP', 'segm_mAP_50', 'segm_mAP_75', 'segm_mAP_s', 'segm_mAP_m', 'segm_mAP_l', 'segm_mAP_copypaste']
    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        for j, metric in enumerate(metrics):
            if 'mAP' in metric:
                xs = []
                ys = []
                for epoch in epochs:
                    ys += log_dict[epoch][metric]
                    if 'val' in log_dict[epoch]['mode']:
                        xs.append(epoch)
                plt.xlabel('epoch')
                plt.ylabel(metric)
                plt.plot(xs, ys, label=legend[i * num_metrics + j], marker='o')
            else:
                xs = []
                ys = []
                num_iters_per_epoch = log_dict[epochs[0]]['iter'][-2]
                for epoch in epochs:
                    iters = log_dict[epoch]['iter']
                    if log_dict[epoch]['mode'][-1] == 'val':
                        iters = iters[:-1]
                    if args.plt_epoch:
                        xs.append(np.array([epoch]))
                        ys.append(np.array([np.array(log_dict[epoch][metric][:len(iters)]).mean()]))
                    else:
                        xs.append(np.array(iters) + (epoch - 1) * num_iters_per_epoch)
                        ys.append(np.array(log_dict[epoch][metric][:len(iters)]))
                xs = np.concatenate(xs)
                ys = np.concatenate(ys)
                if args.plt_epoch:
                    plt.xlabel('epoch')
                else:
                    plt.xlabel('iter')
                plt.ylabel(metric)
                plt.plot(xs, ys, label=legend[i * num_metrics + j], linewidth=0.5)
            plt.legend()
        if args.title is not None:
            plt.title(args.title)
    save_path = os.path.join(args.out_dir, '%s.png' % (metrics[0]))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f'save curve to: {save_path}')
    plt.savefig(save_path)
    plt.show()
    plt.close()


def load_json_logs(json_logs):
    # load and convert json_logs to log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics, e.g. memory, bbox_mAP
    # value of sub dict is a list of corresponding values of all iterations
    log_dicts = [dict() for _ in json_logs]
    for json_log, log_dict in zip(json_logs, log_dicts):
        with open(json_log, 'r') as log_file:
            for i, line in enumerate(log_file):
                log = json.loads(line.strip())
                # skip the first training info line
                if i == 0:
                    continue
                # skip lines without `epoch` field
                if 'epoch' not in log:
                    continue
                epoch = log.pop('epoch')
                if epoch not in log_dict:
                    log_dict[epoch] = defaultdict(list)
                for k, v in log.items():
                    log_dict[epoch][k].append(v)
    return log_dicts


def parse_args():
    parser_plt = argparse.ArgumentParser('plt_logs_curve')
    parser_plt.add_argument('--json_dir', default='work_dirs', help='the metric that you want to plot')
    parser_plt.add_argument('--json_names', default=None, help='legend name if legend is None')
    parser_plt.add_argument('--keys',
                            default=['loss'],
                            help='the metric that you want to plot')
    parser_plt.add_argument('--out_dir', type=str, default='log_curve', help='out_dir')
    parser_plt.add_argument('--plt-epoch', type=bool, default=False, help='plt @epoch')
    parser_plt.add_argument('--start-epoch', type=str, default='1', help='the epoch that you want to start')
    parser_plt.add_argument('--eval-interval', type=str, default='1', help='the eval interval when training')
    parser_plt.add_argument('--title', type=str, default=None, help='title of figure')
    parser_plt.add_argument('--legend', type=str, default=None, help='legend of each plot')
    parser_plt.add_argument('--backend', type=str, default=None, help='backend of plt')
    parser_plt.add_argument('--style', type=str, default=None, help='style of plt:dark')
    args = parser_plt.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    json_dir = args.json_dir
    json_logs = glob.glob(os.path.join(json_dir, '*/*.json'))
    json_names = args.json_names
    if json_names is None:
        json_names = [os.path.basename(os.path.dirname(x)) for x in json_logs]
    log_dicts = load_json_logs(json_logs)
    plot_curve(args, log_dicts, json_names)
