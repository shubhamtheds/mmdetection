import argparse
import os
import os.path as osp
import warnings
from copy import deepcopy

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--test-folder',
        help='the folder containing different projects for testing')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()

    setup_cache_size_limit_of_dynamo()

    test_folder = args.test_folder
    projects = [d for d in os.listdir(test_folder) if osp.isdir(osp.join(test_folder, d))]

    for project in projects:
        print(f'Processing project: {project}')
        project_path = osp.join(test_folder, project)
        import ipdb;ipdb.set_trace()
        # Load config
        cfg = Config.fromfile(args.config)
        cfg.launcher = args.launcher
        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)

        # Replace placeholders only for test dataset paths
        #cfg.test_dataloader.dataset.ann_file = osp.join(project_path, 'val_coco.json')
        #cfg.test_dataloader.dataset.data_prefix.img = osp.join(project_path, 'tagged_raw_images/')
        test_ann_file = osp.join(project_path, 'val_coco.json')
        test_data_prefix = osp.join(project_path, 'tagged_raw_images/')
        data_root = "./"
        dataset_type = 'CocoDataset'
        backend_args = None
        metainfo = {
            'classes': ('Object', ),
            'palette': [
                (220, 20, 60),
            ]
        }
        test_pipeline = [
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                        'scale_factor'))
        ]
        cfg.test_dataloader = dict(
            batch_size=1,
            num_workers=1,
            persistent_workers=True,
            drop_last=False,
            sampler=dict(type='DefaultSampler', shuffle=False),
            dataset=dict(
                type=dataset_type,
                data_root=data_root,
                ann_file=test_ann_file,
                metainfo=metainfo,
                data_prefix=dict(img=test_data_prefix),
                test_mode=True,
                pipeline=test_pipeline,
                backend_args=backend_args)
        )

        cfg.test_evaluator = dict(
            type='CocoMetric',
            ann_file=data_root + test_ann_file,
            metric='bbox',
            format_only=False,
            backend_args=backend_args)

        # Set work_dir and out paths based on the project name
        cfg.work_dir = osp.join(project_path, 'work_dir')
        out_file_path = osp.join(project_path, f'{project}_result.pkl')

        cfg.load_from = args.checkpoint

        if args.show or args.show_dir:
            cfg = trigger_visualization_hook(cfg, args)

        if args.tta:
            if 'tta_model' not in cfg:
                warnings.warn('Cannot find ``tta_model`` in config, '
                              'we will set it as default.')
                cfg.tta_model = dict(
                    type='DetTTAModel',
                    tta_cfg=dict(
                        nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
            if 'tta_pipeline' not in cfg:
                warnings.warn('Cannot find ``tta_pipeline`` in config, '
                              'we will set it as default.')
                test_data_cfg = cfg.test_dataloader.dataset
                while 'dataset' in test_data_cfg:
                    test_data_cfg = test_data_cfg['dataset']
                cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
                flip_tta = dict(
                    type='TestTimeAug',
                    transforms=[
                        [
                            dict(type='RandomFlip', prob=1.),
                            dict(type='RandomFlip', prob=0.)
                        ],
                        [
                            dict(
                                type='PackDetInputs',
                                meta_keys=('img_id', 'img_path', 'ori_shape',
                                           'img_shape', 'scale_factor', 'flip',
                                           'flip_direction'))
                        ],
                    ])
                cfg.tta_pipeline[-1] = flip_tta
            cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
            cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

        # Build the runner from config
        if 'runner_type' not in cfg:
            runner = Runner.from_cfg(cfg)
        else:
            runner = RUNNERS.build(cfg)

        # Add `DumpResults` dummy metric
        runner.test_evaluator.metrics.append(
            DumpDetResults(out_file_path=out_file_path))

        # Start testing for the current project ##################
        runner.test()

if __name__ == '__main__':
    main()