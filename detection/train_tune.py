r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""
import datetime
import os
import time

import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco, get_coco_kp, get_hsdb_vision_coco

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from engine import train_one_epoch, evaluate

import presets
import utils

import ray
import ray.util.sgd.v2 as sgd
from ray.util.sgd.v2.trainer import Trainer
from ray import tune
from ray.tune.integration.mlflow import MLflowLoggerCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter

import numpy as np


def get_dataset(name, image_set, transform, data_path):
    paths = {
        "coco": (data_path, get_coco, 91),
        "coco_kp": (data_path, get_coco_kp, 2),
        "get_hsdb_vision_coco": (data_path, get_hsdb_vision_coco, 32),
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train, data_augmentation):
    return presets.DetectionPresetTrain(
        data_augmentation) if train else presets.DetectionPresetEval()


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Detection Training',
                                     add_help=add_help)

    parser.add_argument(
        '--data-path',
        default=
        '/host_server/raid/jihunyoon/hSDB-vision/new_hSDB_vision/hsdb-vision/gastrectomy-40',
        help='dataset')
    parser.add_argument('--dataset',
                        default='get_hsdb_vision_coco',
                        help='dataset')
    parser.add_argument('--model',
                        default='maskrcnn_resnet50_fpn',
                        help='model')
    parser.add_argument('-j',
                        '--workers',
                        default=4,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument("--address",
                        required=False,
                        type=str,
                        default="auto",
                        help="the address to use for Ray")
    parser.add_argument('-r',
                        '--ray-workers',
                        default=2,
                        type=int,
                        help='number of ray workers (default: 2)')
    parser.add_argument("--use-gpu",
                        action="store_true",
                        default=False,
                        help="Enables GPU training")
    parser.add_argument('--print-freq',
                        default=20,
                        type=int,
                        help='print frequency')
    parser.add_argument("--num-samples",
                        type=int,
                        default=1,
                        help="number of samples for hyperparameter search.")
    parser.add_argument("--tracking-uri",
                        required=True,
                        type=str,
                        help="address for MLflow tracking server")
    parser.add_argument("--experiment-name",
                        required=True,
                        type=str,
                        help="experiment name for MLflow tracking server")
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch',
                        default=0,
                        type=int,
                        help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--rpn-score-thresh',
                        default=None,
                        type=float,
                        help='rpn score threshold for faster-rcnn')
    parser.add_argument('--trainable-backbone-layers',
                        default=None,
                        type=int,
                        help='number of trainable layers of backbone')
    parser.add_argument('--data-augmentation',
                        default="hflip",
                        help='data augmentation policy (default: hflip)')

    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    return parser


def get_data(config):
    # Data loading code
    print("Loading data")

    dataset, num_classes = get_dataset(
        config["dataset"], "train",
        get_transform(True, config["data_augmentation"]), config["data_path"])
    dataset_test, _ = get_dataset(
        config["dataset"], "val",
        get_transform(False, config["data_augmentation"]), config["data_path"])

    print("Creating data loaders")
    if config["distributed"]:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if config["aspect_ratio_group_factor"] >= 0:
        group_ids = create_aspect_ratio_groups(
            dataset, k=config["aspect_ratio_group_factor"])
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids,
                                                  config["batch_size"])
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, config["batch_size"], drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=train_batch_sampler,
        num_workers=config["workers"],
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=8,
        sampler=test_sampler,
        num_workers=config["workers"],
        collate_fn=utils.collate_fn)
    return data_loader, data_loader_test, num_classes, train_sampler


def train_func(config):
    device = torch.device(
        f"cuda:{sgd.local_rank()}" if torch.cuda.is_available() else "cpu")

    data_loader, data_loader_test, num_classes, train_sampler = get_data(
        config)

    print("Creating model")
    kwargs = {"trainable_backbone_layers": config["trainable_backbone_layers"]}
    if "rcnn" in config["model"]:
        if config["rpn_score_thresh"] is not None:
            kwargs["rpn_score_thresh"] = config["rpn_score_thresh"]
    model = torchvision.models.detection.__dict__[config["model"]](
        num_classes=num_classes, pretrained=config["pretrained"], **kwargs)
    model.to(device)
    if config["distributed"] and config["sync_bn"]:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if config["distributed"]:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if torch.cuda.is_available() else None)
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=config["lr"],
                                momentum=config["momentum"],
                                weight_decay=config["weight_decay"])

    config["lr_scheduler"] = config["lr_scheduler"].lower()
    if config["lr_scheduler"] == 'multisteplr':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config["lr_steps"], gamma=config["lr_gamma"])
    elif config["lr_scheduler"] == 'cosineannealinglr':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"])
    else:
        raise RuntimeError(
            "Invalid lr scheduler '{}'. Only MultiStepLR and CosineAnnealingLR "
            "are supported.".format(config["lr_scheduler"]))

    print("Start training")
    sgd_report_results = []
    start_time = time.time()
    for epoch in range(config["start_epoch"], config["epochs"]):
        if config["distributed"]:
            train_sampler.set_epoch(epoch)
        metric_logger = train_one_epoch(model, optimizer, data_loader, device,
                                        epoch, config["print_freq"])
        lr_scheduler.step()
        if config["output_dir"]:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': config,
                'epoch': epoch
            }

        # evaluate after every epoch
        coco_evaluator = evaluate(model, data_loader_test, device=device)
        coco_stats = coco_evaluator.get_stats()
        sgd_report_result = {
            "loss": metric_logger.meters["loss"].value,
            "epoch": epoch,
            "val_bbox_mAP": coco_stats["bbox"][0],
            "val_mask_mAP": coco_stats["segm"][0],
            "val_avg_mAP":
            np.mean([coco_stats["bbox"][0], coco_stats["segm"][0]])
        }
        sgd.report(**sgd_report_result)
        sgd_report_results.append(sgd_report_result)
        sgd.save_checkpoint(epoch=epoch)
        print(f">>>>>>>>>> epoch: {epoch} done.")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    return sgd_report_results


def main(args):
    config = {
        "batch_size":
        tune.choice([8, 16]),
        "epochs":
        tune.choice([13, 20]),
        "lr":
        tune.sample_from(lambda spec: tune.uniform(lower=0.01, upper=0.02)
                         if spec.config.batch_size == 8 else tune.uniform(
                             lower=0.02, upper=0.03)
                         if spec.config.batch_size == 16 else 0.02),
        "momentum":
        0.9,
        "weight_decay":
        0.0001,
        "lr_scheduler":
        "multisteplr",
        "lr_steps":
        tune.sample_from(lambda spec: [8, 11] if spec.config.epochs == 13 else
                         [12, 16] if spec.config.epochs == 20 else [8, 11]),
        "lr_gamma":
        0.1
    }

    config.update(vars(args))

    if config["output_dir"]:
        utils.mkdir(config["output_dir"])

    trainer = Trainer(backend="torch",
                      num_workers=config["ray_workers"],
                      use_gpu=config["use_gpu"])
    Trainable = trainer.to_tune_trainable(train_func)

    scheduler = ASHAScheduler(
        metric="val_avg_mAP",
        mode="max",
        #max_t=10,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter()
    reporter.add_metric_column("epoch")
    reporter.add_metric_column("loss")
    reporter.add_metric_column("val_bbox_mAP")
    reporter.add_metric_column("val_mask_mAP")
    reporter.add_metric_column("val_avg_mAP")

    analysis = tune.run(Trainable,
                        num_samples=config["num_samples"],
                        config=config,
                        scheduler=scheduler,
                        verbose=2,
                        progress_reporter=reporter,
                        callbacks=[
                            MLflowLoggerCallback(
                                tracking_uri=config["tracking_uri"],
                                experiment_name=config["experiment_name"],
                                save_artifact=True)
                        ])
    results = analysis.get_best_config(metric="val_avg_mAP", mode="max")
    print(results)

    return results


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    ray.init(address=args.address)
    main(args)
