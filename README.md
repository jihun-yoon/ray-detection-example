# ray-detection-example

This repository is for training an Mask R-CNN model with [Ray SGD](https://docs.ray.io/en/latest/raysgd/v2/raysgd.html), [Tune](https://docs.ray.io/en/latest/tune/index.html) and online/batch serving with [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) in "How Hutom uses Ray and PyTorch to Scale Surgical Video Analysis and Review" blog. Training Mask R-CNN code is based on [PyTorch official detection reference training scripts](https://github.com/pytorch/vision/tree/master/references/detection) and Following codes are major changes to use Ray SGD and Tune.


1. Using Ray SGD wrapper on `torch.nn.parallel.DistributedDataParallel`.

    ```python
    import ray.util.sgd.v2 as sgd

    device = torch.device(
            f"cuda:{sgd.local_rank()}" if torch.cuda.is_available() else "cpu")

    model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if torch.cuda.is_available() else None)
    ```

2. Using Ray SGD `Trainer` and converting into Ray Tune `Trainable`.
    ```python
    from ray.util.sgd.v2.trainer import Trainer

    def train_func():
        # Train model
        ...

        # Evaluate model
        ...
    
    trainer = Trainer(backend="torch",
                      num_workers=config["ray_workers"],
                      use_gpu=config["use_gpu"])
    Trainable = trainer.to_tune_trainable(train_func)
    ```

3. Using Ray Tune and Ray built-in MLflow logger on Ray SGD `Trainable` for hyperparameter search.
    ```python
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.integration.mlflow import MLflowLoggerCallback

    scheduler = ASHAScheduler(
        metric="val_avg_mAP",
        mode="max",
        grace_period=1,
        reduction_factor=2)

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
    ```

## Prerequisites
The codes were tested on
* python 3.8
* torch 1.7.1
* torchvision 0.8.2
* cudatoolkit 11.0
* ray 1.7
