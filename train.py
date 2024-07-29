import os
from pathlib import Path
from typing import Optional, Tuple
import pyrootutils

root = pyrootutils.setup_root(
    search_from = __file__,
    indicator = [".git", "pyproject.toml"],
    pythonpath = True,
    dotenv = True
)

import hydra

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.logger import TensorBoardLogger
from pyorch_lightning.plugins.environments import SLURMEnvironment


from yacs.config import CfgNode
from hojore.configs import dataset_config
from hojore.datasets import HOJOREDataModule
from hojore.models.hojore import HOJORE
from hojore.utils.pylogger import get_pylogger
from hojore.utils.misc import task_wrapper, log_hyperparameters


import signal
signal.signal(signal.SIGUSR1, signal.SIG_DFL)

log = get_pylogger(__name__)

@pl.utilities.rank_zero.rank_zero_only
def save_configs(model_cfg: CfgNode, dataset_cfg: CfgNode, rootfdir: str):
    Path(rootdir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config = model_cfg, f = os.path.join(rootdir, 'model_config.yaml'))
    with open(os.path.join(rootdir, 'dataset_config.yaml'), 'w') as f:
        f.write(dataset_cfg.dump())


@task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:

    dataset_cfg = dataset_config()

    save_configs(cfg, dataset_cfg, cfg.path.output_dir)

    datamodule = HOJOREDataModule(cfg, dataset_cfg)

    model = HOJORE(cfg)

    logger = TensorBoardLogger(os.path.join(cfg.paths.output_dir, 'tensorboard'), name='', version='', default_hp_metric=False)
    loggers = [logger]



    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfg.paths.output_dir, 'checkpoints'), 
        every_n_train_steps=cfg.GENERAL.CHECKPOINT_STEPS, 
        save_last=True,
        save_top_k=cfg.GENERAL.CHECKPOINT_SAVE_TOP_K,
    )
    rich_callback = pl.callbacks.RichProgressBar()
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callbacks = [
        checkpoint_callback, 
        lr_monitor,
        # rich_callback
    ]

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, 
        callbacks=callbacks, 
        logger=loggers, 
        #plugins=(SLURMEnvironment(requeue_signal=signal.SIGUSR2) if (cfg.get('launcher',None) is not None) else DDPPlugin(find_unused_parameters=False)), # Submitit uses SIGUSR2
        plugins=(SLURMEnvironment(requeue_signal=signal.SIGUSR2) if (cfg.get('launcher',None) is not None) else None), # Submitit uses SIGUSR2
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # Train the model
    trainer.fit(model, datamodule=datamodule, ckpt_path='last')
    log.info("Fitting done")

    trainer.fit(model, datamodule = datamodule, ckpt_path = 'last')

    log.info("Fitting done")



@hydra.main(version_base="1.2", config_path=str(root/"hojore/configs_hydra"), config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    train(cfg)


if __name__ == "__main__":
    main()




