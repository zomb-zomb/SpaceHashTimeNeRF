from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import nerfstudio.data.dataparsers.dnerf_dataparser as dnerf_dataparser
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager,VanillaDataManagerConfig
from nerfstudio.models.instant_ngp import InstantNGPModelConfig, NGPModel
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig, DynamicBatchPipeline
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, OptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig, SchedulerConfig
from model import SpaceTimeHashingModelConfig, SpaceTimeHashingUniformModelConfig
from pipeline import SpaceTimeHashingPipelineConfig, SpaceTimeHashingPipeline
from pathlib import Path
from trainer import TrainerConfig, Trainer


import torch

path = Path("/media/zomb/HDD/dataset/D-NeRF/data/jumpingjacks/")
output_dir = Path("/media/zomb/HDD/output/")
experiment_name = "SpaceTimeHashing"
method_name = "SpaceTimeHashing"  
timestamp = "2023_11_30"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

optimizers={
    "proposal_networks": {
        "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
        "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-6, max_steps=100000),
    },
    "fields": {
        "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
        "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-6, max_steps=100000),
    },
}

if __name__ == '__main__':
    # Create a new instance of the data parser
    parser_config = dnerf_dataparser.DNeRFDataParserConfig(data=path)
    # Create a new instance of the data manager
    datamanager_config = VanillaDataManagerConfig(dataparser=parser_config)
    # model
    model_config = SpaceTimeHashingModelConfig(background_color="white")
    pipeline_config = VanillaPipelineConfig(datamanager=datamanager_config, model=model_config)

    # trainer
    trainer_config = TrainerConfig(pipeline=pipeline_config, optimizers=optimizers,
                                    output_dir=output_dir, experiment_name=experiment_name,
                                    method_name=method_name, timestamp=timestamp)
    trainer = Trainer(trainer_config)
    trainer.setup()
    trainer.train()
    # Print the data
    # print(dataparser_outputs)