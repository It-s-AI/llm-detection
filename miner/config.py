# config.py
import os
import argparse
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class NeuronConfig:
    name: str = "miner"
    device: str = "cuda:0"
    epoch_length: int = 100
    events_retention_size: str = "2 GB"
    dont_save_events: bool = False
    # Miner specific
    deberta_foundation_model_path: str = "models/deberta-v3-large-hf-weights"
    deberta_model_path: str = "models/deberta-large-ls03-ctx1024.pth"
    ppl_model_path: str = "models/ppl_model.pk"
    model_type: str = "deberta"

@dataclass
class BlacklistConfig:
    minimum_stake_requirement: int = 1_000

@dataclass
class Subnet_Config:
    netuid: int = 32
    neuron: NeuronConfig = field(default_factory=NeuronConfig)
    blacklist: BlacklistConfig = field(default_factory=BlacklistConfig)

def get_subnet_config() -> Subnet_Config:
    parser = argparse.ArgumentParser()
    
    # Basic args
    parser.add_argument("--netuid", type=int, default=32, help="Subnet netuid")
    
    # Neuron args
    parser.add_argument("--neuron_name", type=str, default="mminer")
    parser.add_argument("--neuron_device", type=str, default="cuda:0")
    parser.add_argument("--neuron_epoch_length", type=int, default=500)
    parser.add_argument("--neuron_events_retention_size", type=str, default="6 GB")
    parser.add_argument("--neuron_dont_save_events", action="store_true", default=False)
    
    # Miner specific args
    parser.add_argument("--blacklist_minimum_stake_requirement", type=int, default=1_000)
    parser.add_argument(
        "--neuron_deberta_foundation_model_path",
        type=str,
        default="models/deberta-v3-large-hf-weights"
    )
    parser.add_argument(
        "--neuron_deberta_model_path",
        type=str,
        default="models/deberta-large-ls03-ctx1024.pth"
    )
    parser.add_argument(
        "--neuron_ppl_model_path",
        type=str,
        default="models/ppl_model.pk"
    )
    parser.add_argument(
        "--neuron_model_type",
        type=str,
        default="deberta"
    )

    args = parser.parse_args()
    
    # Create config objects
    neuron_config = NeuronConfig(
        name=args.neuron_name,
        device=args.neuron_device,
        epoch_length=args.neuron_epoch_length,
        events_retention_size=args.neuron_events_retention_size,
        dont_save_events=args.neuron_dont_save_events,
        deberta_foundation_model_path=args.neuron_deberta_foundation_model_path,
        deberta_model_path=args.neuron_deberta_model_path,
        ppl_model_path=args.neuron_ppl_model_path,
        model_type=args.neuron_model_type
    )
    
    blacklist_config = BlacklistConfig(
        minimum_stake_requirement=args.blacklist_minimum_stake_requirement
    )
    
    return Subnet_Config(
        netuid=args.netuid,
        neuron=neuron_config,
        blacklist=blacklist_config
    )
