import os
import argparse
from dataclasses import dataclass, field
from typing import Optional, List
from loguru import logger

@dataclass
class NeuronConfig:
    name: str = "miner"
    device: str = "cuda:0"
    epoch_length: int = 100
    events_retention_size: str = "2 GB"
    dont_save_events: bool = False
    sample_size: int = 256
    timeout: int = 20
    disable_set_weights: bool = False
    moving_average_alpha: float = 0.1
    vpermit_tao_limit: int = 4096
    out_of_domain_min_f1_score: float = 0.9
    full_path: str = ""

@dataclass
class BlacklistConfig:
    exclude: List[int] = field(default_factory=list)

@dataclass
class SubnetConfig:
    netuid: int = 251
    should_serve_axon: bool = False
    external_ip: str = "127.0.0.1"
    external_port: int = 8090
    wallet_name: Optional[str] = None
    wallet_hotkey: Optional[str] = None
    neuron: NeuronConfig = field(default_factory=NeuronConfig)
    blacklist: BlacklistConfig = field(default_factory=BlacklistConfig)

def get_subnet_config() -> SubnetConfig:
    parser = argparse.ArgumentParser()
    
    # Basic args
    parser.add_argument("--netuid", type=int, default=32, help="Subnet netuid")
    parser.add_argument("--should_serve_axon", type=bool, default=False, help="Intention if validator wants to set IP and port to the chain")
    parser.add_argument("--external_ip", type=str, default="127.0.0.1", help="External IP to push to the chain")
    parser.add_argument("--external_port", type=int, default=8090, help="External port address to push to the chain")
    parser.add_argument("--wallet_name", type=str, default=None)
    parser.add_argument("--wallet_hotkey", type=str, default=None)
    
    # Neuron args
    parser.add_argument("--neuron_name", type=str, default="validator")
    parser.add_argument("--neuron_device", type=str, default="cuda:0")
    parser.add_argument("--neuron_epoch_length", type=int, default=500)
    parser.add_argument("--neuron_events_retention_size", type=str, default="2 GB")
    parser.add_argument("--neuron_dont_save_events", action="store_true", default=False)
    
    # Validator-specific        
    parser.add_argument("--neuron_sample_size", type=int, default=256, help="The number of miners to query in a single step.")
    parser.add_argument("--neuron_timeout", type=int, default=20, help="Timeout")
    parser.add_argument("--neuron_disable_set_weights", action="store_true", default=False, help="Disables setting weights.")
    parser.add_argument("--neuron_moving_average_alpha", type=float, default=0.1, help="Moving average alpha parameter.")
    parser.add_argument("--neuron_vpermit_tao_limit", type=int, default=4096, help="Max number of TAO allowed with a vpermit.")
    parser.add_argument("--neuron_out_of_domain_min_f1_score", type=float, default=0.9, help="Min f1 score for out-of-domain validation.")
    parser.add_argument("--exclude", type=int, nargs='*', default=[], help="List of excluded miners.")

    args = parser.parse_args()

    # Create config objects
    neuron_config = NeuronConfig(
        name=args.neuron_name,
        device=args.neuron_device,
        epoch_length=args.neuron_epoch_length,
        events_retention_size=args.neuron_events_retention_size,
        dont_save_events=args.neuron_dont_save_events,
        neuron_sample_size=args.neuron_sample_size,
        neuron_timeout=args.neuron_timeout,
        neuron_disable_set_weights=args.neuron_disable_set_weights,
        neuron_moving_average_alpha=args.neuron_moving_average_alpha,
        neuron_vpermit_tao_limit=args.neuron_vpermit_tao_limit,
        neuron_out_of_domain_min_f1_score=args.neuron_out_of_domain_min_f1_score,
        full_path=os.path.expanduser("storing")  
    )

    blacklist_config = BlacklistConfig(
        exclude=args.exclude
    )

    return SubnetConfig(
        netuid=args.netuid,
        should_serve_axon=args.should_serve_axon,
        external_ip=args.external_ip,
        external_port=args.external_port,
        wallet_name=args.wallet_name,
        wallet_hotkey=args.wallet_hotkey,
        neuron=neuron_config,
        blacklist=blacklist_config,
    )
