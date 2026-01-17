# The MIT License (MIT)
 # Copyright © 2024 It's AI
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import copy
import time
import traceback

import requests
import re

import bittensor as bt

from abc import ABC, abstractmethod

# Sync calls set weights and also resyncs the metagraph.
from detection.utils.config import check_config, add_args, config
from detection.utils.misc import ttl_get_block
from detection import __spec_version__ as spec_version
from detection import version_url


class BaseNeuron(ABC):
    """
    Base class for Bittensor miners. This class is abstract and should be inherited by a subclass. It contains the core logic for all neurons; validators and miners.

    In addition to creating a wallet, subtensor, and metagraph, this class also handles the synchronization of the network state via a basic checkpointing mechanism based on epoch length.
    """

    neuron_type: str = "BaseNeuron"

    @classmethod
    def check_config(cls, config: "bt.Config"):
        check_config(cls, config)

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    @classmethod
    def config(cls):
        return config(cls)

    subtensor: "bt.subtensor"
    wallet: "bt.wallet"
    metagraph: "bt.metagraph"
    spec_version: int = spec_version

    @property
    def block(self):
        return ttl_get_block(self)

    def __init__(self, config=None):
        base_config = copy.deepcopy(config or BaseNeuron.config())
        self.config = self.config()
        self.config.merge(base_config)
        self.check_config(self.config)

        # Set up logging with the provided configuration and directory.
        bt.logging(config=self.config, logging_dir=self.config.full_path)

        # If a gpu is required, set the device to cuda:N (e.g. cuda:0)
        self.device = self.config.neuron.device

        # Log the configuration for reference.
        bt.logging.info(self.config)

        # Build Bittensor objects
        # These are core Bittensor classes to interact with the network.
        bt.logging.info("Setting up bittensor objects.")

        # The wallet holds the cryptographic key pairs for the miner.

        self.wallet = bt.Wallet(config=self.config)
        while True:
            try:
                bt.logging.info("Initializing subtensor and metagraph")
                self.subtensor = bt.Subtensor(config=self.config)
                self.metagraph = self.subtensor.metagraph(self.config.netuid)
                break
            except Exception as e:
                bt.logging.error("Couldn't init subtensor and metagraph with error: {}".format(e))
                bt.logging.error("If you use public RPC endpoint try to move to local node")
                time.sleep(5)

        bt.logging.info(f"Wallet: {self.wallet}")
        bt.logging.info(f"Subtensor: {self.subtensor}")
        bt.logging.info(f"Metagraph: {self.metagraph}")

        # Check if the miner is registered on the Bittensor network before proceeding further.
        self.check_registered()

        # Parse versions for weight_version check
        self.parse_versions()

        # Each miner gets a unique identity (UID) in the network for differentiation.
        self.uid = self.metagraph.hotkeys.index(
            self.wallet.hotkey.ss58_address
        )
        bt.logging.info(
            f"Running neuron on subnet: {self.config.netuid} with uid {self.uid} using network: {self.subtensor.chain_endpoint}"
        )
        self.step = 0
        self.last_update = 0        

    @abstractmethod
    async def forward(self, synapse: bt.Synapse) -> bt.Synapse:
        ...

    @abstractmethod
    def run(self):
        ...

    def sync(self):
        """
        Wrapper for synchronizing the state of the network for the given miner or validator.
        """
        # Ensure miner or validator hotkey is still registered on the network.
        self.check_registered()
        

        try:
            if self.should_sync_metagraph():
                self.last_update = self.block
                self.resync_metagraph()
                # Parse versions for weight_check
                self.parse_versions()

            if self.should_set_weights():
                self.set_weights()

            # Always save state.
            self.save_state()
        except Exception as e:
            bt.logging.error("Coundn't sync metagraph or set weights: {}".format(traceback.format_exc()))
            bt.logging.error("If you use public RPC endpoint try to move to local node")
            time.sleep(5)
            
    def check_registered(self):
        # --- Check for registration.
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            exit()

    def should_sync_metagraph(self):
        """
        Check if enough epoch blocks have elapsed since the last checkpoint to sync.
        """
        if self.neuron_type != "MinerNeuron":
            last_update = self.metagraph.last_update[self.uid]
        else:
            last_update = self.last_update
            
        return (
            self.block - last_update
        ) > self.config.neuron.epoch_length

    def should_set_weights(self) -> bool:
        # Don't set weights on initialization.
        if self.step == 0:
            return False

        # Check if enough epoch blocks have elapsed since the last epoch.
        if self.config.neuron.disable_set_weights:
            return False


        # Define appropriate logic for when set weights.
        return (
            (self.block - self.metagraph.last_update[self.uid])
            > self.config.neuron.epoch_length
            and self.neuron_type != "MinerNeuron"
        )  # don't set weights if you're a miner

    def save_state(self):
        pass

    def load_state(self):
        pass

    def parse_versions(self):
        self.version = "10.0.0"
        self.least_acceptable_version = "0.0.0"

        bt.logging.info(f"Parsing versions...")
        response = requests.get(version_url)
        bt.logging.info(f"Response: {response.status_code}")
        if response.status_code == 200:
            content = response.text

            version_pattern = r"__version__\s*=\s*['\"]([^'\"]+)['\"]"
            least_acceptable_version_pattern = r"__least_acceptable_version__\s*=\s*['\"]([^'\"]+)['\"]"

            try:
                version = re.search(version_pattern, content).group(1)
                least_acceptable_version = re.search(least_acceptable_version_pattern, content).group(1)
            except AttributeError as e:
                bt.logging.error(f"While parsing versions got error: {e}")
                return

            self.version = version
            self.least_acceptable_version = least_acceptable_version
        return
