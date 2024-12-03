
import requests
import re
import torch
import numpy as np
import wandb
import time
import json
import copy
import torch
import asyncio
import datetime as dt
import httpx
import traceback

from cryptography.fernet import Fernet
from utils.version import get_version, __version__, last_acceptable_version
from typing import List
from detection import version_url
from validator.config import get_subnet_config
from fiber.constants import FINNEY_SUBTENSOR_ADDRESS
from fiber.chain.metagraph import Metagraph
from fiber.chain.interface import get_substrate
from fiber.chain.chain_utils import load_hotkey_keypair, load_coldkeypub_keypair
from fiber.logging_utils import get_logger
from fiber.constants import FINNEY_SUBTENSOR_ADDRESS
from fiber.chain.weights import set_node_weights, process_weights_for_netuid, convert_weights_and_uids_for_emit
from fiber.chain.post_ip_to_chain import post_node_ip_to_chain
from fiber.chain.models import Node
from fiber.validator import handshake, client
from datetime import datetime, time
from typing import List
from traceback import print_exception
from detection.validator.data_generator import DataGenerator
from detection.validator.models import ValDataRow
from detection.validator.text_completion import OllamaModel
from detection import (
    __version__, WANDB_PROJECT,
    WANDB_ENTITY, MAX_RUN_STEPS_PER_WANDB_RUN
)
from validator.forward import forward

logger = get_logger(__name__)

class Validator:
    
    @property
    def block(self):
        if not self.last_block_fetch or (datetime.now() - self.last_block_fetch).seconds >= 12:
            self.current_block = self.substrate.get_block_number(None)  # type: ignore
            self.last_block_fetch = datetime.now()
            self.attempted_set_weights = False

        return self.current_block

    def __init__(self, config=None):

        self.subnet_config = get_subnet_config()
        logger.info(self.subnet_config)
        
        # If a gpu is required, set the device to cuda:N (e.g. cuda:0)
        self.device = self.subnet_config.neuron.device
        self.version = get_version()

        logger.info("Setting up bittensor objects.")

        while True:
            try:
                logger.info("Initializing subtensor and metagraph")
                
                self.subtensor_url = FINNEY_SUBTENSOR_ADDRESS # this should be customized according to your specific need
                
                logger.info(f"Subtensor url: {self.subtensor_url}")
                
                self.substrate = get_substrate(
                    subtensor_address = self.subtensor_url
                )
                
                self.metagraph = Metagraph(
                    substrate = self.substrate,
                    netuid =  self.subnet_config.netuid,
                    load_old_nodes = False,
                )
                
                self.metagraph.sync_nodes()
                
                break
            except Exception as e:
                logger.error("Couldn't init subtensor and metagraph with error: {}".format(e))
                logger.error("If you use public RPC endpoint try to move to local node")
                time.sleep(5)

        logger.info(f"Metagraph: {self.metagraph}")

        # Check if the miner is registered on the Bittensor network before proceeding further.
        self.check_registration()

        # Parse versions for weight_version check
        self.parse_versions()
        
        self.keypair = load_hotkey_keypair(
            wallet_name=self.subnet_config.wallet_name,
            hotkey_name=self.subnet_config.wallet_hotkey,
        )        
        self.coldkey_keypair = load_coldkeypub_keypair(wallet_name = self.subnet_config.wallet_name)
        
        self.hotkey = self.keypair.ss58_address
        self.coldkey = self.coldkey_keypair.ss58_address
        self.uid = self.hotkeys.index(self.hotkey)
        self.stakes = [node.stake for node in self.metagraph.nodes.values()]
        self.uids = [node.node_id for node in self.metagraph.nodes.values()]
        self.hotkeys = list(self.metagraph.nodes.keys())
        
        logger.info(
            f"Running neuron on subnet: {self.subnet_config.netuid} with uid {self.uid} using network: {self.subtensor_url}"
        )
        
        self.step = 0
        self.last_metagraph_sync = 0
        self.should_serve_axon = self.subnet_config.should_serve_axon
        self.external_ip = self.subnet_config.external_ip
        self.external_port=self.subnet_config.external_port
        
        if self.should_serve_axon:
            try:
                success = post_node_ip_to_chain(
                    substrate = self.substrate,
                    keypair = self.keypair,
                    netuid = self.subnet_config.netuid,
                    external_ip = self.external_ip,
                    external_port = self.external_port,
                    coldkey_ss58_address = self.coldkey,
                )
                if success:
                    logger.info(f"Puhsted ip and port successfully to the chain. ip: {self.external_ip}, port: {self.external_port}")
                else:
                    logger.info(f"Failed ip and port successfully to the chain. ip: {self.external_ip}, port: {self.external_port}")
            except Exception as e:
                logger.error(f"Failed to post ip and port to the chain. ip: {self.external_ip}, port: {self.external_port}")
            
        self.wandb_run = None

        logger.info("Building validation weights.")

        # Instead of loading zero weights we take latest weights from the previous run
        # If it is first run for validator then it will be filled with zeros
        self.scores = weight_metagraph.W[self.uid].to(self.device)

        # Init sync with the network. Updates the metagraph.
        self.sync()
        
        self.new_wandb_run()

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
     
        logger.info("load_state()")
        self.load_state()

        models = [OllamaModel(model_name='llama3:text'),
                  OllamaModel(model_name='llama3.1'),
                  OllamaModel(model_name='llama3.2'),
                  OllamaModel(model_name='llama2:13b'),

                  OllamaModel(model_name='qwen2.5:14b'),
                  OllamaModel(model_name='qwen2.5:32b'),
                  OllamaModel(model_name='qwen:32b-text-v1.5-q4_0'),

                  OllamaModel(model_name='command-r'),
                  OllamaModel(model_name='command-r'),
                  OllamaModel(model_name='command-r'),

                  OllamaModel(model_name='gemma2:9b-instruct-q4_0'),
                  OllamaModel(model_name='gemma2:27b-text-q4_0'),

                  OllamaModel(model_name='mistral:text'),
                  OllamaModel(model_name='mistral-nemo:12b'),
                  OllamaModel(model_name='mistral-small:22b'),

                  OllamaModel(model_name='internlm2:7b'),
                  OllamaModel(model_name='internlm2:20b'),

                  OllamaModel(model_name='yi:34b-chat'),
                  OllamaModel(model_name='deepseek-v2:16b'),
                  OllamaModel(model_name='openhermes'),]

        logger.info(f"Models loaded{models}")

        self.generator = DataGenerator(models)
        logger.info(f"Generator initialized {self.generator}")

        self.out_of_domain_f1_scores = np.ones(257)
        self.out_of_domain_alpha = 0.15

    async def build_queries(self) -> tuple[List[ValDataRow], np.array]:
        logger.info(f"Generating texts for challenges...")
        data = self.generator.generate_data(n_human_samples=50, n_ai_samples=150)
        texts = [el for el in data]
        labels = [el.segmentation_labels for el in data]
        return texts, labels,
        
    def parse_versions(self):
        self.version = __version__
        self.least_acceptable_version = last_acceptable_version

        logger.info(f"Parsing versions...")
        response = requests.get(version_url)
        logger.info(f"Response: {response.status_code}")
        if response.status_code == 200:
            content = response.text

            version_pattern = r"__version__\s*=\s*['\"]([^'\"]+)['\"]"
            least_acceptable_version_pattern = r"__least_acceptable_version__\s*=\s*['\"]([^'\"]+)['\"]"

            try:
                version = re.search(version_pattern, content).group(1)
                least_acceptable_version = re.search(least_acceptable_version_pattern, content).group(1)
            except AttributeError as e:
                logger.error(f"While parsing versions got error: {e}")
                return

            self.version = version
            self.least_acceptable_version = least_acceptable_version
        return

    def sync(self):
        """
        Wrapper for synchronizing the state of the network for the given validator.
        """
        # Ensure miner or validator hotkey is still registered on the network.
        self.check_registration()

        try:
            if self.should_sync_metagraph():
                self.last_metagraph_sync = self.block
                self.resync_metagraph()
                # Parse versions for weight_check
                self.parse_versions()

            if self.should_set_weights():
                self.set_weights()

            # Always save state.
            self.save_state()
        except Exception as e:
            logger.error("Coundn't sync metagraph or set weights: {}".format(e))
            logger.error("If you use public RPC endpoint try to move to local node")
            time.sleep(5)
            
    def should_set_weights(self) -> bool:
        last_update = self.metagraph.nodes[self.keypair.ss58_address].last_updated
        blocks_elapsed = self.block - last_update
        epoch_length = self.subnet_config.neuron.epoch_length

        return blocks_elapsed >= epoch_length

    def should_sync_metagraph(self):
        """
        Check if enough epoch blocks have elapsed since the last checkpoint to sync.
        """
        blocks_elapsed = self.block - self.last_metagraph_sync
        
        return (blocks_elapsed >= self.subnet_config.neuron.epoch_length)
    
    def check_registration(self):
        hotkey = self.keypair.ss58_address
        if hotkey not in self.hotkeys:
            logger.error(
                f"Wallet: {self.keypair} is not registered on netuid {self.metagraph.netuid}."
            )
    
    async def try_handshake(
        self,
        async_client: httpx.AsyncClient,
        server_address: str,
        keypair,
        hotkey
    ) -> tuple:
        return await handshake.perform_handshake(
            async_client, server_address, keypair, hotkey
        )
    
    async def _handshake(self, node: Node, async_client: httpx.AsyncClient) -> Node:
        node_copy = node.model_copy()
        server_address = client.construct_server_address(
            node=node,
            replace_with_docker_localhost = True,
            replace_with_localhost = False,
        )

        try:
            symmetric_key, symmetric_key_uid = await self.try_handshake(
                async_client, server_address, self.keypair, node.hotkey
            )
        except Exception as e:
            if isinstance(e, (httpx.HTTPStatusError, httpx.RequestError, httpx.ConnectError)):
                if hasattr(e, "response"):
                    # logger.debug(f"Response content: {e.response.text}")
                    pass
            return node_copy

        fernet = Fernet(symmetric_key)
        node_copy.fernet = fernet
        node_copy.symmetric_key_uuid = symmetric_key_uid
        return node_copy

    async def perform_handshakes(self, nodes: list[Node]) -> list[Node]:
        tasks = []
        shaked_nodes: list[Node] = []
        for node in nodes:
            if node.fernet is None or node.symmetric_key_uuid is None:
                tasks.append(self._handshake(node, httpx.AsyncClient))
            if len(tasks) > 50:
                shaked_nodes.extend(await asyncio.gather(*tasks))
                tasks = []

        if tasks:
            shaked_nodes.extend(await asyncio.gather(*tasks))

        nodes_where_handshake_worked = [
            node for node in shaked_nodes if node.fernet is not None and node.symmetric_key_uuid is not None
        ]
        if len(nodes_where_handshake_worked) == 0:
            logger.info("❌ Failed to perform handshakes with any nodes!")
            return []
        logger.info(f"✅ performed handshakes successfully with {len(nodes_where_handshake_worked)} nodes!")

        return shaked_nodes    

    def run(self):
        
        self.sync()
        logger.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                logger.info(f"step({self.step}) block({self.block})")

                # Run multiple forwards concurrently.
                self.forward()

                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync()

                self.step += 1

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            if self.wandb_run:
                print("Finishing wandb service...")
                self.wandb_run.finish()
            self.axon.stop()
            logger.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            logger.error("Error during validation", str(err))
            logger.debug(
                print_exception(type(err), err, err.__traceback__)
            )

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """

        try:
            res = await forward(self)
            return res
        except Exception as e:
            logger.error("Got error in forward function")
            logger.info(traceback.format_exc())
            return None
        
    def log_step(
            self,
            uids,
            metrics,
            rewards
    ):
        # If we have already completed X steps then we will complete the current wandb run and make a new one.     
        if self.step % MAX_RUN_STEPS_PER_WANDB_RUN == 0:
            step_log = {
                "timestamp": time.time(),
                "uids": uids.tolist(),
                "uid_metrics": {},
            }
            logger.info(
                f"Validator has completed {self.step} run steps. Creating a new wandb run."
            )
            self.wandb_run.finish()
            self.new_wandb_run()

            for i, uid in enumerate(uids):
                step_log["uid_metrics"][str(uid.item())] = {
                    "uid": uid.item(),
                    "weight": self.scores[uid].item(),
                    "reward": rewards[i].item()
                }
                step_log["uid_metrics"][str(uid.item())].update(metrics[i])

            graphed_data = {
                "block": self.metagraph.block.item(),
                "uid_data": {
                    str(uids[i].item()): rewards[i].item() for i in range(len(uids))
                },
                "weight_data": {str(uid.item()): self.scores[uid].item() for uid in uids},
            }

            logger.info(
                f"step_log: {step_log}"
            )
            logger.info(
                f"graphed_data: {graphed_data}"
            )
            original_format_json = json.dumps(step_log)

            signed_msg = f'0x{self.keypair.sign(original_format_json).hex()}'
            logger.info("Logging to Wandb")
            self.wandb_run.log(
                {
                    **graphed_data,
                    "original_format_json": original_format_json,
                    "signed_msg": signed_msg
                },
                step=self.step,
            )
            logger.info("Logged")

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        logger.info("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)
        
        nodes = self.metagraph.nodes()
        
        self.metagraph.sync_nodes()

        self.check_registration()

        if self.hotkeys == list(self.metagraph.nodes.keys()):
            return
        
        logger.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != nodes[uid].hotkey:
                self.scores[uid] = 0  

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) != len(self.metagraph.nodes):
            # Update the size of the moving average scores.
            new_moving_average = torch.zeros(len(self.metagraph.nodes)).to(
                self.device
            )
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average

        # Update the hotkeys.
        self.stakes = [node.stake for node in self.metagraph.nodes.values()]
        self.uids = [node.node_id for node in self.metagraph.nodes.values()]
        self.hotkeys = list(self.metagraph.nodes.keys())        
        self.last_metagraph_sync = self.block

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """

        # Check if self.scores contains any NaN values and log a warning if it does.
        if torch.isnan(self.scores).any():
            logger.warning(
                f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        # m = torch.nn.Softmax()
        # raw_weights = m(self.scores * 4)
        raw_weights = torch.nn.functional.normalize(self.scores, p=1, dim=0)

        logger.debug("raw_weights", raw_weights)
        logger.debug("raw_weight_uids", self.uids.to("cpu"))
        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = process_weights_for_netuid(
            uids=self.uids.to("cpu"),
            weights=raw_weights.to("cpu"),
            netuid=self.subnet_config.netuid,
            # subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        logger.debug("processed_weights", processed_weights)
        logger.debug("processed_weight_uids", processed_weight_uids)

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
        logger.debug("uint_weights", uint_weights)
        logger.debug("uint_uids", uint_uids)

        result = set_node_weights(
            self.substrate,
            self.keypair,
            node_ids=list(range(len(self.metagraph.nodes))),
            node_weights=uint_weights,
            netuid=self.metagraph.netuid,
            validator_node_id=self.uid,
            version_key=self.version,
        )
        
        if result is True:
            logger.info("set_weights on chain successfully!")
        else:
            logger.error(f"set_weights failed")    
        
    def update_scores(self, rewards: torch.FloatTensor, uids: List[int]):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        # Check if rewards contains NaN values.
        if torch.isnan(rewards).any():
            logger.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = torch.nan_to_num(rewards, 0)

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        scattered_rewards: torch.FloatTensor = self.scores.scatter(
            0, torch.tensor(uids).to(self.device), rewards
        ).to(self.device)
        logger.debug(f"Scattered rewards: {rewards}")

        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        alpha: float = self.subnet_config.neuron.moving_average_alpha
        self.scores: torch.FloatTensor = alpha * scattered_rewards + (
                1 - alpha
        ) * self.scores.to(self.device)
        logger.debug(f"Updated moving avg scores: {self.scores}")

    def save_state(self):
        """Saves the state of the validator to a file."""
        logger.info("Saving validator state.")

        # Save the state of the validator to file.
        torch.save(
            {
                "step": self.step,
                "scores": self.scores,
                "hotkeys": self.hotkeys,
            },
            self.subnet_config.neuron.full_path + "/state.pt",
        )

    def load_state(self):
        """Loads the state of the validator from a file."""
        logger.info("Loading validator state.")

        # Load the state of the validator from file.
        state = torch.load(self.subnet_config.neuron.full_path + "/state.pt")
        self.step = state["step"]
        self.scores = state["scores"]
        self.hotkeys = state["hotkeys"]

    def new_wandb_run(self):
        """Creates a new wandb run to save information to."""
        # Create a unique run id for this run.
        run_id = dt.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")

        name = "validator-" + str(self.uid) + "-" + run_id
        self.wandb_run = wandb.init(
            name=name,
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            anonymous='must',
            config={
                "uid": self.uid,
                "hotkey": self.hotkey,
                "run_name": run_id,
                "version": __version__,
            },
            allow_val_change=True
        )

        logger.debug(f"Started a new wandb run: {name}")

if __name__ == "__main__":
    val = Validator()
    asyncio.run(val.run())