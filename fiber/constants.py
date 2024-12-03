EXCHANGE_SYMMETRIC_KEY_ENDPOINT = "exchange-symmetric-key"
PUBLIC_ENCRYPTION_KEY_ENDPOINT = "public-encryption-key"
SYMMETRIC_KEY_UUID = "symmetric-key-uuid"
HOTKEY = "hotkey"
MINER_HOTKEY = "miner-hotkey"
VALIDATOR_HOTKEY = "validator-hotkey"
NEURON_INFO_LITE = "NeuronInfoLite"

FINNEY_NETWORK = "finney"
FINNEY_TEST_NETWORK = "test"
FINNEY_SUBTENSOR_ADDRESS = "wss://entrypoint-finney.opentensor.ai:443"
FINNEY_TEST_SUBTENSOR_ADDRESS = "wss://test.finney.opentensor.ai:443/"

EMPTY_COMMITMENT_FIELD_TYPE = "None"

SUBTENSOR_NETWORK_TO_SUBTENSOR_ADDRESS = {
    FINNEY_NETWORK: FINNEY_SUBTENSOR_ADDRESS,
    FINNEY_TEST_NETWORK: FINNEY_TEST_SUBTENSOR_ADDRESS,
}


NONCE = "nonce"
SIGNATURE = "signature"


SAVE_NODES_FILEPATH = "nodes/nodes.json"

SS58_FORMAT = 42
U16_MAX = 65535
