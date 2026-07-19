# 🧑‍🏫 Validating

# System Requirements

Validators will need enough processing power to inference multiple models. It is required to have a GPU (we commend NVIDIA A100) with minimum 80GB of VRAM. 
Also you need to have at least 1T of disk space.

## Installation

Make sure that your server provider support systemd (RunPod doesn't support it).
Otherwise ollama service won't be restarting automatically and you'll have to restart it on your own from time to time.

1. Clone the repo

```bash
apt update && apt upgrade -y
git clone https://github.com/It-s-AI/llm-detection
```

1. Setup your python [virtual environment](https://docs.python.org/3/library/venv.html) or [Conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).
2. Install the requirements. From your virtual environment, run

```shell
cd llm-detection
python3 -m pip install -e .
python3 -m pip uninstall mathgenerator -y
python3 -m pip install git+https://github.com/synapse-alpha/mathgenerator.git
```

1. Make sure you've [created a Wallet](https://docs.bittensor.com/getting-started/wallets) and [registered a hotkey](https://docs.bittensor.com/subnets/register-and-participate).

```bash
btcli w new_coldkey
btcli w new_hotkey
btcli s register --netuid 32 --wallet.name YOUR_COLDKEY --wallet.hotkey YOUR_HOTKEY
```

1. Export wandb api key for logging

```bash
export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"
```

## Install driver

Install PM2 and the jq package on your system.

```bash
sudo apt update && sudo apt install jq && sudo apt install npm && sudo npm install pm2 -g && pm2 update
```

Make `run.sh` file executable.  

```bash
chmod +x run.sh
```

So Ollama models can detect GPUs on your system

```bash
apt update
apt install lshw -y
```

## Download models

Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Run ollama service in background (make sure that you don't have any running instances of ollama before running this command)

```
pm2 start --name ollama "ollama serve"
```

If you want to update your pulled models run this:

```
ollama list | tail -n +2 | awk '{print $1}' | while read -r model; do
  ollama pull $model
done
```

Install cc_net (validator-only).

First install the project and the build toolchain cc_net needs to compile
kenlm/sentencepiece:

```bash
sudo apt-get install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev zlib1g-dev libbz2-dev liblzma-dev zip unzip -y
pip install -e .
```

Then build and install cc_net itself. `TAR_OPTIONS=--no-same-owner` avoids a
`tar: Cannot change ownership ... Operation not permitted` failure when running
as root in a container. Install cc_net (editable) *before* `make install` so
that `make` skips its own internal `pip install .` step.

```bash
cd cc_net
export TAR_OPTIONS="--no-same-owner"
pip install -e .
make install
make lang=en dl_lm
cd ..
```

Note: cc_net must be installed into the same environment your validator runs in.
If you manage the venv with `uv`, use `uv pip install -e .` instead of
`pip install -e .` — the plain `pip` binary does not exist inside a uv venv, so
cc_net would silently install elsewhere (or not at all) and you'd hit
`ModuleNotFoundError: No module named 'cc_net'` when starting the validator.

## Running the Validator

Note (from bittensor docs): the validator needs to serve an Axon with their IP or they may be blacklisted by the firewall of serving peers on the network.

If you want to properly serve your Axon you need to change --axon.port from 70000 to a real one.

```bash
pm2 start run.sh --name llm_detection_validators_autoupdate -- --wallet.name YOUR_COLDKEY --wallet.hotkey YOUR_HOTKEY --axon.port 70000 --neuron.device cuda:0
```

