# 🧑‍🏫 Validating

# System Requirements

Validators will need enough processing power to inference multiple models. It is required to have a GPU (atleast NVIDIA RTX 4090) with minimum 16 GB of VRAM (24 GB is recommended). 

## Installation

1. Clone the repo

```bash
git clone https://github.com/It-s-AI/llm-detection
```  

2. Setup your python [virtual environment](https://docs.python.org/3/library/venv.html) or [Conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).

3. Install the requirements. From your virtual environment, run
```shell
cd llm-detection
python -m pip install -e .
```

4. Make sure you've [created a Wallet](https://docs.bittensor.com/getting-started/wallets) and [registered a hotkey](https://docs.bittensor.com/subnets/register-and-participate).

```bash
btcli w new_coldkey
btcli w new_hotkey
btcli s register --netuid 32 --wallet.name YOUR_COLDKEY --wallet.hotkey YOUR_HOTKEY
```

## Install driver

So Ollama models can detect GPUs on your system
```bash
apt update
apt install lshw
```

## Download models

Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Run ollama service in background
```
nohup ollama serve&
```

Then download models
```
ollama pull vicuna
ollama pull mistral
```


## Running the Validator
## With auto-updates

We highly recommend running the validator with auto-updates. This will help ensure your validator is always running the latest release, helping to maintain a high vtrust.

Prerequisites:
1. To run with auto-update, you will need to have [pm2](https://pm2.keymetrics.io/) installed.
2. Make sure your virtual environment is activated. This is important because the auto-updater will automatically update the package dependencies with pip.
3. Make sure you're using the main branch: `git checkout main`.


```bash
pm2 start --name net32-vali-updater --interpreter python3 scripts/start_validator.py -- --pm2_name net32-vali --wallet.name YOUR_COLDKEY --wallet.hotkey YOUR_HOTKEY --neuron.device cuda:0 --axon.port 70000
```

This will start a process called `net32-vali-updater`. This process periodically checks for a new git commit on the current branch. When one is found, it performs a `pip install` for the latest packages, and restarts the validator process (who's name is given by the `--pm2_name` flag)

## Without auto-updates

If you'd prefer to manage your own validator updates...

```bash
pm2 start --name net32-vali --interpreter python3 ./neurons/validator.py -- --wallet.name YOUR_COLDKEY --wallet.hotkey YOUR_HOTKEY --neuron.device cuda:0 --axon.port 70000
```