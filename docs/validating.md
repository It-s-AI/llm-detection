# 🧑‍🏫 Validating

# System Requirements

Validators will need enough processing power to inference multiple models. It is required to have a GPU (atleast NVIDIA RTX 4090) with minimum 16 GB of VRAM (24 GB is recommended). 

## Installation

Make sure that your server provider support systemd (RunPod doesn't support it).
Otherwise ollama service won't be restarting automatically and you'll have to restart it on your own from time to time.

1. Clone the repo

```bash
git clone https://github.com/It-s-AI/llm-detection
```  

2. Setup your python [virtual environment](https://docs.python.org/3/library/venv.html) or [Conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).

3. Install the requirements. From your virtual environment, run
```shell
cd llm-detection
python3 -m pip install -e .
python3 -m pip uninstall mathgenerator -y
python3 -m pip install git+https://github.com/synapse-alpha/mathgenerator.git
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
pm2 start --name ollama "ollama serve"
```

Then download models
```
ollama pull vicuna && ollama pull mistral && ollama pull neural-chat && ollama pull gemma:7b && ollama pull zephyr:7b-beta
```


## Running the Validator

```bash
pm2 start --name net32-vali --interpreter python3 ./neurons/validator.py -- --wallet.name YOUR_COLDKEY --wallet.hotkey YOUR_HOTKEY --neuron.device cuda:0 --axon.port 70000
```