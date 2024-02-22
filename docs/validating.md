# 🧑‍🏫 Validating

## Installation
```bash
git clone https://github.com/sergak0/llm-detection
cd llm-detection
pip install -e . 
```


Creating Coldkey
```bash
btcli w new_coldkey
```
Creating Hotkey
```bash
btcli w new_hotkey
```
Registering your Hotkey
```bash
btcli s register --netuid 87 --wallet.name YOUR_COLDKEY --wallet.hotkey YOUR_HOTKEY
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

In new window do
```
ollama serve
```

Then download models
```
ollama run vicuna
```

```
ollama run mistral
```



## Usage
```bash
python3 neurons/validator.py --netuid 87 --wallet.name YOUR_COLDKEY --wallet.hotkey YOUR_HOTKEY --logging.debug --neuron.device cuda:0 --axon.port 70000
```
