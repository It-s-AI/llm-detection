# 🧑‍🏫 Validating

## Installation
```bash
git clone https://github.com/It-s-AI/llm-detection
cd llm-detection
python3 -m pip install -e . 
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
btcli s register --netuid 32 --wallet.name YOUR_COLDKEY --wallet.hotkey YOUR_HOTKEY
```

## Install driver

So Ollama models can detect GPUs on your system
```bash
apt update
apt install lshw
```

## Setup Ollama

Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Update ollama systemd service configuration

```bash
sh configure_ollama.sh
```

Start ollama service
```bash
sudo systemctl daemon-reload
sudo systemctl enable ollama
```

To view logs of Ollama running as a startup service, run:
```bash
journalctl -u ollama -n 100
```

Then download models
```bash
ollama pull vicuna

ollama pull mistral
```



## Usage
```bash
python3 neurons/validator.py --netuid 32 --wallet.name YOUR_COLDKEY --wallet.hotkey YOUR_HOTKEY --logging.debug --neuron.device cuda:0 --axon.port 70000
```
