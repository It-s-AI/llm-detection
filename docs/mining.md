# ⛏️ Mining Docs

## Installation
```bash
git clone https://github.com/It-s-AI/llm-detection
cd llm-detection
python3 -m pip install -e . 
```

## Usage

To start mining on our subnetwork you need to create your coldkey, hotkey, and register it on netuid *32*.

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
Now you are ready to start mining!
```bash
python3 neurons/miner.py --netuid 32 --wallet.name YOUR_COLDKEY --wallet.hotkey YOUR_HOTKEY --logging.debug --neuron.device cuda:0 --axon.port 70000
```
