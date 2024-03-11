status() { echo ">>> $*" >&2; }

for BINDIR in /usr/local/bin /usr/bin /bin; do
    echo $PATH | grep -q $BINDIR && break || continue
done

status "Using $BINDIR as bin path"

cat <<EOF | $SUDO tee /etc/systemd/system/ollama.service >/dev/null
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=$BINDIR/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
RuntimeMaxSec=60m
Environment="PATH=$PATH"

[Install]
WantedBy=default.target
EOF

status "Successfully updated ollama service"
