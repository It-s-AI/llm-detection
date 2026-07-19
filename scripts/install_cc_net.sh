#!/bin/bash
# Sets up cc_net (validator-only). Run this after `pip install -e .`.
# Kept out of setup.py so package installs don't compile/download at build time.
set -e

cd "$(dirname "$0")/../cc_net"

make install
pip uninstall -y cc_net || true
pip install -e .
make lang=en dl_lm
