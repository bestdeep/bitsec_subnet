#!/bin/bash

# Default to production environment
ENV="prod"
NETUID=60
NETWORK="finney"
PORT=9099  # Default port
PROXY_PORT=10922 # Used on DigitalOcean
COMMAND_WITH_PATH="python3"
WALLET_PREFIX=""

for arg in "$@"; do
  if [ "$arg" = "--test" ] || [ "$arg" = "--testnet" ]; then
    ENV="test"
    NETUID=350
    NETWORK="test"
    WALLET_PREFIX="testnet_"
  fi
done

#if proxy.port in args anywhere, use it
if [[ "$@" == *"--proxy.port"* ]]; then
    PROXY_PORT=$(echo "$@" | grep -o -- "--proxy.port [0-9]*" | grep -o "[0-9]*")
fi

# Activate virtual environment if it exists
if [[ -d "venv" && -f "venv/bin/activate" ]]; then
    echo "Activating virtual environment"
    source venv/bin/activate
    COMMAND_WITH_PATH="venv/bin/python3"
fi

echo "Starting miner in $ENV environment with netuid $NETUID on port $PORT"
$COMMAND_WITH_PATH -m neurons.miner --netuid $NETUID \
    --subtensor.chain_endpoint $NETWORK --subtensor.network $NETWORK \
    --wallet.name "king1" --wallet.hotkey "bothot9" \
    --axon.port $PORT --axon.external_port $PORT \
    --logging.debug --proxy.port $PROXY_PORT
