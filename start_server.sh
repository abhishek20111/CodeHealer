#!/bin/bash
# Start web server with proxy disabled for Ollama connection

cd "$(dirname "$0")"

# Unset proxy environment variables
unset HTTP_PROXY
unset HTTPS_PROXY
unset http_proxy
unset https_proxy
unset NO_PROXY
unset no_proxy

# Use venv Python
./venv/bin/python3 web_server.py


# Usage: ./start_server.sh