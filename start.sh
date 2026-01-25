#!/bin/zsh

# Start Colima
devbox run colima:start

# Start n8n
devbox run n8n:up

# Start Brain API
devbox run brain:setup
devbox run brain:run