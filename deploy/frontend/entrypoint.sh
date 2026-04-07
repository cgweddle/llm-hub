#!/bin/sh
set -e

echo "Starting frontend server..."
exec node build/index.js
