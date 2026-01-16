#!/usr/bin/env bash
set -euo pipefail

SERVER="root@77.42.38.157"
REMOTE_DIR="/root/rag-mvp"

echo "Uploading app/ ..."
rsync -avz --delete ./app/ "$SERVER:$REMOTE_DIR/app/"

echo "Uploading static/ ..."
rsync -avz --delete ./static/ "$SERVER:$REMOTE_DIR/static/"

echo "Uploading connectors/ ..."
rsync -avz --delete ./connectors/ "$SERVER:$REMOTE_DIR/connectors/"

echo "Uploading ingest/ ..."
rsync -avz --delete ./ingest/ "$SERVER:$REMOTE_DIR/ingest/"

echo "Uploading root files ..."
rsync -avz ./Dockerfile ./compose.yaml ./requirements.txt "$SERVER:$REMOTE_DIR/" 2>/dev/null || true

echo "Restarting api ..."
ssh "$SERVER" "cd $REMOTE_DIR && docker compose -f compose.yaml restart api"

echo "Done. Open https://ravelous.cloud/ and hard refresh."
