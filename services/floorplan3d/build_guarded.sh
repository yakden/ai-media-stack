#!/usr/bin/env bash
set -uo pipefail
cd /home/deploy/FloorPlanTo3D-API
sudo docker compose -f docker-compose.yml build > build.log 2>&1 &
BPID=$!
while kill -0 $BPID 2>/dev/null; do
  free=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  if [ "$free" -lt 8 ]; then
    echo "DISK GUARD: only ${free}G free — killing build to protect 1C" | tee -a build.log
    sudo pkill -f 'compose.*floorplan|buildkit' 2>/dev/null
    kill $BPID 2>/dev/null
    exit 42
  fi
  sleep 5
done
wait $BPID; rc=$?
echo "BUILD EXIT $rc (free $(df -BG --output=avail / | tail -1 | tr -d ' ') )" | tee -a build.log
exit $rc
