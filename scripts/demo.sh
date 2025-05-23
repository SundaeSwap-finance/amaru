#!/usr/bin/env bash

exitWithUsage () {
  echo -e "\033[1;31mError: missing argument(s)!\033[00m\n"
  echo -e "\033[1;32mUsage:\033[00m\n    $0 PEER_ADDRESS TARGET_EPOCH [NETWORK]\n"
  echo -e "\033[1mExamples:\033[00m \n    $0 127.0.0.1:3000 173\n    $0 127.0.0.1:3001 200 preprod"
  exit 1
}

PEER_ADDRESS=$1
if [ -z "$PEER_ADDRESS" ]; then
  exitWithUsage
fi

TARGET_EPOCH=$2
if [ -z "$TARGET_EPOCH" ]; then
  exitWithUsage
fi

NETWORK=${3:-preprod}

echo -e "      \033[1;32mTarget\033[00m epoch $TARGET_EPOCH"
set -eo pipefail
AMARU_TRACE="amaru=debug" cargo run -- --with-json-traces daemon --peer-address=$PEER_ADDRESS --network=$NETWORK | while read line; do
  EVENT=$(echo $line | jq -r '.fields.message' 2>/dev/null)
  SPAN=$(echo $line | jq -r '.span.name' 2>/dev/null)
  if [ "$EVENT" == "exit" ] && [ "$SPAN" == "snapshot" ]; then
    EPOCH=$(echo $line | jq -r '.span.epoch' 2>/dev/null)
    if [ "$EPOCH" == "$TARGET_EPOCH" ]; then
      echo "Target epoch reached, stopping the process."
      pkill -INT -P $$
      break
    fi
  fi
done
