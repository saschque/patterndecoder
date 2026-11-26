#!/usr/bin/env bash
# Applies the ticker and the config overrides of a manually started run to
# config/config.yaml. TICKER is a Yahoo Finance ticker or a synthetic series
# (SYNTH-AR, SYNTH-MOTIF, SYNTH-CYCLE); OVERRIDES is a JSON object of config
# keys and scalar values, for example {"input_scale": 100}.
set -euo pipefail
config=config/config.yaml

ticker_pattern='^\^?[A-Za-z0-9.=-]+$'
if ! [[ "$TICKER" =~ $ticker_pattern ]]; then
  echo "::error::invalid ticker '$TICKER'"; exit 1
fi
sed -i -E "s/^ticker:.*/ticker: \"${TICKER}\"/" "$config"
# the checksum describes the DAX data of the published results
if [ "$TICKER" != "^GDAXI" ]; then sed -i '/^data_checksum:/d' "$config"; fi

if ! echo "${OVERRIDES:-{\}}" | jq -e 'type == "object"' > /dev/null; then
  echo "::error::overrides must be a JSON object"; exit 1
fi
key_pattern='^[a-z0-9_]+$'
value_pattern='^[A-Za-z0-9._+"-]+$'
for key in $(echo "${OVERRIDES:-{\}}" | jq -r 'keys[]'); do
  value=$(echo "$OVERRIDES" | jq -c --arg key "$key" '.[$key]')
  if ! [[ "$key" =~ $key_pattern && "$value" =~ $value_pattern ]]; then
    echo "::error::invalid override $key=$value"; exit 1
  fi
  if grep -q "^${key}:" "$config"; then
    sed -i -E "s/^${key}:.*/${key}: ${value}/" "$config"
  else
    echo "${key}: ${value}" >> "$config"
  fi
done
grep -E "^(ticker|data_checksum)$(echo "${OVERRIDES:-{\}}" | jq -r 'keys[] | "|" + .' | tr -d '\n'):" "$config" || true
