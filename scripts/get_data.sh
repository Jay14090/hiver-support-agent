#!/usr/bin/env bash
# Fetch twcs.csv. Tries, in order:
#   1. Kaggle CLI (the canonical source named in the brief)
#   2. A HuggingFace mirror of the identical file (no credentials needed)
#   3. Any local copy already on disk
#   4. A clearly-labelled SYNTHETIC stand-in so the pipeline stays runnable
set -uo pipefail
cd "$(dirname "$0")/.."
DEST="data/raw/twcs.csv"
mkdir -p data/raw

if [ -s "$DEST" ]; then
  echo "[get_data] $DEST already present ($(wc -c <"$DEST") bytes) -- nothing to do."
  exit 0
fi

# ---------------------------------------------------------------- 1. Kaggle
echo "[get_data] attempt 1/4: Kaggle CLI"
if command -v kaggle >/dev/null 2>&1; then
  if kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw --unzip 2>/dev/null; then
    [ -s "$DEST" ] && { echo "[get_data] OK via Kaggle."; exit 0; }
  fi
fi
cat <<'MSG'
[get_data] Kaggle unavailable or unauthenticated. To use the canonical source:
    1. Create an API token at https://www.kaggle.com/settings  ->  "Create New Token"
    2. Save the downloaded kaggle.json to  ~/.kaggle/kaggle.json   (Windows: %USERPROFILE%\.kaggle\kaggle.json)
    3. chmod 600 ~/.kaggle/kaggle.json
    4. pip install kaggle && re-run this script
  Falling back to a public mirror of the identical file.
MSG

# ---------------------------------------------------------------- 2. HF mirror
echo "[get_data] attempt 2/4: HuggingFace mirror (SunidhiSriram/twcs)"
URL="https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv"
if command -v curl >/dev/null 2>&1; then
  curl -sSL --retry 3 --fail -o "$DEST" "$URL" && [ -s "$DEST" ] && { echo "[get_data] OK via HF mirror."; exit 0; }
elif command -v wget >/dev/null 2>&1; then
  wget -q -O "$DEST" "$URL" && [ -s "$DEST" ] && { echo "[get_data] OK via HF mirror."; exit 0; }
fi
rm -f "$DEST"

# ---------------------------------------------------------------- 3. local copy
echo "[get_data] attempt 3/4: searching the machine for an existing twcs.csv"
FOUND=""
for d in "$HOME/Downloads" "$HOME" "./data"; do
  [ -d "$d" ] || continue
  F=$(find "$d" -maxdepth 3 -iname "twcs.csv" -size +100M 2>/dev/null | head -1)
  [ -n "$F" ] && { FOUND="$F"; break; }
done
if [ -n "$FOUND" ]; then
  cp "$FOUND" "$DEST" && echo "[get_data] OK -- copied from $FOUND"; exit 0
fi

# ---------------------------------------------------------------- 4. synthetic
echo "[get_data] attempt 4/4: all real sources failed -> generating SYNTHETIC stand-in."
echo "[get_data] !! Any results produced from this file are NOT REAL RESULTS. !!"
python scripts/make_synthetic.py
exit 0
