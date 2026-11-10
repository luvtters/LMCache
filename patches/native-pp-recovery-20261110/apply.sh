#!/usr/bin/env bash
set -euo pipefail
source /opt/ks-luv-pp/env.sh
patch_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
exec python3 "$patch_dir/patchctl.py" apply
