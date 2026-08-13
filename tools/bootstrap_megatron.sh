#!/usr/bin/env bash
# Materialize an immutable Megatron-LM checkout. Existing checkouts are verified,
# never reset or cleaned implicitly.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 PROFILE_JSON DESTINATION" >&2
  exit 2
fi
PROFILE_JSON=$1
DESTINATION=$2
PYTHON_BIN=${PYTHON_BIN:-python3}

readarray -t UPSTREAM < <("$PYTHON_BIN" - "$PROFILE_JSON" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    cfg = json.load(handle)
print(cfg["upstream"]["repository"])
print(cfg["upstream"]["commit"])
PY
)
REPOSITORY=${UPSTREAM[0]}
COMMIT=${UPSTREAM[1]}

if [[ -e "$DESTINATION" && ! -d "$DESTINATION/.git" ]]; then
  echo "destination exists but is not a git checkout: $DESTINATION" >&2
  exit 1
fi
if [[ ! -d "$DESTINATION/.git" ]]; then
  mkdir -p "$DESTINATION"
  git -C "$DESTINATION" init
  git -C "$DESTINATION" remote add origin "$REPOSITORY"
  git -C "$DESTINATION" fetch --depth=1 origin "$COMMIT"
  git -C "$DESTINATION" checkout --detach FETCH_HEAD
fi

ACTUAL=$(git -C "$DESTINATION" rev-parse HEAD)
if [[ "$ACTUAL" != "$COMMIT" ]]; then
  echo "existing checkout is at $ACTUAL, expected $COMMIT; refusing to mutate it" >&2
  exit 1
fi
if [[ -n $(git -C "$DESTINATION" status --porcelain --untracked-files=no) ]]; then
  echo "existing checkout has tracked modifications; refusing to use it" >&2
  exit 1
fi
printf '{"marker":"MEGATRON_BOOTSTRAP_OK","commit":"%s","path":"%s"}\n' "$ACTUAL" "$DESTINATION"
