#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/xrd-benchmark}"
MAMBA_EXE="${MAMBA_EXE:-$HOME/.local/bin/micromamba}"

if [[ ! -x "$MAMBA_EXE" ]]; then
  mkdir -p "$(dirname "$MAMBA_EXE")"
  if [[ -n "${MICROMAMBA_ARCHIVE:-}" ]]; then
    tar -xjf "$MICROMAMBA_ARCHIVE" -C "$(dirname "$MAMBA_EXE")" \
      --strip-components=1 bin/micromamba
  else
    curl -Ls --fail --retry 3 --connect-timeout 15 --max-time 120 \
      https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xvj -C "$(dirname "$MAMBA_EXE")" --strip-components=1 bin/micromamba
  fi
fi

cd "$PROJECT_ROOT"
"$MAMBA_EXE" create -y -f environment-workstation.yml
"$MAMBA_EXE" run -n sitexrd python -m pip install --no-deps -e .
"$MAMBA_EXE" run -n sitexrd python -m unittest discover -s tests -p 'test_occupancy_xrd_cli.py'
