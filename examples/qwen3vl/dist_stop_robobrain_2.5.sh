#!/bin/bash
set -euo pipefail

# Stop stale Qwen3-VL RoboBrain training processes on every node in hostfile.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HOSTFILE=${HOSTFILE:-${SCRIPT_DIR}/hostfile}
SSH_CMD=${SSH_CMD:-ssh}
SSH_OPTS=${SSH_OPTS:-}
DRY_RUN=${DRY_RUN:-0}

HOSTFILE=$(cd "$(dirname "${HOSTFILE}")" && pwd)/$(basename "${HOSTFILE}")

if [[ ! -f "${HOSTFILE}" ]]; then
    echo "Hostfile not found: ${HOSTFILE}" >&2
    exit 1
fi

mapfile -t HOSTS < <(awk '$0 !~ /^[[:space:]]*(#|$)/ {print $1}' "${HOSTFILE}")
if [[ "${#HOSTS[@]}" -eq 0 ]]; then
    echo "No hosts found in ${HOSTFILE}" >&2
    exit 1
fi

# Use bracketed patterns so pkill does not match its own command line.
KILL_PATTERNS=(
    "[t]rain_qwen3_vl.py"
    "[r]un_robobrain_2.5.sh"
    "[t]orchrun"
)

if [[ -n "${EXTRA_KILL_PATTERNS:-}" ]]; then
    read -r -a EXTRA_PATTERNS <<< "${EXTRA_KILL_PATTERNS}"
    KILL_PATTERNS+=("${EXTRA_PATTERNS[@]}")
fi

build_remote_cmd() {
    local cmd="set +e"
    local pattern
    for pattern in "${KILL_PATTERNS[@]}"; do
        printf -v cmd '%s; pkill -f %q' "${cmd}" "${pattern}"
    done
    printf -v cmd '%s; exit 0' "${cmd}"
    printf '%s' "${cmd}"
}

REMOTE_CMD=$(build_remote_cmd)

echo "=========================================="
echo "Qwen3-VL RoboBrain stop script"
echo "Hostfile: ${HOSTFILE}"
echo "Patterns:"
printf '  %s\n' "${KILL_PATTERNS[@]}"
echo "=========================================="

for host in "${HOSTS[@]}"; do
    echo "Stopping training processes on ${host}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY RUN: '
        # shellcheck disable=SC2086
        printf '%q ' "${SSH_CMD}" ${SSH_OPTS} "${host}" "${REMOTE_CMD}"
        printf '\n'
    else
        # shellcheck disable=SC2086
        if ! "${SSH_CMD}" ${SSH_OPTS} "${host}" "${REMOTE_CMD}"; then
            echo "WARNING: stop command on ${host} returned non-zero; continuing." >&2
        fi
    fi
done

echo "Stop requests submitted."
