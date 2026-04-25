#!/bin/bash
# Download an OpenAlex slice via src/data/download.py in the background with
# nohup so it survives shell exit. Output is streamed to a per-dataset log
# file under logs/.
#
# Usage:
#   ./download_openalex.sh                                          # uses defaults below
#   ./download_openalex.sh <name> <filter> <max> <label> [workers]  # override any/all
#
# Examples:
#   ./download_openalex.sh general "has_doi:true" 100000 field
#   ./download_openalex.sh ai "primary_topic.subfield.id:1702" 30000 topic
#   ./download_openalex.sh cs "primary_topic.field.id:17" 50000 subfield
#   ./download_openalex.sh huge "has_doi:true" 1000000 field 4   # parallel: 4 workers
#
# Monitor live:
#   tail -f logs/openalex_<name>.log
#
# Stop a running download:
#   kill $(cat logs/openalex_<name>.pid)

set -e

NAME="${1:-general}"
FILTER="${2:-has_doi:true}"
MAX_WORKS="${3:-100000}"
LABEL_FIELD="${4:-field}"
WORKERS="${5:-1}"
EMAIL="daachirita@gmail.com"

OUT_DIR="data/openalex_${NAME}"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/openalex_${NAME}.log"
PID_FILE="${LOG_DIR}/openalex_${NAME}.pid"

mkdir -p "${LOG_DIR}"

# Refuse to start a second copy on top of a still-running one — that would
# duplicate API calls and confuse the resume cursor.
if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "[run] already running with PID $(cat "${PID_FILE}"). Stop it first or wait."
  echo "[run] kill \$(cat ${PID_FILE})"
  exit 1
fi

echo "[run] dataset=openalex_${NAME} filter='${FILTER}' max=${MAX_WORKS} label=${LABEL_FIELD} workers=${WORKERS}"
echo "[run] out_dir=${OUT_DIR}"
echo "[run] log_file=${LOG_FILE}"

# python -u + PYTHONUNBUFFERED keep stdout flushing live so `tail -f` shows
# progress in real time. nohup detaches from the controlling terminal so the
# process keeps running after the shell closes.
nohup env PYTHONUNBUFFERED=1 python -u src/data/download.py \
  --dataset openalex \
  --out_dir "${OUT_DIR}" \
  --config_template configs/config.yaml \
  --oa_filter "${FILTER}" \
  --oa_max_works "${MAX_WORKS}" \
  --oa_label_field "${LABEL_FIELD}" \
  --oa_workers "${WORKERS}" \
  --oa_email "${EMAIL}" \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${PID_FILE}"

echo "[run] started PID=${PID} (detached, survives shell exit)"
echo "[run] follow:  tail -f ${LOG_FILE}"
echo "[run] stop:    kill \$(cat ${PID_FILE})"
