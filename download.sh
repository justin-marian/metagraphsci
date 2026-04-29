#!/usr/bin/env bash

set -euo pipefail

# Resolve the repository root whether the script is run from the project root
# or from a dedicated scripts/ directory.
resolve_project_root() {
  local script_dir parent_dir

  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  parent_dir="$(cd "$script_dir/.." && pwd)"

  if [ -f "$script_dir/src/data/download.py" ]; then
    printf '%s\n' "$script_dir"
    return 0
  fi

  if [ -f "$parent_dir/src/data/download.py" ]; then
    printf '%s\n' "$parent_dir"
    return 0
  fi

  printf '✗ Could not find src/data/download.py from %s\n' "$script_dir" >&2
  return 1
}

# Keep terminal formatting helpers centralized so status output stays uniform.
bold() {
  printf '\033[1m%s\033[0m\n' "$*"
}

green() {
  printf '\033[1;32m✓ %s\033[0m\n' "$*"
}

blue() {
  printf '\033[1;34m→ %s\033[0m\n' "$*"
}

red() {
  printf '\033[1;31m✗ %s\033[0m\n' "$*"
}

# Render a value or a human-readable fallback for optional environment settings.
display_value() {
  local value fallback

  value="$1"
  fallback="$2"

  if [ -n "$value" ]; then
    printf '%s\n' "$value"
    return
  fi
  printf '%s\n' "$fallback"
}

# Render a simple yes/no flag based on whether a value is populated.
enabled_flag() {
  if [ -n "$1" ]; then
    printf 'yes\n'
    return
  fi
  printf 'no\n'
}

# Validate project layout and required Python packages before starting work.
run_preflight_checks() {
  if [ ! -f "$SRC" ]; then
    red "Could not find $SRC"
    exit 1
  fi

  if ! python -c "import polars, requests" 2>/dev/null; then
    red "Required packages missing. Run: pip install polars requests"
    exit 1
  fi
}

# Initialize the argument array for one download invocation.
init_oa_args() {
  local domain out_dir

  domain="$1"
  out_dir="$2"

  OA_ARGS=(
    --domain "$domain"
    --out_dir "$out_dir"
    --config_template "$CFG"
    --max_papers "$MAX_PAPERS"
    --from_year "$FROM_YEAR"
    --to_year "$TO_YEAR"
  )
}

# Append optional flags only when they are explicitly configured.
append_optional_oa_args() {
  if [ -n "$PAPERS_PER_CLASS" ]; then
    OA_ARGS+=(--papers_per_class "$PAPERS_PER_CLASS")
  fi

  if [ -n "$OA_CLASSES" ]; then
    # Intentional word splitting: OA_CLASSES is expected to contain a shell-like
    # whitespace-separated list of class names.
    # shellcheck disable=SC2206
    local class_args=($OA_CLASSES)
    OA_ARGS+=(--classes "${class_args[@]}")
  fi

  if [ -n "$OA_QUERY" ]; then
    OA_ARGS+=(--query "$OA_QUERY")
  fi

  if [ -n "$LLM_API_KEY" ]; then
    OA_ARGS+=(--llm_fallback --llm_api_key "$LLM_API_KEY")
  fi

  if [ -n "$NO_CITATIONS" ]; then
    OA_ARGS+=(--no_citations)
  fi
}

# Build the final downloader arguments for one domain.
build_oa_args() {
  init_oa_args "$1" "$2"
  append_optional_oa_args
}

# Print the shared configuration once so logs capture the exact run settings.
print_header() {
  bold "MetaGraphSci — OpenAlex multi-domain dataset downloader"
  echo
  echo "Global configuration:"
  echo "  base_out_dir     = $BASE_OUT_DIR"
  echo "  max_papers       = $MAX_PAPERS"
  echo "  year window      = $FROM_YEAR – $TO_YEAR"
  echo "  papers_per_class = $(display_value "$PAPERS_PER_CLASS" "(auto)")"
  echo "  classes          = $(display_value "$OA_CLASSES" "(all per domain)")"
  echo "  query prefix     = $(display_value "$OA_QUERY" "(none)")"
  echo "  llm_fallback     = $(enabled_flag "$LLM_API_KEY")"
  echo "  citations        = $([ -n "$NO_CITATIONS" ] && echo no || echo yes)"
  echo
}

# Print per-domain settings so each invocation is easy to audit in logs.
print_domain_header() {
  local domain out_dir

  domain="$1"
  out_dir="$2"

  blue "Building domain: $domain"
  echo "  out_dir          = $out_dir"
  echo "  max_papers       = $MAX_PAPERS"
  echo "  year window      = $FROM_YEAR - $TO_YEAR"
  echo "  papers_per_class = $(display_value "$PAPERS_PER_CLASS" "(auto)")"
  echo "  classes          = $(display_value "$OA_CLASSES" "(all for $domain)")"
  echo "  query prefix     = $(display_value "$OA_QUERY" "(none)")"
  echo
}

# Run one domain build end to end while keeping the loop body simple.
build_domain() {
  local domain out_dir

  domain="$1"
  out_dir="$BASE_OUT_DIR/$domain"

  build_oa_args "$domain" "$out_dir"
  print_domain_header "$domain" "$out_dir"
  python "$SRC" "${OA_ARGS[@]}"
  green "$domain done  →  $out_dir"
  echo
}

# Print the final directory layout so downstream consumers know what to expect.
print_summary() {
  local domain

  bold "All domain datasets downloaded successfully."
  echo
  for domain in "${DOMAINS[@]}"; do
    echo "  $BASE_OUT_DIR/$domain"
  done
  echo
  echo "Each directory contains:"
  echo "  documents.csv   — papers with title, abstract, venue, year, label"
  echo "  citations.csv   — directed citation edges (source → target)"
  echo "  config.yaml     — pipeline config pre-filled for this benchmark"
  echo
}

PROJECT_ROOT="$(resolve_project_root)"
cd "$PROJECT_ROOT"

readonly SRC="src/data/download.py"
readonly CFG="configs/config.yaml"

MAX_PAPERS="${MAX_PAPERS:-24000}"
FROM_YEAR="${FROM_YEAR:-2015}"
TO_YEAR="${TO_YEAR:-2024}"
BASE_OUT_DIR="${BASE_OUT_DIR:-data}"
PAPERS_PER_CLASS="${PAPERS_PER_CLASS:-2500}"
OA_CLASSES="${OA_CLASSES:-}"
OA_QUERY="${OA_QUERY:-}"
LLM_API_KEY="${LLM_API_KEY:-}"
NO_CITATIONS="${NO_CITATIONS:-}"

DOMAINS=(
  # "cs_ai"
  # "medicine"
  # "physics"
  "economics"
  "social_science"
  "law"
)

run_preflight_checks
print_header

for domain in "${DOMAINS[@]}"; do
  build_domain "$domain"
done

print_summary
