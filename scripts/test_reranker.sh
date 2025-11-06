#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Default to all 4 GPUs unless user explicitly sets it
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

TEST_FILE="${TEST_FILE:-/dataset/evqa/test.csv}"
KB_JSON="${KB_JSON:-/dataset/evqa/encyclopedic_kb_wiki.json}"
FAISS_INDEX_DIR="${FAISS_INDEX_DIR:-/dataset/evqa/}"
QFORMER_CKPT="${QFORMER_CKPT:-${PROJECT_ROOT}/reranker.pth}"
NLI_CONFIG="${NLI_CONFIG:-${PROJECT_ROOT}/config/nli.example.yaml}"
DATASET_START="${DATASET_START:-}"
DATASET_END="${DATASET_END:-}"
DATASET_LIMIT="${DATASET_LIMIT:-}"
RETRIEVER_VIT="${RETRIEVER_VIT:-eva-clip}"
NLI_SECTION_LIMIT="${NLI_SECTION_LIMIT:-10}"
NLI_CONTEXT_SENTENCES="${NLI_CONTEXT_SENTENCES:-5}"
PERFORM_VQA="${PERFORM_VQA:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Router controls (optional)
DEFAULT_ROUTER_CONFIG="${PROJECT_ROOT}/config/router.yaml"
if [[ ! -f "${DEFAULT_ROUTER_CONFIG}" ]]; then
  DEFAULT_ROUTER_CONFIG="${PROJECT_ROOT}/config/router.example.yaml"
fi

ROUTER_ENABLE="${ROUTER_ENABLE:-}"
ROUTER_DISABLE="${ROUTER_DISABLE:-}"
ROUTER_CONFIG="${ROUTER_CONFIG:-${DEFAULT_ROUTER_CONFIG}}"
ROUTER_BACKEND="${ROUTER_BACKEND:-}"
ROUTER_THRESHOLD="${ROUTER_THRESHOLD:-}"

CMD=(
  python3 "${PROJECT_ROOT}/test/test_reranker.py"
  --test_file "${TEST_FILE}"
  --knowledge_base "${KB_JSON}"
  --faiss_index "${FAISS_INDEX_DIR}"
  --retriever_vit "${RETRIEVER_VIT}"
  --top_ks 1,5,10,20
  --retrieval_top_k 20
  --perform_qformer_reranker
  --qformer_ckpt_path "${QFORMER_CKPT}"
  --enable_nli
  --nli_config "${NLI_CONFIG}"
  --nli_section_limit "${NLI_SECTION_LIMIT}"
  --nli_context_sentences "${NLI_CONTEXT_SENTENCES}"
)

# Router options
if [[ -n "${ROUTER_CONFIG}" && -f "${ROUTER_CONFIG}" ]]; then
  CMD+=(--router_config "${ROUTER_CONFIG}")
fi
if [[ -n "${ROUTER_ENABLE}" ]]; then
  CMD+=(--enable_router)
fi
if [[ -n "${ROUTER_DISABLE}" ]]; then
  CMD+=(--disable_router)
fi
if [[ -n "${ROUTER_BACKEND}" ]]; then
  CMD+=(--router_backend "${ROUTER_BACKEND}")
fi
if [[ -n "${ROUTER_THRESHOLD}" ]]; then
  CMD+=(--router_threshold "${ROUTER_THRESHOLD}")
fi

if [[ -n "${DATASET_START}" ]]; then
  CMD+=(--dataset_start "${DATASET_START}")
fi
if [[ -n "${DATASET_END}" ]]; then
  CMD+=(--dataset_end "${DATASET_END}")
fi
if [[ -n "${DATASET_LIMIT}" ]]; then
  CMD+=(--dataset_limit "${DATASET_LIMIT}")
fi
if [[ -n "${PERFORM_VQA}" ]]; then
  CMD+=(--perform_vqa)
fi
# Allow arbitrary passthrough flags via EXTRA_ARGS
if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  CMD+=( ${EXTRA_ARGS} )
fi

echo "[Router] Script config: config='${ROUTER_CONFIG:-unset}' enable_flag='${ROUTER_ENABLE:-auto}' disable_flag='${ROUTER_DISABLE:-auto}' backend_override='${ROUTER_BACKEND:-none}' threshold_override='${ROUTER_THRESHOLD:-none}'"

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" "${CMD[@]}"
