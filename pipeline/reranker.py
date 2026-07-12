"""
Qwen3-Reranker-0.6B reranker (4-bit quantized, generative LLM-based).

Unlike cross-encoder rerankers, Qwen3-Reranker is a causal LM that scores
relevance by comparing the logits of the "yes" vs "no" tokens at the final
position of a structured prompt. Score = softmax(yes_logit, no_logit)[yes].
"""
import logging
import gc
from typing import List, Dict, Any

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import config

logger = logging.getLogger(__name__)

RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"

# Prompt template from Qwen3-Reranker official docs
_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and the "
    "Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
)
_INSTRUCTION = "Given a document retrieval query, retrieve the most relevant passages that answer the query."

_model = None
_tokenizer = None
_token_true_id = None   # token id for "yes"
_token_false_id = None  # token id for "no"
_prefix_ids = None      # cached token ids for the assistant prefix "<|im_start|>assistant\n"


def _build_prompt(query: str, doc: str) -> str:
    return (
        f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"<Instruct>: {_INSTRUCTION}\n"
        f"<Query>: {query}\n"
        f"<Document>: {doc}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _load():
    global _model, _tokenizer, _token_true_id, _token_false_id, _prefix_ids

    if _model is not None:
        return

    logger.info(f"Loading Qwen3-Reranker: {RERANKER_MODEL} (4-bit)")

    _tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL, padding_side="left")

    # Resolve yes/no token ids once
    _token_true_id = _tokenizer.convert_tokens_to_ids("yes")
    _token_false_id = _tokenizer.convert_tokens_to_ids("no")
    logger.info(f"yes token id={_token_true_id}, no token id={_token_false_id}")

    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        device_map = {"": 0} if torch.cuda.is_available() else "cpu"
        _model = AutoModelForCausalLM.from_pretrained(
            RERANKER_MODEL,
            quantization_config=bnb_config,
            device_map=device_map,
            attn_implementation="sdpa",
        )
        _model.eval()
        logger.info("Qwen3-Reranker loaded on GPU (4-bit).")
    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA OOM loading reranker. Falling back to CPU fp32.")
        torch.cuda.empty_cache()
        gc.collect()
        _model = AutoModelForCausalLM.from_pretrained(
            RERANKER_MODEL,
            device_map="cpu",
        )
        _model.eval()
    except Exception as e:
        logger.error(f"Failed to load Qwen3-Reranker: {e}")
        raise


def unload():
    global _model, _tokenizer, _token_true_id, _token_false_id
    if _model is not None:
        del _model
        _model = None
    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None
    _token_true_id = None
    _token_false_id = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("Unloaded Qwen3-Reranker.")


@torch.no_grad()
def _score_batch(prompts: List[str]) -> List[float]:
    """Tokenize a batch of prompts and return yes-probability for each."""
    inputs = _tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=4096,
        return_tensors="pt",
    )
    # Move to the same device as the model
    device = next(_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = _model(**inputs)
    # Last token logits: shape (batch, vocab)
    last_logits = outputs.logits[:, -1, :]

    yes_logits = last_logits[:, _token_true_id]
    no_logits = last_logits[:, _token_false_id]

    # Softmax over {no, yes}
    stacked = torch.stack([no_logits, yes_logits], dim=1)
    probs = torch.softmax(stacked.float(), dim=1)
    return probs[:, 1].tolist()  # probability of "yes"


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_n: int = 10,
    batch_size: int = 4,
) -> List[Dict[str, Any]]:
    """
    Rerank candidates using Qwen3-Reranker-0.6B.

    Scores each (query, passage) pair by P("yes" | prompt).
    Returns top_n candidates sorted by rerank_score descending.
    """
    if not candidates:
        return []

    if _model is None:
        _load()

    prompts = [
        _build_prompt(query, c.get("raw_content") or c.get("content", ""))
        for c in candidates
    ]

    scores: List[float] = []
    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i: i + batch_size]
            batch_scores = _score_batch(batch)
            scores.extend(batch_scores)
    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA OOM during reranking — retrying with batch_size=1 on CPU.")
        torch.cuda.empty_cache()
        gc.collect()
        _model.to("cpu")
        scores = []
        for p in prompts:
            scores.extend(_score_batch([p]))
    except Exception as e:
        logger.error(f"Reranker inference failed: {e}. Returning unranked candidates.")
        return candidates[:top_n]

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = score

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    top = reranked[:top_n]
    logger.info(
        f"Reranked {len(candidates)} → top {top_n} | "
        f"scores: {[f'{r["rerank_score"]:.3f}' for r in top]}"
    )
    return top
