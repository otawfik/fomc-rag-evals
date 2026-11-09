"""Turning retrieved chunks into an answer.

Two answerers with one interface (`answer(question, hits) -> Answer`):
  ExtractiveReader  Hugging Face Transformers question-answering model (DistilBERT fine-tuned on SQuAD).
                    Free, local and deterministic: the default, and what CI evaluates on every push.
  BedrockAnswerer   An LLM on Amazon Bedrock through the Converse API, told to answer only from the
                    numbered sources and cite one. Used when LLM=bedrock and AWS credentials are set
                    (BEDROCK_MODEL_ID picks the model; there is deliberately no default).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

from .store import Hit

READER_MODEL = os.environ.get("READER_MODEL", "distilbert/distilbert-base-cased-distilled-squad")


@dataclass
class Answer:
    text: str
    source: str | None  # doc_id the answer came from
    confidence: float
    meta: dict = field(default_factory=dict)


@lru_cache(maxsize=1)
def _reader():
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(READER_MODEL)
    from .store import own_weights

    model = own_weights(AutoModelForQuestionAnswering.from_pretrained(READER_MODEL).eval())
    return tok, model


def extract(question: str, context: str, max_len: int = 40) -> tuple[str, float]:
    """The best answer span in `context` and its probability (start prob x end prob).

    Transformers 5 dropped the question-answering pipeline, so this is the same computation by hand:
    score every start and end token of the context, keep spans that run forward and are at most
    `max_len` tokens long, and map the best one back to characters.
    """
    import torch

    tok, model = _reader()
    enc = tok(question, context, truncation="only_second", max_length=384, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    ctx = [i for i, sid in enumerate(enc.sequence_ids(0)) if sid == 1]
    with torch.no_grad():
        out = model(**{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")})
    ps = out.start_logits[0].softmax(-1)
    pe = out.end_logits[0].softmax(-1)
    lo, hi = ctx[0], ctx[-1]
    # every (start, end) pair inside the context, end >= start, at most max_len apart
    grid = ps[lo : hi + 1, None] * pe[None, lo : hi + 1]
    grid = torch.triu(grid) - torch.triu(grid, diagonal=max_len)
    best = int(grid.argmax())
    i, j = divmod(best, grid.shape[1])
    s, e = offsets[lo + i][0], offsets[lo + j][1]
    return context[s:e], float(grid.flatten()[best])


def sentence(context: str, span: str) -> str:
    """The sentence of `context` holding `span`: a reader quotes a phrase, but a person (and the
    'did they raise, cut or hold?' questions) need the verb around it. A period right after a lone
    capital is an initial ("Esther L. George"), not a sentence end."""
    i = context.find(span)
    if i < 0 or not span.strip():
        return span.strip()
    ends = [m.end() for m in re.finditer(r"(?<![A-Z])\.(?:\s|$)", context)]
    start = max([e for e in ends if e <= i], default=0)
    end = min([e for e in ends if e >= i + len(span)], default=len(context))
    return context[start:end].strip()


class ExtractiveReader:
    name = "extractive"

    def answer(self, question: str, hits: list[Hit]) -> Answer:
        if not hits:
            return Answer("I couldn't find that in the FOMC statements.", None, 0.0)
        best = None
        for h in hits:
            # the chunk minus its title prefix: the reader should quote the statement, not our heading
            ctx = h.chunk.text[len(h.chunk.title) + 2 :] if h.chunk.text.startswith(h.chunk.title) else h.chunk.text
            span, p = extract(question, ctx)
            score = p * max(h.score, 0.0)
            if best is None or score > best[0]:
                best = (score, span, h.chunk.doc_id, p, ctx)
        _, span, doc, conf, ctx = best
        return Answer(sentence(ctx, span), doc, float(conf), {"span": span.strip()})


class BedrockAnswerer:
    name = "bedrock"

    def __init__(self, model_id: str | None = None):
        import boto3

        self.model_id = model_id or os.environ["BEDROCK_MODEL_ID"]
        self.client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))

    def answer(self, question: str, hits: list[Hit]) -> Answer:
        sources = "\n\n".join(f"[{i + 1}] ({h.chunk.title})\n{h.chunk.text}" for i, h in enumerate(hits))
        prompt = (
            "Answer the question using only the numbered FOMC statement excerpts below. Be brief: one sentence. "
            "End with the number of the excerpt you used in square brackets, e.g. [2]. If the excerpts don't "
            f"contain the answer, say so.\n\n{sources}\n\nQuestion: {question}"
        )
        r = self.client.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 200, "temperature": 0},
        )
        text = r["output"]["message"]["content"][0]["text"].strip()
        m = re.search(r"\[(\d+)\]\s*$", text)
        doc = hits[int(m.group(1)) - 1].chunk.doc_id if m and 0 < int(m.group(1)) <= len(hits) else None
        return Answer(re.sub(r"\s*\[\d+\]\s*$", "", text), doc, 1.0 if doc else 0.0, {"usage": r.get("usage", {})})


def answerer():
    return BedrockAnswerer() if os.environ.get("LLM") == "bedrock" else ExtractiveReader()
