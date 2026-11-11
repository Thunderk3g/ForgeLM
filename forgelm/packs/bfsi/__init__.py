"""BFSI pack: insurance sales-conversation screening and a use-case map.

What is here
------------
* ``USE_CASES``: for each common BFSI / compliance task, which kind of system
  the evidence supports (compact classifier, classical ML, retrieval, a large
  reasoning model, or a cascade), and how strong that evidence is.
* ``review_call``: a three-tier screen for a sales conversation:
  deterministic rules -> compact classifier -> items routed to a reviewer.
  Every flag carries the exact text span that triggered it.
* A small seed set of sales-call utterances (English and Hinglish) for the
  taxonomy below.

What this is not
----------------
The checklist, patterns and seed utterances are illustrative starting points
written for this repository. They are **not** a statement of what any
regulator requires and have not been validated on real calls. The disclosure
list and risk taxonomy must be set by your compliance function, and the
classifier must be trained and evaluated on your own reviewed transcripts
(including ASR errors and code-mixed speech) before it screens anything.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

HERE = Path(__file__).parent

# --------------------------------------------------------------------------
# Taxonomy for utterance-level screening of agent speech
# --------------------------------------------------------------------------

TAXONOMY: Dict[str, str] = {
    "compliant_disclosure": "the agent accurately states a material fact: free-look, premium and policy term, "
                            "lock-in, charges, exclusions, or that returns are not guaranteed",
    "guaranteed_return_claim": "the agent promises fixed, assured or guaranteed returns or a certain maturity "
                               "amount on a product whose returns are not guaranteed",
    "product_misrepresentation": "the agent presents the policy as something else: a fixed deposit, a bank "
                                 "savings or government scheme, a one-time payment, or a loan requirement",
    "pressure_tactic": "the agent creates false urgency or discourages the customer from reading the document "
                       "or consulting anyone",
    "suppressed_charges_or_lockin": "the agent denies or plays down charges, lock-in, surrender penalties or "
                                    "the consequences of stopping premiums",
    "false_inducement": "the agent offers a loan, bonus, refund, gift, job or other benefit that is conditional "
                        "on buying, or impersonates an authority",
    "neutral": "greetings, identity checks, fact-finding and logistics",
}
RISK_LABELS = [l for l in TAXONOMY if l not in ("compliant_disclosure", "neutral")]


def load_seed_utterances() -> List[dict]:
    """The illustrative seed set. Written for this repo; not real call data."""
    with open(HERE / "seed_sales_utterances.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------
# Tier 1: deterministic rules (cheap, auditable, span-level)
# --------------------------------------------------------------------------

# Things that should be *said* at least once in a call. Patterns cover English
# and common romanised Hindi phrasings; extend them from your own transcripts.
DISCLOSURE_CHECKS: Dict[str, str] = {
    "free_look_period": r"free[\s-]?look",
    "returns_not_guaranteed": r"(not|nahi|nahin)\s+(be\s+)?guarantee|guarantee[d]?\s+(nahi|nahin|not)|"
                              r"market[\s-]?linked|subject to market|depend[s]? on the market",
    "premium_paying_term": r"premium.{0,40}(year|saal|term|month|mahine)|(year|saal).{0,25}premium",
    "lock_in_or_surrender": r"lock[\s-]?in|surrender",
    "call_recorded_notice": r"call.{0,30}record|record.{0,20}call|recording par",
    "exclusions_or_waiting": r"exclusion|waiting period|pre[\s-]?existing",
}

# Phrases that are almost never acceptable in agent speech. A hit is a flag for
# review with the matched span, not a finding of a violation.
RED_FLAGS: Dict[str, str] = {
    "guaranteed_return_claim": r"guaranteed?\s+(\w+\s+){0,3}(return|profit|percent)|pakka\s+(double|return)|"
                               r"double\s+(paisa|your money|in \w+ years)|money will (definitely |surely )?double|"
                               r"(fix|fixed|assured)\s+return|no risk|koi risk nahi|100\s?%\s+safe|"
                               r"hundred percent safe|meri guarantee",
    "product_misrepresentation": r"(like|jaisa)\s+(a\s+)?(fixed deposit|fd|recurring deposit|rd)\b|"
                                 r"\bfd\s+jaisa|not\s+(really\s+)?an?\s+insurance|insurance nahi hai|"
                                 r"government scheme|sarkari yojana|"
                                 r"(loan).{0,40}(mandatory|zaroori|compulsory)|(mandatory|zaroori).{0,40}loan",
    "pressure_tactic": r"(offer|plan).{0,30}(today|aaj).{0,20}(only|closes|ends|tak|band)|"
                       r"(no need|zaroorat nahi).{0,20}(read|padh)|share the otp|otp bata",
    "suppressed_charges_or_lockin": r"no charges?( at all)?|koi charge nahi|no lock[\s-]?in|koi lock[\s-]?in nahi|"
                                    r"nothing is deducted|kuch nahi kat",
    "false_inducement": r"(interest[\s-]?free|bina byaaj).{0,20}loan|bonus.{0,40}(lying|pada|release)|"
                        r"(irda|irdai|regulator).{0,25}(office|se bol)|gold coin",
}


@dataclass
class Flag:
    tier: str  # "rule" | "classifier"
    category: str
    utterance_index: int
    span: str  # the exact text that triggered the flag
    confidence: Optional[float] = None


@dataclass
class CallReview:
    disclosures: Dict[str, Optional[str]]  # check -> evidence span, or None if not found
    flags: List[Flag] = field(default_factory=list)
    needs_review: List[int] = field(default_factory=list)  # utterance indices for a person or larger model

    @property
    def missing_disclosures(self) -> List[str]:
        return [k for k, v in self.disclosures.items() if v is None]

    def to_dict(self) -> dict:
        return {"disclosures": self.disclosures, "missing_disclosures": self.missing_disclosures,
                "flags": [f.__dict__ for f in self.flags], "needs_review": self.needs_review}


def check_disclosures(utterances: Sequence[str], checks: Optional[Dict[str, str]] = None) -> Dict[str, Optional[str]]:
    """For each required disclosure, the first utterance span that satisfies it."""
    found: Dict[str, Optional[str]] = {}
    for name, pattern in (checks or DISCLOSURE_CHECKS).items():
        rx = re.compile(pattern, re.IGNORECASE)
        found[name] = next((m.group(0) for u in utterances if (m := rx.search(u))), None)
    return found


def rule_flags(utterances: Sequence[str], patterns: Optional[Dict[str, str]] = None) -> List[Flag]:
    out: List[Flag] = []
    for category, pattern in (patterns or RED_FLAGS).items():
        rx = re.compile(pattern, re.IGNORECASE)
        for i, u in enumerate(utterances):
            m = rx.search(u)
            if m:
                out.append(Flag("rule", category, i, m.group(0)))
    return out


def review_call(utterances: Sequence[str], classifier=None, flag_threshold: float = 0.6,
                unsure_below: float = 0.5) -> CallReview:
    """Screen the agent's utterances of one call.

    Tier 1  rules: required disclosures present? red-flag phrases?
    Tier 2  ``classifier`` (any forgelm student trained on ``TAXONOMY``):
            flags utterances it assigns to a risk label with confidence
            >= ``flag_threshold``.
    Tier 3  anything flagged, plus anything the classifier is unsure about
            (top confidence < ``unsure_below``), is listed in ``needs_review``
            for a person or a larger model. Nothing is auto-decided here.
    """
    utterances = list(utterances)
    review = CallReview(check_disclosures(utterances), rule_flags(utterances))
    unsure: List[int] = []
    if classifier is not None and utterances:
        probs = classifier.predict_proba(utterances)
        for i, row in enumerate(probs):
            top = int(row.argmax())
            label, conf = classifier.labels[top], float(row[top])
            if label in RISK_LABELS and conf >= flag_threshold:
                review.flags.append(Flag("classifier", label, i, utterances[i], round(conf, 4)))
            elif conf < unsure_below:
                unsure.append(i)
    review.needs_review = sorted({f.utterance_index for f in review.flags} | set(unsure))
    return review


# --------------------------------------------------------------------------
# Use case -> architecture
# --------------------------------------------------------------------------

USE_CASES: Dict[str, dict] = {
    "sales_call_compliance": {
        "title": "Sales-call and sales-script compliance, mis-selling detection",
        "architecture": "hybrid cascade",
        "recipe": "rules for mandatory disclosures -> compact classifier or fine-tuned small judge on agent "
                  "utterances -> larger model on flagged spans -> human reviewer",
        "why": "frontier LLM judges alone were only modestly accurate on conversation-compliance benchmarks and "
               "a fine-tuned compact judge matched them; audit needs span-level evidence, not a free-text verdict",
        "costly_error": "missed violation",
        "evidence": ["compliance-judges", "e6-screening"],
        "proxy_dataset": "none public for Indian insurance calls; coastalcph/lex_glue (unfair_tos) as a stand-in",
    },
    "policy_regulation_qa": {
        "title": "Policy interpretation and regulatory Q&A",
        "architecture": "retrieval with a reasoning model",
        "recipe": "hybrid retrieval over the current text, answer with verbatim citations, human review for "
                  "anything customer-facing",
        "why": "a knowledge task whose source text changes; retrieval beats fine-tuning for knowledge",
        "costly_error": "confident wrong answer",
        "evidence": ["rag-over-finetune"],
        "proxy_dataset": "ObliQA (regulatory QA), theatticusproject/cuad-qa",
    },
    "document_classification": {
        "title": "Document classification and routing (KYC, proposals, claims)",
        "architecture": "compact classifier",
        "recipe": "fine-tuned encoder (layout-aware for scans); ModernBERT-class for long text",
        "why": "labels are cheap and plentiful and the classes are stable",
        "costly_error": "mis-routing (moderate)",
        "evidence": ["e1-ladder", "encoders-beat-zero-shot-llm"],
        "proxy_dataset": "coastalcph/lex_glue (ledgar), RVL-CDIP for scanned documents",
    },
    "risk_categorisation": {
        "title": "Risk categorisation, underwriting triage, lapse and fraud scoring",
        "architecture": "classical ML",
        "recipe": "logistic regression / EBM as the explainable champion, gradient boosting as the accuracy "
                  "bar, an in-context tabular model as a challenger on small IID data",
        "why": "tabular baselines were not beaten in published insurance tests and each decision may need "
               "stated reasons",
        "costly_error": "under-priced risk, unfair decline",
        "evidence": ["e4-tabular", "tabular-shift", "ebm"],
        "proxy_dataset": "scikit-learn/credit-card-clients, scikit-learn/adult-census-income",
    },
    "marketing_guideline_adherence": {
        "title": "Guideline adherence of marketing and advertising material",
        "architecture": "hybrid cascade",
        "recipe": "rules for mandatory wording -> retrieval over the advertising rules -> larger model -> "
                  "human sign-off",
        "why": "no benchmark exists for this task; treat every automated verdict as unproven",
        "costly_error": "missed misleading claim",
        "evidence": ["e6-screening"],
        "proxy_dataset": "coastalcph/lex_glue (unfair_tos) for the screening step",
    },
    "complaint_classification": {
        "title": "Complaint classification and regulatory reporting",
        "architecture": "compact classifier",
        "recipe": "frozen encoder + linear head from a few labels per category, fine-tuned encoder once "
                  "labels accumulate; abstain and route below a confidence threshold",
        "why": "short texts, abundant labels, fine-grained stable categories",
        "costly_error": "a mis-selling complaint filed under a benign category",
        "evidence": ["e1-ladder", "e5-cascade"],
        "proxy_dataset": "PolyAI/banking77, CFPB consumer complaints",
    },
    "regulatory_change_monitoring": {
        "title": "Regulatory change monitoring and obligation extraction",
        "architecture": "hybrid cascade",
        "recipe": "compact obligation-sentence classifier over new circulars -> retrieval against internal "
                  "policies -> larger model drafts the mapping -> human approves",
        "why": "sentence-level obligation detection is a solved classifier task; end-to-end mapping is not",
        "costly_error": "missed obligation",
        "evidence": ["compliance-judges", "rag-over-finetune"],
        "proxy_dataset": "ObliQA",
    },
}
