from collections import Counter


from forgelm.evidence import EVIDENCE
from forgelm.packs import bfsi
from forgelm.students import TfidfStudent


def test_seed_set_covers_the_taxonomy_in_two_languages():
    seed = bfsi.load_seed_utterances()
    counts = Counter(s["label"] for s in seed)
    assert set(counts) == set(bfsi.TAXONOMY)
    assert min(counts.values()) >= 10
    assert {s["lang"] for s in seed} == {"en", "hi-en"}
    assert len({s["text"] for s in seed}) == len(seed)  # no duplicates


def test_rules_never_fire_on_the_compliant_and_neutral_seeds():
    for s in bfsi.load_seed_utterances():
        if s["label"] in ("compliant_disclosure", "neutral"):
            assert bfsi.rule_flags([s["text"]]) == [], s["text"]


def test_rules_alone_miss_risk_utterances_which_is_why_there_is_a_classifier_tier():
    risky = [s for s in bfsi.load_seed_utterances() if s["label"] in bfsi.RISK_LABELS]
    caught = sum(bool(bfsi.rule_flags([s["text"]])) for s in risky)
    assert 0.3 < caught / len(risky) < 0.95


def test_rule_flags_carry_the_triggering_span():
    flags = bfsi.rule_flags(["Good morning", "Sir isme pakka double paisa milega, koi risk nahi."])
    assert flags and flags[0].utterance_index == 1 and flags[0].category == "guaranteed_return_claim"
    assert flags[0].span.lower() in "sir isme pakka double paisa milega, koi risk nahi."


def test_disclosure_checks_find_evidence_or_report_missing():
    call = ["This call is being recorded for quality purposes.",
            "You get a free look period after receiving the policy.",
            "It is a market linked plan and returns are not guaranteed."]
    found = bfsi.check_disclosures(call)
    assert found["call_recorded_notice"] and found["free_look_period"] and found["returns_not_guaranteed"]
    assert found["lock_in_or_surrender"] is None
    review = bfsi.review_call(call)
    assert "lock_in_or_surrender" in review.missing_disclosures and review.flags == []


def test_review_call_combines_rules_classifier_and_reviewer_routing():
    seed = bfsi.load_seed_utterances()
    labels = list(bfsi.TAXONOMY)
    clf = TfidfStudent().fit([s["text"] for s in seed], [labels.index(s["label"]) for s in seed], labels)
    call = ["Namaste sir, kya meri baat Sharma ji se ho rahi hai?",
            "Your money will definitely double in five years, I am giving you my word on that.",
            "There are no charges at all in this plan, every rupee you pay is invested.",
            "Premium har saal das saal tak dena hoga."]
    review = bfsi.review_call(call, clf, flag_threshold=0.3, unsure_below=0.0)
    flagged = {f.utterance_index for f in review.flags}
    assert {1, 2} <= flagged and 0 not in flagged
    assert {f.tier for f in review.flags} == {"rule", "classifier"}
    assert review.needs_review == sorted(flagged)
    assert review.to_dict()["disclosures"]["premium_paying_term"]


def test_use_case_map_is_complete_and_cites_known_evidence():
    assert len(bfsi.USE_CASES) == 7
    for key, case in bfsi.USE_CASES.items():
        assert case["architecture"] in ("compact classifier", "classical ML", "hybrid cascade",
                                        "retrieval with a reasoning model"), key
        for ev in case["evidence"]:
            assert ev in EVIDENCE, (key, ev)
