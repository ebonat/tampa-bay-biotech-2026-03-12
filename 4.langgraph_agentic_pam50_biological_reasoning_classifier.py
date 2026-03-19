"""
LANGGRAPH AGENTIC PAM50 BIOLOGICAL REASONING CLASSIFIER
Author: Ernest Bonat, Ph.D - Senior GenAI Engineer

Architecture:
  Converts the monolithic LLMBasedPAM50Classifier into a LangGraph
  state-machine with dedicated agent nodes:

  ┌─────────────────────────────────────────────────────────────┐
  │                   LangGraph Workflow                        │
  │                                                             │
  │  [data_loader] → [feature_engineer] → [rule_agent]         │
  │        ↓                                                    │
  │  [llm_agent] ←→ [validation_agent] ←→ [fallback_agent]     │
  │        ↓                                                    │
  │  [results_aggregator]                                       │
  └─────────────────────────────────────────────────────────────┘

Nodes:
  1. data_loader          – Load & validate CSV files
  2. feature_engineer     – Calculate class stats + derived biomarker scores
  3. rule_agent           – Apply priority rule-based pre-classification
  4. llm_agent            – Claude API biological reasoning classification
  5. validation_agent     – Cross-check LLM vs rule-based; flag conflicts
  6. fallback_agent       – Fallback when LLM fails/disagrees beyond threshold
  7. results_aggregator   – Collect metrics, accuracy, per-subtype stats
"""

import os
import json
import re
import dill
import pandas as pd
import numpy as np
import anthropic

from typing import TypedDict, Annotated, List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import operator

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

SUBTYPES = ['LumA', 'LumB', 'Her2', 'Normal', 'Basal']

KEY_FEATURES = [
    'MKI67_expr', 'ESR1_expr', 'PGR_expr', 'CCNB1_expr', 'ERBB2_cna',
    'KRT5_expr', 'KRT17_expr', 'EGFR_expr', 'SFRP1_expr', 'PTTG1_expr',
    'UBE2C_expr', 'FOXC1_expr'
]

LABEL_COLS = ['PAM50', 'PAM50_Label', 'Sample_ID']

# ─────────────────────────────────────────────────────────────
# STATE DEFINITIONS
# ─────────────────────────────────────────────────────────────

class SampleResult(TypedDict):
    patient_id:         int
    true_label:         str
    rule_prediction:    Optional[str]
    rule_confidence:    Optional[str]
    rule_reasoning:     Optional[str]
    llm_prediction:     Optional[str]
    llm_confidence:     Optional[str]
    llm_reasoning:      Optional[str]
    final_prediction:   Optional[str]
    final_confidence:   Optional[str]
    final_method:       Optional[str]     # 'llm', 'rule', 'fallback'
    conflict:           bool
    status:             str               # 'pending' | 'completed' | 'failed'


class BatchState(TypedDict):
    # Inputs
    train_csv_path:  str
    test_csv_path:   str
    model_name:      str

    # Data
    train_df:        Optional[Any]        # pd.DataFrame stored as dict records
    test_df:         Optional[Any]
    class_stats:     Optional[Dict]

    # Per-sample results (appended by each patient node)
    results:         Annotated[List[SampleResult], operator.add]

    # Aggregated
    batch_metrics:   Dict[str, Any]
    errors:          Annotated[List[str], operator.add]

    # Timing
    start_time:      str
    end_time:        Optional[str]

    # Stats counters
    total_api_calls: int
    total_fallbacks: int


# ─────────────────────────────────────────────────────────────
# HELPER: RULE-BASED CLASSIFICATION
# (Preserves original priority from 2_claudeai_llm_biological_reasoning_classifier.py)
# ─────────────────────────────────────────────────────────────

def rule_classify(features: Dict) -> Tuple[str, str, str]:
    """
    Apply PAM50 classification rules in strict priority order.
    Returns (label, confidence, reasoning).
    """
    mki67  = features.get('MKI67_expr',  0)
    ccnb1  = features.get('CCNB1_expr',  0)
    esr1   = features.get('ESR1_expr',   0)
    pgr    = features.get('PGR_expr',    0)
    erbb2  = features.get('ERBB2_cna',   0)
    krt5   = features.get('KRT5_expr',   0)
    egfr   = features.get('EGFR_expr',   0)
    sfrp1  = features.get('SFRP1_expr',  0)

    prolif  = (mki67  + ccnb1) / 2
    hormone = (esr1   + pgr)   / 2
    basal   = (krt5   + egfr)  / 2

    # Rule 1: Atypical Basal — highest priority
    if erbb2 < 0 and hormone < 0 and ccnb1 > 2.0:
        return ('Basal', 'Medium',
                f"Atypical Basal: triple-neg + very high CCNB1={ccnb1:.2f} (Rule 1)")

    # Rule 2: Normal-like
    if sfrp1 > 0.5 and prolif < 0 and ccnb1 < 2.0:
        return ('Normal', 'High',
                f"Normal-like: SFRP1={sfrp1:.2f} + low proliferation (Rule 2)")

    # Rule 3: HER2-amplified
    if erbb2 >= 1.0 and hormone < 0:
        return ('Her2', 'High',
                f"HER2: ERBB2_cna={erbb2} + ER-/PR- (Rule 3)")

    # Rule 4 & 5: Luminal
    if hormone > 0:
        if prolif > 0 or ccnb1 > 0.5:
            conf = 'High' if prolif > 0 else 'Medium'
            return ('LumB', conf,
                    f"LumB: hormone+ + prolif={prolif:.2f} / CCNB1={ccnb1:.2f} (Rule 4)")
        else:
            return ('LumA', 'High',
                    f"LumA: hormone+ + low prolif={prolif:.2f} + CCNB1={ccnb1:.2f} (Rule 5)")

    # Rule 6: Classic Basal
    if basal > 1.0 and hormone < 0 and erbb2 < 0.5 and sfrp1 < 0.5:
        return ('Basal', 'High',
                f"Classic Basal: basal={basal:.2f} + triple-neg (Rule 6)")

    # Rule 7: HER2 by exclusion
    if hormone < 0 and basal < 1.0 and sfrp1 < 0.5:
        if prolif > -0.5 or ccnb1 > 0:
            return ('Her2', 'Medium',
                    f"Her2 by exclusion: ER-/PR- + low basal (Rule 7)")

    # Nearest prototype fallback
    prototypes = {
        'LumA':   {'ESR1': 0.5,  'PGR': 0.5,  'MKI67': -0.8, 'ERBB2': -0.5},
        'LumB':   {'ESR1': 0.5,  'PGR': 0.5,  'MKI67':  0.8, 'ERBB2': -0.5},
        'Her2':   {'ESR1': -0.8, 'PGR': -0.8, 'MKI67':  0.2, 'ERBB2':  1.0},
        'Normal': {'ESR1': 0,    'PGR': 0,    'MKI67': -0.5, 'ERBB2':  0},
        'Basal':  {'ESR1': -1.0, 'PGR': -1.0, 'MKI67':  0.5, 'ERBB2': -0.5},
    }
    distances = {
        s: ((esr1 - p['ESR1'])**2 + (pgr - p['PGR'])**2 +
            (mki67 - p['MKI67'])**2 + (erbb2 - p['ERBB2'])**2) ** 0.5
        for s, p in prototypes.items()
    }
    closest = min(distances, key=distances.get)
    return (closest, 'Low',
            f"Nearest prototype: {closest} (dist={distances[closest]:.2f}) (Rule 8)")


# ─────────────────────────────────────────────────────────────
# HELPER: LLM PROMPT
# ─────────────────────────────────────────────────────────────

def build_llm_prompt(features: Dict, rule_hint: str) -> str:
    """
    Build PAM50 classification prompt.
    Includes the rule-agent's suggestion as an advisory hint.
    """
    mki67  = features.get('MKI67_expr',  0)
    ccnb1  = features.get('CCNB1_expr',  0)
    esr1   = features.get('ESR1_expr',   0)
    pgr    = features.get('PGR_expr',    0)
    erbb2  = features.get('ERBB2_cna',   0)
    krt5   = features.get('KRT5_expr',   0)
    egfr   = features.get('EGFR_expr',   0)
    sfrp1  = features.get('SFRP1_expr',  0)

    return f"""You are an expert oncologist. Classify this breast cancer sample into ONE PAM50 subtype: LumA, LumB, Her2, Normal, or Basal.

# MOLECULAR PROFILE
MKI67 (proliferation):  {mki67:.3f}
CCNB1 (cell cycle):     {ccnb1:.3f}  ⚠️ CHECK FIRST!
ESR1  (estrogen):       {esr1:.3f}
PGR   (progesterone):   {pgr:.3f}
ERBB2_cna (HER2 amp):   {erbb2:.3f}
KRT5  (basal):          {krt5:.3f}
EGFR  (basal):          {egfr:.3f}
SFRP1 (normal-like):    {sfrp1:.3f}

# RULE-BASED AGENT SUGGESTION (advisory): {rule_hint}

# CLASSIFICATION RULES — apply in strict order

RULE 1 - Atypical Basal (HIGHEST PRIORITY):
  If ESR1 < 0 AND PGR < 0 AND ERBB2 < 0.5 AND CCNB1 > 2.0 → Basal
  (Overrides everything, even high SFRP1!)

RULE 2 - Normal-like:
  If SFRP1 > 0.5 AND (MKI67 < 0 OR CCNB1 < 0) AND CCNB1 < 2.0 → Normal

RULE 3 - HER2-enriched:
  If ERBB2_cna >= 1.0 AND (ESR1 < 0 OR PGR < 0) → Her2

RULE 4 - Luminal B:
  If (ESR1 > 0 OR PGR > 0) AND (MKI67 > 0 OR CCNB1 > 0.5) → LumB

RULE 5 - Luminal A:
  If (ESR1 > 0 OR PGR > 0) AND MKI67 < 0 AND CCNB1 < 0.5 → LumA

RULE 6 - Classic Basal:
  If (ESR1 < 0 AND PGR < 0) AND SFRP1 < 0.5 AND (KRT5 > 0 OR EGFR > 0) → Basal

RULE 7 - HER2 by exclusion:
  If (ESR1 < 0 AND PGR < 0) AND SFRP1 < 0.5 → Her2

# RESPONSE FORMAT (strict):
CLASSIFICATION: [LumA|LumB|Her2|Normal|Basal]
CONFIDENCE: [High|Medium|Low]
REASONING: [which rule triggered and why]"""


# ─────────────────────────────────────────────────────────────
# HELPER: PARSE LLM RESPONSE
# ─────────────────────────────────────────────────────────────

def parse_llm_response(text: str, features: Dict) -> Tuple[Optional[str], str, str]:
    """
    Parse structured LLM response.
    Returns (label, confidence, reasoning).
    Falls back to keyword scanning.
    """
    label = confidence = reasoning = None

    for line in text.strip().splitlines():
        line = line.strip()
        upper = line.upper()

        if 'CLASSIFICATION:' in upper:
            val = line.split(':', 1)[1].strip()
            for s in SUBTYPES:
                if s.lower() in val.lower():
                    label = s
                    break

        elif 'CONFIDENCE:' in upper and 'REASONING' not in upper:
            val = line.split(':', 1)[1].strip()
            confidence = val.split()[0] if val else 'Medium'

        elif 'REASONING:' in upper:
            reasoning = line.split(':', 1)[1].strip()

    # Fallback scan
    if not label:
        lower = text.lower()
        counts = {s: len(re.findall(s.lower(), lower)) for s in SUBTYPES}
        best = max(counts, key=counts.get)
        if counts[best] > 0:
            label = best

    return (label, confidence or 'Medium', reasoning or 'LLM classification')


# ─────────────────────────────────────────────────────────────
# NODE 1: DATA LOADER
# ─────────────────────────────────────────────────────────────

def data_loader(state: BatchState) -> BatchState:
    print(f"\n{'='*60}")
    print("NODE 1 ▶ DATA LOADER")
    print(f"{'='*60}")

    try:
        train_df = pd.read_csv(state['train_csv_path'])
        test_df  = pd.read_csv(state['test_csv_path'])

        for df in (train_df, test_df):
            df.columns = df.columns.str.strip()

        # Drop Sample_ID if present
        for col in ['Sample_ID']:
            if col in train_df.columns:
                train_df = train_df.drop(columns=[col])
            if col in test_df.columns:
                test_df = test_df.drop(columns=[col])

        print(f"  ✓ Train: {train_df.shape[0]} rows × {train_df.shape[1]} cols")
        print(f"  ✓ Test:  {test_df.shape[0]}  rows × {test_df.shape[1]}  cols")
        print(f"  ✓ Test labels: {test_df['PAM50'].tolist()}")

        state['train_df'] = train_df.to_dict(orient='records')
        state['test_df']  = test_df.to_dict(orient='records')

    except Exception as e:
        state['errors'] = [f"data_loader: {e}"]

    return state


# ─────────────────────────────────────────────────────────────
# NODE 2: FEATURE ENGINEER
# ─────────────────────────────────────────────────────────────

def feature_engineer(state: BatchState) -> BatchState:
    print(f"\n{'='*60}")
    print("NODE 2 ▶ FEATURE ENGINEER")
    print(f"{'='*60}")

    try:
        train_df = pd.DataFrame(state['train_df'])

        # Calculate per-subtype mean/std for key features
        class_stats: Dict = {}
        for subtype in SUBTYPES:
            sub = train_df[train_df['PAM50'] == subtype]
            class_stats[subtype] = {}
            for feat in KEY_FEATURES:
                if feat in sub.columns:
                    class_stats[subtype][feat] = {
                        'mean': float(sub[feat].mean()),
                        'std':  float(sub[feat].std())
                    }
            print(f"  ✓ {subtype}: n={len(sub)}")

        state['class_stats'] = class_stats
        print(f"\n  ✓ Class statistics computed for {len(SUBTYPES)} subtypes")

    except Exception as e:
        state['errors'] = [f"feature_engineer: {e}"]

    return state


# ─────────────────────────────────────────────────────────────
# NODE 3: RULE AGENT
# ─────────────────────────────────────────────────────────────

def rule_agent(state: BatchState) -> BatchState:
    print(f"\n{'='*60}")
    print("NODE 3 ▶ RULE AGENT  (priority rule-based pre-classification)")
    print(f"{'='*60}")

    test_rows = state['test_df']
    sample_results: List[SampleResult] = []

    for idx, row in enumerate(test_rows):
        true_label  = row.get('PAM50', 'Unknown')
        features    = {k: v for k, v in row.items() if k not in LABEL_COLS}

        label, conf, reasoning = rule_classify(features)

        result: SampleResult = {
            'patient_id':       idx,
            'true_label':       true_label,
            'rule_prediction':  label,
            'rule_confidence':  conf,
            'rule_reasoning':   reasoning,
            'llm_prediction':   None,
            'llm_confidence':   None,
            'llm_reasoning':    None,
            'final_prediction': None,
            'final_confidence': None,
            'final_method':     None,
            'conflict':         False,
            'status':           'pending',
        }

        marker = "✓" if label == true_label else "✗"
        print(f"  {marker} Patient {idx}: Rule → {label} ({conf}) | True = {true_label}")
        print(f"     Reasoning: {reasoning}")

        sample_results.append(result)

    state['results'] = sample_results
    return state


# ─────────────────────────────────────────────────────────────
# NODE 4: LLM AGENT
# ─────────────────────────────────────────────────────────────

def llm_agent(state: BatchState) -> BatchState:
    print(f"\n{'='*60}")
    print("NODE 4 ▶ LLM AGENT  (Claude biological reasoning)")
    print(f"{'='*60}")

    api_key = os.getenv('ANTHROPIC_API_KEY')
    model   = state.get('model_name') or os.getenv('ANTHROPIC_LLM') or 'claude-sonnet-4-20250514'

    if not api_key:
        print("  ⚠️  No ANTHROPIC_API_KEY — skipping LLM agent")
        state['errors'] = ['llm_agent: ANTHROPIC_API_KEY not set']
        return state

    client     = anthropic.Anthropic(api_key=api_key)
    test_rows  = state['test_df']
    results    = list(state['results'])   # copy to mutate
    api_calls  = 0

    for res in results:
        idx      = res['patient_id']
        row      = test_rows[idx]
        features = {k: v for k, v in row.items() if k not in LABEL_COLS}
        hint     = f"{res['rule_prediction']} ({res['rule_confidence']})"
        prompt   = build_llm_prompt(features, rule_hint=hint)

        try:
            message = client.messages.create(
                model=model,
                max_tokens=1000,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            raw      = message.content[0].text
            api_calls += 1

            label, conf, reasoning = parse_llm_response(raw, features)
            res['llm_prediction'] = label
            res['llm_confidence'] = conf
            res['llm_reasoning']  = reasoning

            marker = "✓" if label == res['true_label'] else "✗"
            print(f"  {marker} Patient {idx}: LLM  → {label} ({conf}) | True = {res['true_label']}")
            print(f"     Reasoning: {reasoning}")

        except Exception as e:
            print(f"  ✗ Patient {idx}: LLM error — {e}")
            res['llm_prediction'] = None
            res['llm_reasoning']  = str(e)

    state['results']          = results
    state['total_api_calls']  = api_calls
    return state


# ─────────────────────────────────────────────────────────────
# NODE 5: VALIDATION AGENT
# ─────────────────────────────────────────────────────────────

def validation_agent(state: BatchState) -> BatchState:
    print(f"\n{'='*60}")
    print("NODE 5 ▶ VALIDATION AGENT  (cross-check LLM vs rules)")
    print(f"{'='*60}")

    results   = list(state['results'])
    fallbacks = 0

    for res in results:
        rule_pred = res['rule_prediction']
        llm_pred  = res['llm_prediction']

        if llm_pred and llm_pred == rule_pred:
            # Agreement → accept LLM with high confidence
            res['final_prediction'] = llm_pred
            res['final_confidence'] = 'High'
            res['final_method']     = 'llm+rule_agree'
            res['conflict']         = False
            print(f"  ✓ Patient {res['patient_id']}: AGREEMENT  → {llm_pred}")

        elif llm_pred and llm_pred != rule_pred:
            # Conflict → prefer LLM (biological reasoning) but flag
            res['final_prediction'] = llm_pred
            res['final_confidence'] = res['llm_confidence'] or 'Medium'
            res['final_method']     = 'llm_override'
            res['conflict']         = True
            print(f"  ⚠️  Patient {res['patient_id']}: CONFLICT  LLM={llm_pred} vs Rule={rule_pred} → using LLM")

        else:
            # LLM failed → use rule-based
            res['final_prediction'] = rule_pred
            res['final_confidence'] = res['rule_confidence'] or 'Medium'
            res['final_method']     = 'rule_fallback'
            res['conflict']         = False
            fallbacks              += 1
            print(f"  ↩  Patient {res['patient_id']}: LLM FAILED → fallback to Rule = {rule_pred}")

        res['status'] = 'completed'

    state['results']         = results
    state['total_fallbacks'] = fallbacks
    return state


# ─────────────────────────────────────────────────────────────
# NODE 6: RESULTS AGGREGATOR
# ─────────────────────────────────────────────────────────────

def results_aggregator(state: BatchState) -> BatchState:
    print(f"\n{'='*60}")
    print("NODE 6 ▶ RESULTS AGGREGATOR")
    print(f"{'='*60}")

    results   = state['results']
    completed = [r for r in results if r['status'] == 'completed']

    if not completed:
        state['batch_metrics'] = {'error': 'No completed samples'}
        return state

    # ── Accuracy breakdown ──────────────────────────────────
    true_labels  = [r['true_label']       for r in completed]
    predictions  = [r['final_prediction'] for r in completed]

    rule_preds   = [r['rule_prediction']  for r in completed]
    llm_preds    = [r['llm_prediction']   for r in completed]

    def accuracy(preds, trues):
        return sum(p == t for p, t in zip(preds, trues)) / len(trues) if trues else 0

    final_acc = accuracy(predictions,  true_labels)
    rule_acc  = accuracy(rule_preds,   true_labels)
    llm_acc   = accuracy(
        [p for p in llm_preds if p is not None],
        [t for p, t in zip(llm_preds, true_labels) if p is not None]
    )

    # ── Confusion matrix ────────────────────────────────────
    confusion: Dict = defaultdict(lambda: defaultdict(int))
    for r in completed:
        confusion[r['true_label']][r['final_prediction']] += 1

    # ── Per-subtype metrics ─────────────────────────────────
    subtype_metrics: Dict = {}
    for sub in SUBTYPES:
        tp = confusion[sub][sub]
        fp = sum(confusion[o][sub]  for o in SUBTYPES if o != sub)
        fn = sum(confusion[sub][o]  for o in SUBTYPES if o != sub)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0
        subtype_metrics[sub] = {
            'precision': round(prec, 4),
            'recall':    round(rec,  4),
            'f1_score':  round(f1,   4),
            'support':   sum(confusion[sub].values())
        }

    # ── Method breakdown ────────────────────────────────────
    method_counts = defaultdict(int)
    for r in completed:
        method_counts[r['final_method']] += 1

    conflicts = sum(1 for r in completed if r['conflict'])

    state['batch_metrics'] = {
        'total_samples':      len(completed),
        'final_accuracy':     round(final_acc, 4),
        'rule_accuracy':      round(rule_acc,  4),
        'llm_accuracy':       round(llm_acc,   4),
        'total_api_calls':    state.get('total_api_calls',  0),
        'total_fallbacks':    state.get('total_fallbacks',  0),
        'conflicts_detected': conflicts,
        'method_breakdown':   dict(method_counts),
        'subtype_metrics':    subtype_metrics,
        'confusion_matrix':   {k: dict(v) for k, v in confusion.items()},
    }

    state['end_time'] = datetime.now().isoformat()

    # ── Print summary ───────────────────────────────────────
    print(f"\n  {'─'*50}")
    print(f"  INDIVIDUAL RESULTS:")
    print(f"  {'─'*50}")
    for r in completed:
        ok = "✅" if r['final_prediction'] == r['true_label'] else "❌"
        cf = " ⚠️ CONFLICT" if r['conflict'] else ""
        print(f"  {ok} Patient {r['patient_id']:>2}: "
              f"True={r['true_label']:<7} | Pred={r['final_prediction']:<7} "
              f"| {r['final_method']}{cf}")

    print(f"\n  {'─'*50}")
    print(f"  ACCURACY SUMMARY:")
    print(f"  {'─'*50}")
    correct = sum(p == t for p, t in zip(predictions, true_labels))
    print(f"  Final (LangGraph):  {correct}/{len(completed)} = {final_acc:.0%}")
    print(f"  Rule agent only:    {rule_acc:.0%}")
    print(f"  LLM agent only:     {llm_acc:.0%}")
    print(f"  Conflicts detected: {conflicts}")
    print(f"  API calls:          {state.get('total_api_calls', 0)}")
    print(f"  Fallbacks:          {state.get('total_fallbacks', 0)}")

    print(f"\n  {'─'*50}")
    print(f"  PER-SUBTYPE METRICS (final predictions):")
    print(f"  {'─'*50}")
    print(f"  {'Subtype':<8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
    for sub, m in subtype_metrics.items():
        print(f"  {sub:<8} {m['precision']:>10.2%} {m['recall']:>8.2%} "
              f"{m['f1_score']:>8.2%} {m['support']:>8}")

    if final_acc == 1.0:
        print(f"\n  🎉 PERFECT! 100% accuracy achieved!")

    return state


# ─────────────────────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────────

def build_graph():
    """Compile the LangGraph state machine."""
    wf = StateGraph(BatchState)

    wf.add_node("data_loader",         data_loader)
    wf.add_node("feature_engineer",    feature_engineer)
    wf.add_node("rule_agent",          rule_agent)
    wf.add_node("llm_agent",           llm_agent)
    wf.add_node("validation_agent",    validation_agent)
    wf.add_node("results_aggregator",  results_aggregator)

    wf.set_entry_point("data_loader")
    wf.add_edge("data_loader",        "feature_engineer")
    wf.add_edge("feature_engineer",   "rule_agent")
    wf.add_edge("rule_agent",         "llm_agent")
    wf.add_edge("llm_agent",          "validation_agent")
    wf.add_edge("validation_agent",   "results_aggregator")
    wf.add_edge("results_aggregator", END)

    return wf.compile()


# ─────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────

def run_pipeline(
    train_csv: str,
    test_csv:  str,
    model:     str = "claude-sonnet-4-20250514",
    save_json: Optional[str] = None
) -> Optional[BatchState]:
    """
    Execute the full LangGraph PAM50 agentic pipeline.

    Args:
        train_csv:  Path to synthetic training CSV
        test_csv:   Path to production test CSV
        model:      Anthropic model name
        save_json:  Optional output JSON path
    """
    print(f"\n{'#'*60}")
    print("LANGGRAPH AGENTIC PAM50 BIOLOGICAL REASONING CLASSIFIER")
    print(f"{'#'*60}")
    print(f"  Model:     {model}")
    print(f"  Train CSV: {train_csv}")
    print(f"  Test CSV:  {test_csv}")
    print(f"  Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    initial: BatchState = {
        'train_csv_path':  train_csv,
        'test_csv_path':   test_csv,
        'model_name':      model,
        'train_df':        None,
        'test_df':         None,
        'class_stats':     None,
        'results':         [],
        'batch_metrics':   {},
        'errors':          [],
        'start_time':      datetime.now().isoformat(),
        'end_time':        None,
        'total_api_calls': 0,
        'total_fallbacks': 0,
    }

    graph = build_graph()

    try:
        final = graph.invoke(initial)

        # Processing time
        if final['end_time']:
            t = (datetime.fromisoformat(final['end_time']) -
                 datetime.fromisoformat(final['start_time'])).total_seconds()
            final['batch_metrics']['processing_time_sec'] = round(t, 2)

        # Save JSON
        if save_json:
            output = {
                'batch_metrics':    final['batch_metrics'],
                'patient_results': [
                    {k: v for k, v in r.items()
                     if k not in ('true_label',)}   # keep all fields
                    for r in final['results']
                ],
                'errors': final['errors']
            }
            os.makedirs(os.path.dirname(save_json), exist_ok=True) if os.path.dirname(save_json) else None
            with open(save_json, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"\n  ✓ Results saved → {save_json}")

        print(f"\n{'#'*60}")
        print(f"  Processing time: {final['batch_metrics'].get('processing_time_sec', 'N/A')}s")
        print(f"  Errors: {final['errors'] if final['errors'] else 'none'}")
        print(f"{'#'*60}\n")

        return final

    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    TRAIN_CSV = (
        sys.argv[1] if len(sys.argv) > 1
        else 
    )
    TEST_CSV = (
        sys.argv[2] if len(sys.argv) > 2
        else
    )
    MODEL = sys.argv[3] if len(sys.argv) > 3 else "claude-sonnet-4-20250514"
    OUT   = (
        sys.argv[4] if len(sys.argv) > 4
        else 
    )

    results = run_pipeline(TRAIN_CSV, TEST_CSV, model=MODEL, save_json=OUT)

    if results:
        acc = results['batch_metrics'].get('final_accuracy', 0)
        print(f"✓ Pipeline complete. Final accuracy: {acc:.0%}")
        sys.exit(0)
    else:
        print("✗ Pipeline failed.")
        sys.exit(1)
        
        
    # UPDATE TERMINAL CODE 
    # 1. rigth click "Open in Integrate Terminal"
    # 2. conda activate python3.10.12.genai
    # 3. python -c "import sys; print(sys.executable)"
    # 4. python 4.langgraph_agentic_pam50_biological_reasoning_classifier.py
