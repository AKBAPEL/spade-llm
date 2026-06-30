#!/usr/bin/env python3
"""Calculate auction metrics from dialogue/bid records and legacy logs."""

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Seaborn/pandas make charts more beautiful; fall back to plain matplotlib if absent.
try:
    import seaborn as sns

    HAS_SEABORN = True
except Exception:  # pragma: no cover
    HAS_SEABORN = False

try:
    import pandas as pd

    HAS_PANDAS = True
except Exception:  # pragma: no cover
    HAS_PANDAS = False

DIALOGUE_PATTERN = re.compile(
    r"^(Self|Opponent)\s+\[(\S+)\]\s+\(([^)]+)\):\s*(.*)$"
)
OUTCOME_PATTERN = re.compile(r"^=== Outcome:\s*(.*?)\s*===")
OUTCOME_TYPES = {
    "success": "success — контрагент согласился и прислал ShopList",
    "timeout": "timeout — не получен ответ от контрагента",
    "round_limit": "round_limit — исчерпан лимит раундов",
    "refused": "refused — контрагент отказался от переговоров",
    "error": "error — некорректный формат или неожиданный performative",
    "incomplete": "incomplete — диалог завершился без результата",
}
OUTCOME_ALIASES = {v: k for k, v in OUTCOME_TYPES.items()}

MECHANISM_ORDER = ["none", "personal", "system", "personal_system"]


def set_style() -> None:
    """Apply a consistent, presentation-ready style to all plots."""
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    if HAS_SEABORN:
        sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)


def _mechanism_order(label: str) -> int:
    try:
        return MECHANISM_ORDER.index(label)
    except ValueError:
        return len(MECHANISM_ORDER)


def _palette(n: int) -> List[Tuple[float, float, float]]:
    if HAS_SEABORN:
        return sns.color_palette("muted", n_colors=n)
    return plt.cm.tab10(np.linspace(0, 1, n))


@dataclass
class DialogueResult:
    initiator: str
    contragent: str
    messages: List[Tuple[str, str, str, str]] = field(default_factory=list)
    """List of (role, agent, action_type, content)."""
    outcome_text: str = ""
    outcome: str = "unknown"
    timestamp: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.messages)

    @property
    def agreement_round(self) -> Optional[int]:
        """Return round number (1-based) where opponent sent ShopList."""
        for idx, (role, _agent, action_type, _content) in enumerate(self.messages, start=1):
            if role == "Opponent" and "ShopList" in action_type:
                return idx // 2
        return None


@dataclass
class AgentSplit:
    agent: str
    ingredients: Dict[str, int]
    order_sum: int
    shop_sum: int
    margin_pct: float


@dataclass
class WinningBidResult:
    agents: List[str]
    v: int
    p: int
    c: int
    metric_1: float
    metric_2: float
    splits: List[AgentSplit]
    timestamp: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Legacy text-log parsing (kept for backward compatibility)
# ---------------------------------------------------------------------------


def parse_dialogue(path: Path) -> DialogueResult:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    timestamp = ""
    initiator = ""
    contragent = ""
    outcome = "unknown"
    outcome_text = ""
    messages: List[Tuple[str, str, str, str]] = []

    header_match = re.search(r"=== Dialogue log started at (\S+) ===", text)
    if header_match:
        timestamp = header_match.group(1)

    for line in lines:
        line = line.rstrip()
        if line.startswith("Initiator agent:"):
            initiator = line.split(":", 1)[1].strip()
        elif line.startswith("Contragent agent:"):
            contragent = line.split(":", 1)[1].strip()
        elif match := DIALOGUE_PATTERN.match(line):
            role, agent, action_type, content = match.groups()
            messages.append((role, agent, action_type.strip(), content.strip()))
        elif match := OUTCOME_PATTERN.match(line):
            outcome_text = match.group(1).strip()
            outcome = OUTCOME_ALIASES.get(outcome_text, "unknown")

    return DialogueResult(
        initiator=initiator,
        contragent=contragent,
        messages=messages,
        outcome_text=outcome_text,
        outcome=outcome,
        timestamp=timestamp,
    )


def parse_winning_bid(path: Path) -> WinningBidResult:
    text = path.read_text(encoding="utf-8")

    timestamp = ""
    header_match = re.search(r"=== Winning bid split log started at (\S+) ===", text)
    if header_match:
        timestamp = header_match.group(1)

    agents_line = re.search(r"Winning agents:\s*(.+)", text)
    agents = [a.strip() for a in agents_line.group(1).split(",")] if agents_line else []

    def extract_int(pattern: str) -> int:
        match = re.search(pattern, text)
        return int(match.group(1)) if match else 0

    v = extract_int(r"Secret user reserve price \(V\):\s*(\d+)")
    p = extract_int(r"Winning bid total price \(P\):\s*(\d+)")
    c = extract_int(r"Total cost for agents \(C\):\s*(\d+)")

    metric_1_match = re.search(r"Metric 1:.*=\s*([\d.eE+-]+)", text)
    metric_2_match = re.search(r"Metric 2:.*=\s*([\d.eE+-]+)", text)
    metric_1 = float(metric_1_match.group(1)) if metric_1_match else 0.0
    metric_2 = float(metric_2_match.group(1)) if metric_2_match else 0.0

    splits: List[AgentSplit] = []
    agent_blocks = re.split(r"\nAgent \[", text)
    for block in agent_blocks[1:]:
        agent_match = re.match(r"(\S+)\] \(Ingredients\):", block)
        if not agent_match:
            continue
        agent = agent_match.group(1)

        json_match = re.search(r"\{[\s\S]*?\}", block)
        ingredients: Dict[str, int] = {}
        if json_match:
            try:
                raw = json.loads(json_match.group(0))
                ingredients = {str(k): int(v) for k, v in raw.items()}
            except (json.JSONDecodeError, ValueError, TypeError):
                ingredients = {}

        order_sum = extract_int_from_block(block, r"Total sum for ingredients in order:\s*(\d+)")
        shop_sum = extract_int_from_block(block, r"Total sum for these ingredients in shop_sku:\s*(\d+)")
        margin_match = re.search(r"Margin:\s*([\d.eE+-]+)%", block)
        margin_pct = float(margin_match.group(1)) if margin_match else 0.0

        splits.append(
            AgentSplit(
                agent=agent,
                ingredients=ingredients,
                order_sum=order_sum,
                shop_sum=shop_sum,
                margin_pct=margin_pct,
            )
        )

    return WinningBidResult(
        agents=agents,
        v=v,
        p=p,
        c=c,
        metric_1=metric_1,
        metric_2=metric_2,
        splits=splits,
        timestamp=timestamp,
    )


def extract_int_from_block(block: str, pattern: str) -> int:
    match = re.search(pattern, block)
    return int(match.group(1)) if match else 0


def load_dialogues(dialogues_dir: Path) -> List[DialogueResult]:
    if not dialogues_dir.exists():
        return []
    results = []
    for path in sorted(dialogues_dir.glob("dialogue_start_*.txt")):
        try:
            results.append(parse_dialogue(path))
        except Exception as exc:
            print(f"Warning: failed to parse dialogue {path}: {exc}", file=sys.stderr)
    return results


def load_winning_bids(auction_logs_dir: Path) -> List[WinningBidResult]:
    if not auction_logs_dir.exists():
        return []
    results = []
    for path in sorted(auction_logs_dir.glob("winning_bid_split_*.txt")):
        try:
            results.append(parse_winning_bid(path))
        except Exception as exc:
            print(f"Warning: failed to parse winning bid {path}: {exc}", file=sys.stderr)
    return results


# ---------------------------------------------------------------------------
# Machine-readable JSON record loading (preferred source)
# ---------------------------------------------------------------------------


def _record_to_dialogue_result(record: Dict[str, Any]) -> DialogueResult:
    messages: List[Tuple[str, str, str, str]] = []
    for msg in record.get("messages", []):
        role = msg.get("role", "")
        agent = msg.get("agent", "")
        action_type = msg.get("type", "Unknown")
        content = msg.get("content", "")
        messages.append((role, agent, action_type, content))
    return DialogueResult(
        initiator=record.get("initiator", ""),
        contragent=record.get("contragent", ""),
        messages=messages,
        outcome_text=OUTCOME_TYPES.get(record.get("outcome", "unknown"), ""),
        outcome=record.get("outcome", "unknown"),
        timestamp=record.get("timestamp", ""),
        extra=record,
    )


def _record_to_winning_bid(record: Dict[str, Any]) -> WinningBidResult:
    splits: List[AgentSplit] = []
    margins = record.get("margins_pct", {})
    for agent, ingredients in record.get("splits", {}).items():
        ingr = {str(k): int(v) for k, v in ingredients.items()}
        order_sum = sum(ingr.values())
        margin_pct = float(margins.get(agent, 0.0))
        splits.append(
            AgentSplit(
                agent=agent,
                ingredients=ingr,
                order_sum=order_sum,
                shop_sum=0,
                margin_pct=margin_pct,
            )
        )
    return WinningBidResult(
        agents=record.get("agents", []),
        v=int(record.get("v", 0)),
        p=int(record.get("p", 0)),
        c=int(record.get("c", 0)),
        metric_1=float(record.get("metric_1", 0.0)),
        metric_2=float(record.get("metric_2", 0.0)),
        splits=splits,
        timestamp=record.get("timestamp", ""),
        extra=record,
    )


def load_dialogue_records(dialogues_dir: Path) -> List[DialogueResult]:
    if not dialogues_dir.exists():
        return []
    results = []
    for path in sorted(dialogues_dir.glob("dialogue_record_*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            results.append(_record_to_dialogue_result(record))
        except Exception as exc:
            print(f"Warning: failed to parse dialogue record {path}: {exc}", file=sys.stderr)
    return results


def load_bid_records(auction_logs_dir: Path) -> List[WinningBidResult]:
    if not auction_logs_dir.exists():
        return []
    results = []
    for path in sorted(auction_logs_dir.glob("bid_record_*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            results.append(_record_to_winning_bid(record))
        except Exception as exc:
            print(f"Warning: failed to parse bid record {path}: {exc}", file=sys.stderr)
    return results


def load_data(
    dialogues_dir: Path, auction_logs_dir: Path
) -> Tuple[List[DialogueResult], List[WinningBidResult]]:
    """Prefer JSON records; fall back to legacy text logs."""
    dialogues = load_dialogue_records(dialogues_dir)
    bids = load_bid_records(auction_logs_dir)
    if not dialogues:
        dialogues = load_dialogues(dialogues_dir)
    if not bids:
        bids = load_winning_bids(auction_logs_dir)
    return dialogues, bids


def load_run_meta() -> Optional[Dict[str, Any]]:
    """Load the run metadata file written by ProposalBoardAgent.setup()."""
    base_dir = Path(__file__).resolve().parent
    meta_dir = base_dir / "artifacts" / "metadata"
    if not meta_dir.exists():
        return None
    paths = sorted(meta_dir.glob("run_meta_*.json"))
    if not paths:
        return None
    newest = max(paths, key=lambda p: p.stat().st_mtime)
    try:
        return json.loads(newest.read_text(encoding="utf-8"))
    except Exception:
        return None
    paths = sorted(meta_dir.glob("run_meta_*.json"))
    if not paths:
        return None
    # If multiple meta files exist, return the newest by mtime.
    newest = max(paths, key=lambda p: p.stat().st_mtime)
    try:
        return json.loads(newest.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def gini_coefficient(values: List[float]) -> float:
    """Return the Gini coefficient of a list of values."""
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2 or arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    cumsum = np.cumsum(arr)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


def bootstrap_ci(values: List[float], n_boot: int = 5000, ci: float = 0.95) -> Tuple[float, float, float]:
    """Return (mean, lower, upper) bootstrap percentile confidence interval."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed=42)
    samples = rng.choice(arr, size=(n_boot, len(arr)), replace=True)
    means = samples.mean(axis=1)
    alpha = (1 - ci) / 2
    lower, upper = np.percentile(means, [alpha * 100, (1 - alpha) * 100])
    return float(arr.mean()), float(lower), float(upper)


# ---------------------------------------------------------------------------
# Global / agent metrics
# ---------------------------------------------------------------------------


def compute_global_metrics(
    dialogues: List[DialogueResult], bids: List[WinningBidResult]
) -> Dict[str, Any]:
    total_dialogues = len(dialogues)
    successful_dialogues = sum(1 for d in dialogues if d.outcome == "success")
    refused_dialogues = sum(1 for d in dialogues if d.outcome == "refused")
    failed_dialogues = total_dialogues - successful_dialogues - refused_dialogues
    failure_rate = failed_dialogues / total_dialogues if total_dialogues else 0.0
    refused_rate = refused_dialogues / total_dialogues if total_dialogues else 0.0

    avg_dialogue_length = (
        sum(d.length for d in dialogues) / total_dialogues if total_dialogues else 0.0
    )
    avg_length_success = (
        sum(d.length for d in dialogues if d.outcome == "success") / successful_dialogues
        if successful_dialogues else 0.0
    )
    avg_length_refused = (
        sum(d.length for d in dialogues if d.outcome == "refused") / refused_dialogues
        if refused_dialogues else 0.0
    )
    avg_length_failed = (
        sum(d.length for d in dialogues if d.outcome not in ("success", "refused"))
        / failed_dialogues
        if failed_dialogues else 0.0
    )

    success_rounds = [d.agreement_round for d in dialogues if d.outcome == "success" and d.agreement_round is not None]
    avg_agreement_round = sum(success_rounds) / len(success_rounds) if success_rounds else 0.0

    successful_user_purchases = len(bids)

    all_margins: List[float] = []
    for bid in bids:
        for split in bid.splits:
            all_margins.append(split.margin_pct)
    avg_agent_margin_pct = sum(all_margins) / len(all_margins) if all_margins else 0.0
    std_agent_margin_pct = float(np.std(all_margins, ddof=0)) if all_margins else 0.0
    gini_margin = gini_coefficient(all_margins)

    avg_v = sum(b.v for b in bids) / len(bids) if bids else 0.0
    avg_p = sum(b.p for b in bids) / len(bids) if bids else 0.0
    avg_c = sum(b.c for b in bids) / len(bids) if bids else 0.0

    user_surplus = []
    agent_efficiency = []
    welfare = []
    for b in bids:
        if b.v > 0:
            user_surplus.append((b.v - b.p) / b.v)
        if b.p > 0:
            agent_efficiency.append((b.p - b.c) / b.p)
        welfare.append(b.v - b.c)

    avg_user_surplus = sum(user_surplus) / len(user_surplus) if user_surplus else 0.0
    avg_agent_efficiency = sum(agent_efficiency) / len(agent_efficiency) if agent_efficiency else 0.0
    avg_welfare = sum(welfare) / len(welfare) if welfare else 0.0

    # Trust review ratings from dialogue records
    review_ratings = []
    for d in dialogues:
        review = d.extra.get("system_review")
        if review and isinstance(review, dict):
            rating = review.get("rating")
            if rating is not None:
                review_ratings.append(int(rating))
    avg_review_rating = sum(review_ratings) / len(review_ratings) if review_ratings else 0.0

    # System trust board average rating (if persisted)
    system_rating = 0.0
    system_board_path = Path("data/memory/system_trust_board.json")
    if system_board_path.exists():
        try:
            board = json.loads(system_board_path.read_text(encoding="utf-8"))
            ratings = []
            for target, reviews in board.items():
                for r in reviews:
                    if isinstance(r, dict) and r.get("rating") is not None:
                        ratings.append(int(r["rating"]))
            system_rating = sum(ratings) / len(ratings) if ratings else 0.0
        except Exception:
            system_rating = 0.0

    return {
        "avg_dialogue_length": round(avg_dialogue_length, 2),
        "avg_length_success": round(avg_length_success, 2),
        "avg_length_refused": round(avg_length_refused, 2),
        "avg_length_failed": round(avg_length_failed, 2),
        "avg_agreement_round": round(avg_agreement_round, 2),
        "total_dialogues": total_dialogues,
        "successful_dialogues": successful_dialogues,
        "refused_dialogues": refused_dialogues,
        "failed_dialogues": failed_dialogues,
        "refused_rate": round(refused_rate, 4),
        "failure_rate": round(failure_rate, 4),
        "successful_user_purchases": successful_user_purchases,
        "avg_agent_margin_pct": round(avg_agent_margin_pct, 2),
        "std_agent_margin_pct": round(std_agent_margin_pct, 2),
        "gini_agent_margin": round(gini_margin, 3),
        "avg_v": round(avg_v, 2),
        "avg_p": round(avg_p, 2),
        "avg_c": round(avg_c, 2),
        "avg_user_surplus": round(avg_user_surplus, 4),
        "avg_agent_efficiency": round(avg_agent_efficiency, 4),
        "avg_welfare": round(avg_welfare, 2),
        "avg_review_rating": round(avg_review_rating, 2),
        "avg_system_rating": round(system_rating, 2),
    }


def default_agent_entry() -> dict:
    return {
        "winning_bid_count": 0,
        "margins": [],
        "initiated_dialogues": 0,
        "received_dialogues": 0,
        "successful_dialogues": 0,
        "dialogue_lengths": [],
        "refused_negotiations": 0,
        "negotiation_decisions": 0,
        "agreement_rounds": [],
    }


def compute_agent_metrics(
    dialogues: List[DialogueResult], bids: List[WinningBidResult]
) -> Dict[str, dict]:
    agent_stats: Dict[str, dict] = {}

    for dialogue in dialogues:
        participants = {dialogue.initiator, dialogue.contragent}
        for agent in participants:
            agent_stats.setdefault(agent, default_agent_entry())

        agent_stats[dialogue.initiator]["initiated_dialogues"] += 1
        agent_stats[dialogue.contragent]["received_dialogues"] += 1

        for agent in participants:
            agent_stats[agent]["dialogue_lengths"].append(dialogue.length)

        if dialogue.outcome == "success":
            for agent in participants:
                agent_stats[agent]["successful_dialogues"] += 1
            if dialogue.agreement_round is not None:
                for agent in participants:
                    agent_stats[agent]["agreement_rounds"].append(dialogue.agreement_round)
        if dialogue.outcome == "refused":
            agent_stats[dialogue.contragent]["refused_negotiations"] += 1
            for agent in participants:
                agent_stats[agent]["negotiation_decisions"] += 1

    for bid in bids:
        for split in bid.splits:
            agent = split.agent
            agent_stats.setdefault(agent, default_agent_entry())
            agent_stats[agent]["winning_bid_count"] += 1
            agent_stats[agent]["margins"].append(split.margin_pct)

    result: Dict[str, dict] = {}
    for agent, stats in agent_stats.items():
        total_dialogues = stats["initiated_dialogues"] + stats["received_dialogues"]
        success_rate = (
            stats["successful_dialogues"] / total_dialogues if total_dialogues else 0.0
        )
        avg_length = (
            sum(stats["dialogue_lengths"]) / len(stats["dialogue_lengths"])
            if stats["dialogue_lengths"]
            else 0.0
        )
        avg_margin = (
            sum(stats["margins"]) / len(stats["margins"]) if stats["margins"] else 0.0
        )
        avg_agree_round = (
            sum(stats["agreement_rounds"]) / len(stats["agreement_rounds"])
            if stats["agreement_rounds"]
            else 0.0
        )
        result[agent] = {
            "winning_bid_count": stats["winning_bid_count"],
            "avg_margin_pct": round(avg_margin, 2),
            "initiated_dialogues": stats["initiated_dialogues"],
            "received_dialogues": stats["received_dialogues"],
            "successful_dialogues": stats["successful_dialogues"],
            "success_rate": round(success_rate, 4),
            "avg_dialogue_length": round(avg_length, 2),
            "avg_agreement_round": round(avg_agree_round, 2),
            "refused_negotiations": stats["refused_negotiations"],
            "negotiation_decisions": stats["negotiation_decisions"],
            "refusal_rate": round(
                stats["refused_negotiations"] / stats["received_dialogues"],
                4,
            ) if stats["received_dialogues"] else 0.0,
        }
    return result


# ---------------------------------------------------------------------------
# Per-report plots (styled, with new extras)
# ---------------------------------------------------------------------------


def _annotate_bars(ax, bars, fmt: str = "{:.2f}") -> None:
    for bar in bars:
        height = bar.get_height()
        if height > 0 or fmt != "{:.2f}":
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                fmt.format(height),
                ha="center",
                va="bottom",
                fontsize=9,
            )


def plot_dialogue_lengths(dialogues: List[DialogueResult], output_path: Path) -> None:
    lengths = [d.length for d in dialogues]
    if not lengths:
        lengths = [0]
    plt.figure(figsize=(8, 5))
    bins = np.arange(0, max(lengths + [10]) + 2) - 0.5
    _, _, patches = plt.hist(lengths, bins=bins, edgecolor="black", color="steelblue")
    for patch in patches:
        patch.set_alpha(0.85)
    plt.xlabel("Количество сообщений в диалоге")
    plt.ylabel("Число диалогов")
    plt.title("Распределение длины диалогов")
    plt.xticks(range(0, max(lengths + [10]) + 1))
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_round_agreement(dialogues: List[DialogueResult], output_path: Path) -> None:
    success_rounds: List[int] = []
    failed = 0
    refused = 0
    for d in dialogues:
        r = d.agreement_round
        if d.outcome == "success" and r is not None:
            success_rounds.append(r)
        elif d.outcome == "refused":
            refused += 1
        else:
            failed += 1

    max_round = max(success_rounds + [5])
    counts = [success_rounds.count(r) for r in range(1, max_round + 1)]
    labels = [str(r) for r in range(1, max_round + 1)] + ["refused"] + ["fail"]
    values = counts + [refused] + [failed]
    colors = ["steelblue"] * max_round + ["gold"] + ["crimson"]

    total = sum(values)
    failure_rate = failed / total if total else 0.0
    refused_rate = refused / total if total else 0.0

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color=colors, edgecolor="black")
    plt.xlabel("Раунд достижения соглашения")
    plt.ylabel("Число диалогов")
    plt.title(
        f"Динамика соглашения по раундам "
        f"(refused: {refused_rate:.1%}, неудач: {failure_rate:.1%})"
    )
    _annotate_bars(plt.gca(), bars, fmt="{:d}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_agent_margins(agent_metrics: Dict[str, dict], output_path: Path) -> None:
    agents = sorted(agent_metrics.keys())
    margins = [agent_metrics[a]["avg_margin_pct"] for a in agents]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(agents, margins, color="seagreen", edgecolor="black")
    plt.ylabel("Средняя маржа, %")
    plt.title("Средняя маржа агентов в победных сделках")
    plt.xticks(rotation=30, ha="right")
    _annotate_bars(plt.gca(), bars, fmt="{:.1f}%")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_agent_success_rates(agent_metrics: Dict[str, dict], output_path: Path) -> None:
    agents = sorted(agent_metrics.keys())
    rates = [agent_metrics[a]["success_rate"] * 100 for a in agents]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(agents, rates, color="coral", edgecolor="black")
    plt.ylabel("Доля успешных диалогов, %")
    plt.title("Доля успешных диалогов по агентам")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 110)
    _annotate_bars(plt.gca(), bars, fmt="{:.0f}%")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_winning_bids_overview(bids: List[WinningBidResult], output_path: Path) -> None:
    if not bids:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "Нет успешных сделок", ha="center", va="center")
        plt.axis("off")
        plt.savefig(output_path, dpi=200)
        plt.close()
        return

    x = np.arange(len(bids))
    v_values = [b.v for b in bids]
    p_values = [b.p for b in bids]
    c_values = [b.c for b in bids]

    width = 0.25
    plt.figure(figsize=(12, 5))
    plt.bar(x - width, v_values, width, label="V (max price)", color="steelblue")
    plt.bar(x, p_values, width, label="P (winning price)", color="seagreen")
    plt.bar(x + width, c_values, width, label="C (agents cost)", color="coral")

    plt.xlabel("Сделка")
    plt.ylabel("Цена")
    plt.title("Обзор успешных сделок: V, P, C")
    plt.xticks(x, [f"{i + 1}" for i in x])
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_outcome_distribution(
    dialogues: List[DialogueResult], output_path: Path, mechanism: Optional[str] = None
) -> None:
    outcomes = ["success", "refused", "timeout", "round_limit", "error", "incomplete"]
    counts = [sum(1 for d in dialogues if d.outcome == o) for o in outcomes]
    colors = ["seagreen", "gold", "coral", "mediumpurple", "crimson", "lightgray"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(outcomes, counts, color=colors, edgecolor="black")
    title = "Распределение исходов диалогов"
    if mechanism:
        title += f" ({mechanism})"
    plt.title(title)
    plt.xlabel("Исход")
    plt.ylabel("Число диалогов")
    _annotate_bars(plt.gca(), bars, fmt="{:d}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_agreement_rounds(dialogues: List[DialogueResult], output_path: Path) -> None:
    rounds = [d.agreement_round for d in dialogues if d.outcome == "success" and d.agreement_round is not None]
    plt.figure(figsize=(8, 5))
    if rounds:
        bins = np.arange(0.5, max(rounds) + 1.5)
        plt.hist(rounds, bins=bins, color="steelblue", edgecolor="black", alpha=0.85)
        plt.xticks(range(1, max(rounds) + 1))
    plt.xlabel("Раунд соглашения")
    plt.ylabel("Число успешных диалогов")
    plt.title("Распределение раундов достижения соглашения")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_vpc_per_bid(bids: List[WinningBidResult], output_path: Path) -> None:
    if not bids:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "Нет успешных сделок", ha="center", va="center")
        plt.axis("off")
        plt.savefig(output_path, dpi=200)
        plt.close()
        return

    x = np.arange(1, len(bids) + 1)
    v = [b.v for b in bids]
    p = [b.p for b in bids]
    c = [b.c for b in bids]
    plt.figure(figsize=(12, 5))
    plt.plot(x, v, marker="o", label="V", color="steelblue")
    plt.plot(x, p, marker="s", label="P", color="seagreen")
    plt.plot(x, c, marker="^", label="C", color="coral")
    plt.xlabel("Сделка")
    plt.ylabel("Цена")
    plt.title("Динамика V, P, C по сделкам")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Aggregate / comparison plots with confidence intervals
# ---------------------------------------------------------------------------


def _sorted_reports(reports: List[dict]) -> List[dict]:
    """Sort reports by timestamp if available, otherwise keep input order."""

    def _key(r: dict) -> str:
        ts = r.get("run_meta", {}).get("timestamp") or r.get("global", {}).get("timestamp") or ""
        return ts

    return sorted(reports, key=_key)


def _label_to_reports(reports: List[dict]) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {}
    for r in reports:
        label = r.get("label", "unknown")
        groups.setdefault(label, []).append(r)
    for label in groups:
        groups[label] = _sorted_reports(groups[label])
    return groups


def _metric_series(groups: Dict[str, List[dict]], metric: str, subkey: str = "global") -> Dict[str, List[float]]:
    return {label: [r[subkey].get(metric, 0.0) for r in runs] for label, runs in groups.items()}


def plot_main_comparison(
    reports: List[dict],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    multiply: float = 1.0,
) -> None:
    """Line plot per mechanism with a constant bootstrap-CI ribbon."""
    groups = _label_to_reports(reports)
    labels = sorted(groups.keys(), key=_mechanism_order)
    colors = _palette(len(labels))
    color_map = dict(zip(labels, colors))

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    for label in labels:
        values = [r["global"].get(metric, 0.0) * multiply for r in groups[label]]
        x = np.arange(1, len(values) + 1)
        mean, lower, upper = bootstrap_ci(values)
        color = color_map[label]
        ax.plot(x, values, marker="o", label=label, color=color, linewidth=2)
        if not (np.isnan(lower) or np.isnan(upper)):
            ax.fill_between(
                x,
                lower,
                upper,
                color=color,
                alpha=0.18,
            )

    ax.set_xlabel("Номер прогона")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Механизм доверия", loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_aggregate_bars(
    reports: List[dict],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    multiply: float = 1.0,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """Bar chart per mechanism with bootstrap-CI error bars."""
    groups = _label_to_reports(reports)
    labels = sorted(groups.keys(), key=_mechanism_order)
    means = []
    errors_lower = []
    errors_upper = []
    for label in labels:
        values = [r["global"].get(metric, 0.0) * multiply for r in groups[label]]
        mean, lower, upper = bootstrap_ci(values)
        means.append(mean)
        errors_lower.append(mean - lower)
        errors_upper.append(upper - mean)

    x = np.arange(len(labels))
    colors = _palette(len(labels))

    plt.figure(figsize=(8, 5))
    bars = plt.bar(x, means, color=colors, edgecolor="black", yerr=[errors_lower, errors_upper], capsize=5)
    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim:
        plt.ylim(*ylim)
    _annotate_bars(plt.gca(), bars, fmt="{:.2f}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_outcome_distribution_by_mechanism(reports: List[dict], output_path: Path) -> None:
    groups = _label_to_reports(reports)
    labels = sorted(groups.keys(), key=_mechanism_order)
    outcomes = ["success", "refused", "timeout", "round_limit", "error", "incomplete"]
    colors = ["seagreen", "gold", "coral", "mediumpurple", "crimson", "lightgray"]

    values = {o: [] for o in outcomes}
    for label in labels:
        # Sum across runs of this mechanism
        totals = {o: 0 for o in outcomes}
        for r in groups[label]:
            g = r["global"]
            totals["success"] += g.get("successful_dialogues", 0)
            totals["refused"] += g.get("refused_dialogues", 0)
            # Global metrics do not break failed into sub-types; put all failed into 'error'
            totals["error"] += g.get("failed_dialogues", 0)
        for o in outcomes:
            values[o].append(totals[o])

    x = np.arange(len(labels))
    width = 0.6
    bottom = np.zeros(len(labels))
    plt.figure(figsize=(9, 5))
    for o, color in zip(outcomes, colors):
        plt.bar(x, values[o], width, label=o, bottom=bottom, color=color, edgecolor="black")
        bottom += np.array(values[o])
    plt.xticks(x, labels)
    plt.ylabel("Число диалогов")
    plt.title("Распределение исходов по механизмам доверия")
    plt.legend(title="Исход", loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_surplus_vs_margin(reports: List[dict], output_path: Path) -> None:
    """Scatter of user surplus vs average agent margin per run, colored by mechanism."""
    groups = _label_to_reports(reports)
    labels = sorted(groups.keys(), key=_mechanism_order)
    colors = _palette(len(labels))
    color_map = dict(zip(labels, colors))

    plt.figure(figsize=(8, 6))
    for label in labels:
        xs = []
        ys = []
        for r in groups[label]:
            g = r["global"]
            xs.append(g.get("avg_user_surplus", 0.0) * 100)
            ys.append(g.get("avg_agent_margin_pct", 0.0))
        plt.scatter(xs, ys, color=color_map[label], label=label, s=80, alpha=0.8, edgecolor="black")

    plt.xlabel("Средняя пользовательская выгода (V-P)/V, %")
    plt.ylabel("Средняя маржа агентов, %")
    plt.title("Пользовательская выгода vs маржа агентов")
    plt.legend(title="Механизм")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_trust_evolution(reports: List[dict], output_path: Path) -> None:
    """Average system/review rating per run, per mechanism."""
    groups = _label_to_reports(reports)
    labels = sorted(groups.keys(), key=_mechanism_order)
    colors = _palette(len(labels))
    color_map = dict(zip(labels, colors))

    plt.figure(figsize=(10, 6))
    for label in labels:
        x = []
        y = []
        for i, r in enumerate(groups[label], start=1):
            x.append(i)
            # Prefer review rating; fall back to system board rating
            rating = r["global"].get("avg_review_rating", 0.0)
            if rating == 0.0:
                rating = r["global"].get("avg_system_rating", 0.0)
            y.append(rating)
        plt.plot(x, y, marker="o", label=label, color=color_map[label], linewidth=2)

    plt.xlabel("Номер прогона")
    plt.ylabel("Средний рейтинг доверия")
    plt.title("Эволюция среднего рейтинга доверия по прогонам")
    plt.legend(title="Механизм")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 5.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_agent_heatmap(reports: List[dict], output_path: Path) -> None:
    """Heatmap of success_rate × avg_margin_pct × refusal_rate per agent per mechanism."""
    if not HAS_PANDAS:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "pandas required for heatmap", ha="center", va="center")
        plt.axis("off")
        plt.savefig(output_path, dpi=200)
        plt.close()
        return

    rows = []
    for r in reports:
        mechanism = r.get("label", "unknown")
        for agent, metrics in r.get("agents", {}).items():
            rows.append(
                {
                    "mechanism": mechanism,
                    "agent": agent,
                    "success_rate": metrics.get("success_rate", 0.0) * 100,
                    "margin": metrics.get("avg_margin_pct", 0.0),
                    "refusal_rate": metrics.get("refusal_rate", 0.0) * 100,
                }
            )
    if not rows:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "No agent data", ha="center", va="center")
        plt.axis("off")
        plt.savefig(output_path, dpi=200)
        plt.close()
        return

    df = pd.DataFrame(rows)
    grouped = df.groupby(["mechanism", "agent"]).mean().reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    metrics = [("success_rate", "Success rate, %"), ("margin", "Margin, %"), ("refusal_rate", "Refusal rate, %")]
    for ax, (col, title) in zip(axes, metrics):
        pivot = grouped.pivot(index="agent", columns="mechanism", values=col)
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", ax=ax, vmin=0)
        ax.set_title(title)
        ax.set_xlabel("Механизм")
        ax.set_ylabel("Агент")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_radar(reports: List[dict], output_path: Path) -> None:
    """Radar chart comparing mechanisms across normalized metrics."""
    groups = _label_to_reports(reports)
    labels = sorted(groups.keys(), key=_mechanism_order)
    metrics = [
        ("successful_user_purchases", "Покупки"),
        ("avg_user_surplus", "Выгода пользователя"),
        ("avg_agent_efficiency", "Эффективность агентов"),
        ("avg_agent_margin_pct", "Маржа"),
        ("avg_review_rating", "Рейтинг доверия"),
    ]

    # Normalize each metric to 0..1 across mechanisms (using mean per mechanism)
    means = {label: [] for label in labels}
    raw_values: Dict[str, List[float]] = {m: [] for m, _ in metrics}
    for metric, _ in metrics:
        for label in labels:
            vals = [r["global"].get(metric, 0.0) for r in groups[label]]
            mean = sum(vals) / len(vals) if vals else 0.0
            raw_values[metric].append(mean)

    normalized: Dict[str, List[float]] = {label: [] for label in labels}
    for metric, _ in metrics:
        arr = np.array(raw_values[metric], dtype=float)
        min_v, max_v = arr.min(), arr.max()
        if max_v - min_v < 1e-9:
            norm = [0.5] * len(labels)
        else:
            norm = ((arr - min_v) / (max_v - min_v)).tolist()
        for i, label in enumerate(labels):
            normalized[label].append(norm[i])

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = _palette(len(labels))
    for label, color in zip(labels, colors):
        values = normalized[label]
        values += values[:1]
        ax.plot(angles, values, label=label, color=color, linewidth=2)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_title("Сравнение механизмов доверия (нормализовано)", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Report generation and aggregation
# ---------------------------------------------------------------------------


def archive_run_data(label: str, output_dir: Path) -> None:
    """Move all artifacts into backup_artifacts/<label>_<run_id> and clean artifacts dir."""
    base_dir = Path(__file__).resolve().parent
    backup_dir = _archive_artifacts(base_dir, label)
    print(f"Archived run data to {backup_dir}")


def _archive_artifacts(base_dir: Path, label: str) -> Path:
    """Move all artifacts into backup_artifacts/<label>_<run_id> and clean artifacts dir."""
    import shutil

    run_id = ""
    meta_dir = base_dir / "artifacts" / "metadata"
    if meta_dir.exists():
        paths = sorted(meta_dir.glob("run_meta_*.json"))
        if paths:
            try:
                run_id = json.loads(paths[-1].read_text(encoding="utf-8")).get("run_id", "")
            except Exception:
                pass
    if not run_id:
        run_id = datetime.now().strftime("%d-%H-%M-%S-%f")

    artifacts = base_dir / "artifacts"
    backup_root = base_dir / "backup_artifacts"
    backup_dir = backup_root / f"{label}_{run_id}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for sub in ["dialogues", "auction_logs", "data/memory", "metadata"]:
        src = artifacts / sub
        if not src.exists():
            continue
        dst = backup_dir / sub
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            target = dst / item.name
            if item.is_file():
                shutil.move(str(item), str(target))
                moved += 1
            elif item.is_dir():
                shutil.move(str(item), str(target))
                moved += 1

    logger.info("Archived %d artifact items from %s to %s", moved, artifacts, backup_dir)
    return backup_dir


def generate_report(
    label: str, dialogues_dir: Path, auction_logs_dir: Path, output_dir: Path
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    dialogues, bids = load_data(dialogues_dir, auction_logs_dir)

    global_metrics = compute_global_metrics(dialogues, bids)
    agent_metrics = compute_agent_metrics(dialogues, bids)

    run_meta = load_run_meta() or {}

    report = {
        "label": label,
        "timestamp": run_meta.get("timestamp") or (dialogues[0].timestamp if dialogues else ""),
        "run_meta": run_meta,
        "global": global_metrics,
        "agents": agent_metrics,
    }

    report_path = output_dir / f"report_{label}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_dialogue_lengths(dialogues, output_dir / "dialogue_length_hist.png")
    plot_round_agreement(dialogues, output_dir / "round_agreement_hist.png")
    plot_agent_margins(agent_metrics, output_dir / "agent_margin_bar.png")
    plot_agent_success_rates(agent_metrics, output_dir / "agent_success_rate_bar.png")
    plot_winning_bids_overview(bids, output_dir / "winning_bids_overview.png")
    plot_outcome_distribution(dialogues, output_dir / "outcome_distribution.png", mechanism=label)
    plot_agreement_rounds(dialogues, output_dir / "agreement_round_hist.png")
    plot_vpc_per_bid(bids, output_dir / "vpc_per_bid.png")

    print(f"Report saved to {report_path}")
    print(f"Plots saved to {output_dir}")
    return report


def aggregate_reports(report_paths: List[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    reports: List[dict] = []
    for path in report_paths:
        with open(path, "r", encoding="utf-8") as f:
            reports.append(json.load(f))

    # Summary table with CI
    groups = _label_to_reports(reports)
    labels = sorted(groups.keys(), key=_mechanism_order)
    summary_rows = []
    metrics_for_summary = [
        ("successful_user_purchases", 1.0),
        ("avg_user_surplus", 100.0),
        ("avg_agent_margin_pct", 1.0),
        ("avg_agent_efficiency", 100.0),
        ("avg_welfare", 1.0),
        ("avg_dialogue_length", 1.0),
        ("avg_agreement_round", 1.0),
        ("refused_rate", 100.0),
        ("failure_rate", 100.0),
        ("avg_review_rating", 1.0),
    ]
    for label in labels:
        row = {"mechanism": label, "runs": len(groups[label])}
        for metric, mult in metrics_for_summary:
            values = [r["global"].get(metric, 0.0) * mult for r in groups[label]]
            mean, lower, upper = bootstrap_ci(values)
            row[f"{metric}_mean"] = round(mean, 3)
            row[f"{metric}_ci_lower"] = round(lower, 3)
            row[f"{metric}_ci_upper"] = round(upper, 3)
        summary_rows.append(row)

    if HAS_PANDAS:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_dir / "summary_table.csv", index=False)
    else:
        with open(output_dir / "summary_table.json", "w", encoding="utf-8") as f:
            json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    # Main comparison line plots with CI ribbons
    main_metrics = [
        ("successful_user_purchases", "Успешные покупки", "Покупок"),
        ("avg_user_surplus", "Пользовательская выгода", "(V-P)/V, %", 100.0),
        ("avg_agent_margin_pct", "Средняя маржа агентов", "Маржа, %"),
        ("avg_agent_efficiency", "Эффективность агентов", "(P-C)/P, %", 100.0),
        ("avg_dialogue_length", "Средняя длина диалогов", "Сообщений"),
        ("avg_agreement_round", "Средний раунд соглашения", "Раунд"),
        ("refused_rate", "Доля отказов", "%", 100.0),
        ("failure_rate", "Доля неудач", "%", 100.0),
        ("avg_review_rating", "Средний рейтинг доверия", "Звёзды"),
    ]
    for item in main_metrics:
        metric, title, ylabel = item[0], item[1], item[2]
        mult = item[3] if len(item) > 3 else 1.0
        filename = f"main_{metric}.png"
        plot_main_comparison(reports, metric, ylabel, title, output_dir / filename, multiply=mult)

    # Aggregate bar charts with CI error bars
    bar_metrics = [
        ("successful_user_purchases", "Среднее число успешных покупок", "Покупок"),
        ("avg_user_surplus", "Средняя пользовательская выгода", "(V-P)/V, %", 100.0),
        ("avg_agent_margin_pct", "Средняя маржа агентов", "Маржа, %"),
        ("avg_agent_efficiency", "Средняя эффективность агентов", "(P-C)/P, %", 100.0),
        ("avg_welfare", "Среднее общее благосостояние", "V - C"),
        ("avg_dialogue_length", "Средняя длина диалогов", "Сообщений"),
        ("avg_agreement_round", "Средний раунд соглашения", "Раунд"),
        ("refused_rate", "Доля отказов", "%", 100.0, (0, 110)),
        ("failure_rate", "Доля неудач", "%", 100.0, (0, 110)),
        ("avg_review_rating", "Средний рейтинг доверия", "Звёзды", 1.0, (0, 5.5)),
    ]
    for item in bar_metrics:
        metric, title, ylabel = item[0], item[1], item[2]
        mult = item[3] if len(item) > 3 else 1.0
        ylim = item[4] if len(item) > 4 else None
        filename = f"aggregate_{metric}.png"
        plot_aggregate_bars(
            reports, metric, ylabel, title, output_dir / filename, multiply=mult, ylim=ylim
        )

    # New comparison charts
    plot_outcome_distribution_by_mechanism(reports, output_dir / "outcome_by_mechanism.png")
    plot_surplus_vs_margin(reports, output_dir / "surplus_vs_margin.png")
    plot_trust_evolution(reports, output_dir / "trust_evolution.png")
    plot_agent_heatmap(reports, output_dir / "agent_heatmap.png")
    plot_radar(reports, output_dir / "radar_comparison.png")

    combined_path = output_dir / "aggregate_report.json"
    combined_path.write_text(json.dumps({"reports": reports}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Aggregate report saved to {combined_path}")
    print(f"Summary and aggregate plots saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate auction metrics from logs.")
    parser.add_argument("--dialogues", type=Path, default=Path("dialogues"),
                        help="Directory with dialogue_start_*.txt / dialogue_record_*.json files")
    parser.add_argument("--auction-logs", type=Path, default=Path("auction_logs"),
                        help="Directory with winning_bid_split_*.txt / bid_record_*.json files")
    parser.add_argument("--output", type=Path, default=Path("output"),
                        help="Output directory for reports and plots")
    parser.add_argument("--label", type=str, default="run",
                        help="Label for this run/mechanism")
    parser.add_argument("--archive", action="store_true",
                        help="Archive artifacts into backup_artifacts/<label>_<run_id> and clean artifacts dir")
    parser.add_argument("--aggregate", nargs="+", type=Path, default=None,
                        help="Aggregate report JSON files into comparison plots")

    args = parser.parse_args()

    if args.aggregate:
        aggregate_reports(args.aggregate, args.output)
    else:
        generate_report(args.label, args.dialogues, args.auction_logs, args.output)
        if args.archive:
            archive_run_data(label=args.label, output_dir=args.output)


if __name__ == "__main__":
    main()
