#!/usr/bin/env python3
"""Calculate auction metrics from dialogue logs and winning bid logs."""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

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


@dataclass
class DialogueResult:
    initiator: str
    contragent: str
    messages: List[Tuple[str, str, str, str]] = field(default_factory=list)
    """List of (role, agent, action_type, content)."""
    outcome_text: str = ""
    outcome: str = "unknown"
    timestamp: str = ""

    @property
    def length(self) -> int:
        return len(self.messages)

    @property
    def agreement_round(self) -> Optional[int]:
        """Return round number (1-based) where opponent sent ShopList.

        One round is a completed exchange of one message from each agent.
        The opponent's final ShopList is always at an even message index,
        so the round number is the message index divided by 2.
        """
        for idx, (role, _agent, action_type, _content) in enumerate(self.messages, start=1):
            if role == "Opponent" and "продавец соглашается" in action_type:
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

        splits.append(AgentSplit(
            agent=agent,
            ingredients=ingredients,
            order_sum=order_sum,
            shop_sum=shop_sum,
            margin_pct=margin_pct,
        ))

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


def compute_global_metrics(dialogues: List[DialogueResult], bids: List[WinningBidResult]) -> dict:
    total_dialogues = len(dialogues)
    successful_dialogues = sum(1 for d in dialogues if d.outcome == "success")
    failed_dialogues = total_dialogues - successful_dialogues
    failure_rate = failed_dialogues / total_dialogues if total_dialogues else 0.0

    avg_dialogue_length = (
        sum(d.length for d in dialogues) / total_dialogues if total_dialogues else 0.0
    )

    successful_user_purchases = len(bids)

    all_margins: List[float] = []
    for bid in bids:
        for split in bid.splits:
            all_margins.append(split.margin_pct)
    avg_agent_margin_pct = sum(all_margins) / len(all_margins) if all_margins else 0.0

    avg_v = sum(b.v for b in bids) / len(bids) if bids else 0.0
    avg_p = sum(b.p for b in bids) / len(bids) if bids else 0.0
    avg_c = sum(b.c for b in bids) / len(bids) if bids else 0.0

    return {
        "avg_dialogue_length": round(avg_dialogue_length, 2),
        "total_dialogues": total_dialogues,
        "successful_dialogues": successful_dialogues,
        "failed_dialogues": failed_dialogues,
        "failure_rate": round(failure_rate, 4),
        "successful_user_purchases": successful_user_purchases,
        "avg_agent_margin_pct": round(avg_agent_margin_pct, 2),
        "avg_v": round(avg_v, 2),
        "avg_p": round(avg_p, 2),
        "avg_c": round(avg_c, 2),
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
        if dialogue.outcome == "refused":
            agent_stats[dialogue.contragent]["refused_negotiations"] += 1

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
        result[agent] = {
            "winning_bid_count": stats["winning_bid_count"],
            "avg_margin_pct": round(avg_margin, 2),
            "initiated_dialogues": stats["initiated_dialogues"],
            "received_dialogues": stats["received_dialogues"],
            "successful_dialogues": stats["successful_dialogues"],
            "success_rate": round(success_rate, 4),
            "avg_dialogue_length": round(avg_length, 2),
            "refused_negotiations": stats["refused_negotiations"],
        }
    return result


def plot_dialogue_lengths(dialogues: List[DialogueResult], output_path: Path) -> None:
    lengths = [d.length for d in dialogues]
    plt.figure(figsize=(8, 5))
    bins = np.arange(0, max(lengths + [10]) + 2) - 0.5
    plt.hist(lengths, bins=bins, edgecolor="black", color="steelblue")
    plt.xlabel("Количество сообщений в диалоге")
    plt.ylabel("Число диалогов")
    plt.title("Распределение длины диалогов")
    plt.xticks(range(0, max(lengths + [10]) + 1))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_round_agreement(dialogues: List[DialogueResult], output_path: Path) -> None:
    success_rounds: List[int] = []
    failed = 0
    for d in dialogues:
        r = d.agreement_round
        if d.outcome == "success" and r is not None:
            success_rounds.append(r)
        else:
            failed += 1

    max_round = max(success_rounds + [5])
    counts = [success_rounds.count(r) for r in range(1, max_round + 1)]
    labels = [str(r) for r in range(1, max_round + 1)] + ["fail"]
    values = counts + [failed]
    colors = ["steelblue"] * max_round + ["crimson"]

    total = sum(values)
    failure_rate = failed / total if total else 0.0

    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, values, color=colors, edgecolor="black")
    plt.xlabel("Раунд достижения соглашения")
    plt.ylabel("Число диалогов")
    plt.title(f"Динамика соглашения по раундам (неудач: {failure_rate:.1%})")

    for bar, val in zip(bars, values):
        if val > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(val),
                     ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_agent_margins(agent_metrics: Dict[str, dict], output_path: Path) -> None:
    agents = sorted(agent_metrics.keys())
    margins = [agent_metrics[a]["avg_margin_pct"] for a in agents]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(agents, margins, color="seagreen", edgecolor="black")
    plt.ylabel("Средняя маржа, %")
    plt.title("Средняя маржа агентов в победных сделках")
    plt.xticks(rotation=30, ha="right")

    for bar, val in zip(bars, margins):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.1f}%",
                 ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
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

    for bar, val in zip(bars, rates):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.0f}%",
                 ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_winning_bids_overview(bids: List[WinningBidResult], output_path: Path) -> None:
    if not bids:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "Нет успешных сделок", ha="center", va="center")
        plt.axis("off")
        plt.savefig(output_path, dpi=150)
        plt.close()
        return

    x = np.arange(len(bids))
    v_values = [b.v for b in bids]
    p_values = [b.p for b in bids]
    c_values = [b.c for b in bids]

    width = 0.25
    plt.figure(figsize=(10, 5))
    plt.bar(x - width, v_values, width, label="V (max price)", color="steelblue")
    plt.bar(x, p_values, width, label="P (winning price)", color="seagreen")
    plt.bar(x + width, c_values, width, label="C (agents cost)", color="coral")

    plt.xlabel("Сделка")
    plt.ylabel("Цена")
    plt.title("Обзор успешных сделок: V, P, C")
    plt.xticks(x, [f"{i + 1}" for i in x])
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_report(
    label: str, dialogues_dir: Path, auction_logs_dir: Path, output_dir: Path
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    dialogues = load_dialogues(dialogues_dir)
    bids = load_winning_bids(auction_logs_dir)

    global_metrics = compute_global_metrics(dialogues, bids)
    agent_metrics = compute_agent_metrics(dialogues, bids)

    report = {
        "label": label,
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

    print(f"Report saved to {report_path}")
    print(f"Plots saved to {output_dir}")
    return report


def aggregate_reports(report_paths: List[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: List[dict] = []
    for path in report_paths:
        with open(path, "r", encoding="utf-8") as f:
            reports.append(json.load(f))

    labels = [r["label"] for r in reports]
    avg_lengths = [r["global"]["avg_dialogue_length"] for r in reports]
    success_rates = [r["global"]["successful_dialogues"] / max(r["global"]["total_dialogues"], 1) * 100
                     for r in reports]
    purchases = [r["global"]["successful_user_purchases"] for r in reports]
    margins = [r["global"]["avg_agent_margin_pct"] for r in reports]
    failures = [r["global"]["failure_rate"] * 100 for r in reports]

    def save_bar(title: str, ylabel: str, values: List[float], filename: str, ylim: Optional[Tuple[float, float]] = None):
        plt.figure(figsize=(8, 5))
        bars = plt.bar(labels, values, color="steelblue", edgecolor="black")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(rotation=15, ha="right")
        if ylim:
            plt.ylim(*ylim)
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}",
                     ha="center", va="bottom")
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150)
        plt.close()

    save_bar("Средняя длина диалогов", "Сообщений", avg_lengths,
             "aggregate_avg_dialogue_length.png")
    save_bar("Доля успешных диалогов", "%", success_rates,
             "aggregate_success_rate.png", (0, 110))
    save_bar("Число успешных покупок", "Покупок", purchases,
             "aggregate_successful_purchases.png")
    save_bar("Средняя маржа агентов", "Маржа, %", margins,
             "aggregate_avg_margin.png")
    save_bar("Доля неудачных диалогов", "%", failures,
             "aggregate_failure_rate.png", (0, 110))

    combined_path = output_dir / "aggregate_report.json"
    combined_path.write_text(json.dumps({"reports": reports}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Aggregate report saved to {combined_path}")
    print(f"Aggregate plots saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate auction metrics from logs.")
    parser.add_argument("--dialogues", type=Path, default=Path("dialogues"),
                        help="Directory with dialogue_start_*.txt files")
    parser.add_argument("--auction-logs", type=Path, default=Path("auction_logs"),
                        help="Directory with winning_bid_split_*.txt files")
    parser.add_argument("--output", type=Path, default=Path("output"),
                        help="Output directory for reports and plots")
    parser.add_argument("--label", type=str, default="run",
                        help="Label for this run/mechanism")
    parser.add_argument("--aggregate", nargs="+", type=Path, default=None,
                        help="Aggregate report JSON files into comparison plots")

    args = parser.parse_args()

    if args.aggregate:
        aggregate_reports(args.aggregate, args.output)
    else:
        generate_report(args.label, args.dialogues, args.auction_logs, args.output)


if __name__ == "__main__":
    main()
