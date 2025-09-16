#!/usr/bin/env python3
"""Simple CLI visualizer for Fairy-Stockfish tree analyzer state."""

import argparse
import json
import math
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

STATUS_COLORS = {
    "proven_win": "#a6d96a",
    "proven_loss": "#f46d43",
    "proven_draw": "#bdbdbd",
    "unknown": "#fee08b",
}


def load_state(path: Path) -> Dict[str, Any]:
    with path.open("r") as handle:
        return json.load(handle)


def summarise(positions: Dict[str, Dict[str, Any]]) -> Counter:
    return Counter(pos.get("status", "unknown") for pos in positions.values())


def format_number(value: Any) -> str:
    if value is None:
        return "inf"
    return str(value)


def print_summary(state: Dict[str, Any], top: int) -> None:
    positions = state.get("positions", {})
    counts = summarise(positions)
    total = len(positions)
    queue = state.get("queue") or []

    print(f"Total positions: {total}")
    for status in ("proven_win", "proven_loss", "proven_draw", "unknown"):
        if status in counts:
            share = counts[status] / total * 100 if total else 0
            print(f"  {status:12s}: {counts[status]:6d} ({share:5.1f}%)")
    print(f"Frontier queued: {len(queue)}")

    if top <= 0:
        return

    unresolved = [
        (fen, pos)
        for fen, pos in positions.items()
        if pos.get("status", "unknown") == "unknown"
    ]
    unresolved.sort(
        key=lambda item: (
            item[1].get("proof_number", math.inf),
            item[1].get("disproof_number", math.inf),
            item[0],
        )
    )
    print("\nTop unresolved nodes:")
    for fen, pos in unresolved[:top]:
        print(
            f"  {fen} | proof={format_number(pos.get('proof_number'))}"
            f", disproof={format_number(pos.get('disproof_number'))}"
            f", visits={pos.get('visits', 0)}"
        )


def collect_nodes(state: Dict[str, Any], max_nodes: int) -> List[str]:
    positions = state.get("positions", {})
    root = state.get("root_fen")
    if not root or root not in positions:
        return []
    order: List[str] = []
    queue: deque[str] = deque([root])
    seen = set()
    while queue and (max_nodes < 0 or len(order) < max_nodes):
        fen = queue.popleft()
        if fen in seen or fen not in positions:
            continue
        seen.add(fen)
        order.append(fen)
        children = positions[fen].get("children_fens", [])
        for child in children:
            queue.append(child)
    return order


def escape_label(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\"", "\\\"")


def write_dot(state: Dict[str, Any], selected: Iterable[str], output: Path) -> None:
    positions = state.get("positions", {})
    selected = list(selected)
    id_map = {fen: f"n{i}" for i, fen in enumerate(selected)}
    with output.open("w") as handle:
        handle.write("digraph solver {\n")
        handle.write("  rankdir=LR;\n")
        for fen in selected:
            pos = positions.get(fen, {})
            node_id = id_map[fen]
            status = pos.get("status", "unknown")
            color = STATUS_COLORS.get(status, "#f0f0f0")
            proof = format_number(pos.get("proof_number"))
            disproof = format_number(pos.get("disproof_number"))
            label = escape_label(
                f"{fen}\n{status} | pn={proof} dn={disproof}\nvisits={pos.get('visits', 0)}"
            )
            style_bits = "style=filled" if status != "unknown" else "style=filled,dashed"
            handle.write(
                f"  {node_id} [{style_bits}, fillcolor=\"{color}\", label=\"{label}\"];\n"
            )

        for fen in selected:
            pos = positions.get(fen, {})
            node_id = id_map[fen]
            moves = pos.get("moves_to_children", [])
            children = pos.get("children_fens", [])
            for move, child in zip(moves, children):
                if child not in id_map:
                    continue
                child_id = id_map[child]
                handle.write(
                    f"  {node_id} -> {child_id} [label=\"{escape_label(move)}\"];\n"
                )
        handle.write("}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect solver progress")
    parser.add_argument("state", type=Path, help="Path to JSON state dump")
    parser.add_argument("--dot", type=Path, help="Optional path to Graphviz DOT output")
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=200,
        help="Maximum nodes to include in the DOT output (default: 200)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Display the top-N unresolved nodes (default: 10, 0 disables)",
    )
    args = parser.parse_args()

    state = load_state(args.state)
    print_summary(state, args.top)

    if args.dot:
        selected = collect_nodes(state, args.max_nodes)
        write_dot(state, selected, args.dot)
        print(f"DOT graph written to {args.dot}")


if __name__ == "__main__":
    main()
