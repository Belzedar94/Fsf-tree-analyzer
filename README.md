# Fsf-tree-analyzer

Early tooling around Fairy-Stockfish aimed at persistent tree analysis for chess variants. The current iteration grows from Wolyn (moky_92399 on Discord)'s original book builder prototype.

## Current State
- `tree_analyzer.py` maintains a persistent game-tree with parent/child links, visit counts, and proof/disproof counters that respect the side-to-move (OR/AND) structure.
- Leaf handling recognises terminal nodes via missing best moves (mate/stalemate) and prunes/resets children atomically; checkpoints keep both the solved state and the remaining frontier.
- Nodes flush to JSON/EPD regularly, and the in-memory priority queue prefers the smallest proof/disproof combination for unresolved leaves.

## Running The Script
1. Compile Fairy-Stockfish and locate the engine binary plus the relevant NNUE/`variants.ini` files.
2. Invoke: `python tree_analyzer.py <path-to-fairy-stockfish> atomic --depth 25 --multipv 3 --threads 4 --proof-threshold 0 --disproof-threshold 0` (adjust depth, multipv, hash, and thresholds for your run).
3. Outputs land next to the script unless `--output-dir` is provided; JSON captures the pending queue so workers can resume where they left off.

## Proof-Number Controls
- `--proof-threshold` marks a node as a proven win once its proof number falls at, or below, the given value (use `-1` to disable the shortcut).
- `--disproof-threshold` does the same for proven losses via the disproof number.
- Combined with the OR/AND propagation, this lets you steer PN/DP search aggressiveness without rebuilding the state file.

## Status Visualizer
- `visualize_status.py` summarises a JSON dump (`python visualize_status.py <state.json> --top 5`).
- Add `--dot solver.dot --max-nodes 150` to emit a Graphviz view (nodes are coloured by status, edges labelled with moves); render with `dot -Tpng solver.dot -o solver.png`.
- The CLI also surfaces the smallest proof/disproof pairs so you can target stubborn subtrees manually.

## Data Model & Checkpointing
- Each FEN stores children, parent references, proof/disproof numbers (`None` stands for infinity), visit counts, and a status flag (`unknown`, `proven_win`, `proven_loss`, `proven_draw`).
- The queue snapshot in the JSON file lists unresolved leaf nodes to be expanded; workers can hand out these FENs to remote machines for distributed solving.
- Logs are appended per analysis, and checkpoints are flushed every ten nodes or on exit.

## Remaining Gaps
- Transposition handling and hashing are still missing; duplicate subtrees will be re-analysed unless deduplicated upstream.
- Distributed coordination (leases, heartbeats, multi-worker inventory) remains manual; consider an external controller or SQLite-backed job queue.
- UCI communications are synchronous; adding timeouts/retries and richer telemetry will make long runs more robust.
- Exploration policy is still depth-first by PN ordering; plugging a full DFPN or iterative widening policy would reduce redundant work on wide branches.

## Credits
- Original book builder scaffold by Wolyn (moky_92399 on Discord); extended here to support persistent queues, solver-oriented metadata, and visualisation tooling.
