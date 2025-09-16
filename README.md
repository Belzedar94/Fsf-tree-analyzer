# Fsf-tree-analyzer

Early tooling around Fairy-Stockfish aimed at persistent tree analysis for chess variants. The current iteration grows from Wolyn (moky_92399 on Discord)'s original book builder prototype.

## Current State
- `tree_analyzer.py` maintains a persistent game-tree with parent/child links, proof/disproof counters, visit counts, and a priority queue of unresolved nodes.
- The analyzer expands positions by priority (loosely following proof-number heuristics), logs each batch, and persists JSON/EPD checkpoints along with the pending frontier for resume/distribution.
- Node status tracks proven wins/losses/draws via mate scores, and leaf expansion is idempotent (no duplicate child entries).

## Running The Script
1. Compile Fairy-Stockfish and locate the engine binary plus the relevant NNUE/`variants.ini` files.
2. Invoke: `python tree_analyzer.py <path-to-fairy-stockfish> atomic --depth 25 --multipv 3 --threads 4` (adjust depth, multipv, and hash for your hardware).
3. Outputs land next to the script unless `--output-dir` is provided; JSON contains the queue so workers can resume where they left off.

## Data Model & Checkpointing
- Each FEN stores children, parent references, proof/disproof numbers (`None` stands for infinity), visit counts, and a status flag (`unknown`, `proven_win`, `proven_loss`, `proven_draw`).
- The queue snapshot in the JSON file lists unresolved leaf nodes to be expanded; workers can hand out these FENs to remote machines for distributed solving.
- Logs are appended per analysis, and checkpoints are flushed every ten nodes or on exit.

## Remaining Gaps
- Proof/disproof updates still use simple heuristics; integrate full PN/DFPN formulas with color-to-move awareness and trophy thresholds.
- Terminal detection relies on engine mate scores; add rules for stalemate, repetition, and variant-specific end conditions.
- Introduce a coordinator for multi-worker usage (leases, heartbeats, deduplication) and an optional SQLite backend for robustness.
- Harden UCI handling with timeouts, retries, and richer telemetry (per-move stats, fail reasons).
- Slot in policy controls (e.g., depth-per-node scheduling, evaluation thresholds) to better target critical branches.

## Credits
- Original book builder scaffold by Wolyn (moky_92399 on Discord); extended here to support persistent queues and solver-oriented metadata.
