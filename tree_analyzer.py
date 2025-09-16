#!/usr/bin/env python3
"""
Fairy-Stockfish Tree analyzer - Local Version
Analyze trees and saves them in opening books for chess variants using Fairy-Stockfish engine.
"""

import os
import sys
import json
import argparse
import subprocess
import heapq
from itertools import count
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field, asdict, fields

@dataclass
class Position:
    fen: str
    best_move: Optional[str] = None
    best_child_fen: Optional[str] = None
    eval_cp: Optional[int] = None
    mate_in: Optional[int] = None
    moves_to_children: List[str] = field(default_factory=list)
    children_fens: List[str] = field(default_factory=list)
    parent_fens: List[str] = field(default_factory=list)
    status: str = "unknown"
    proof_number: Optional[int] = 1
    disproof_number: Optional[int] = 1
    visits: int = 0

class FairyStockfishEngine:
    def __init__(self, engine_path: str, variant: str, nnue_path: str,
                 threads: int, hash_size: int, multipv: int, variants_ini: str = None):
        """Initialize Fairy-Stockfish engine with given parameters."""
        self.process = subprocess.Popen(
            [engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=0
        )
        self._terminated = False

        self.send("uci")
        self.receive("uciok")

        if variants_ini and Path(variants_ini).exists():
            self.send(f"load {os.path.abspath(variants_ini)}")

        self.send(f"setoption name UCI_Variant value {variant}")
        if nnue_path and Path(nnue_path).exists():
            nnue_abs = os.path.abspath(nnue_path)
            self.send(f"setoption name EvalFile value {nnue_abs}")
            self.send("setoption name Use NNUE value true")
        self.send(f"setoption name Threads value {threads}")
        self.send(f"setoption name Hash value {hash_size}")
        self.send(f"setoption name MultiPV value {multipv}")

        self.multipv_count = multipv

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.quit()

    def send(self, command: str):
        """Send command to engine."""
        if self.process.poll() is None and self.process.stdin:
            self.process.stdin.write(f"{command}\n")
            self.process.stdin.flush()

    def receive(self, terminator: Optional[str]) -> List[str]:
        """Receive output from engine until terminator is found."""
        lines: List[str] = []
        while True:
            raw = self.process.stdout.readline()
            if not raw:
                if self.process.poll() is not None:
                    break
                continue
            line = raw.strip()
            lines.append(line)
            if terminator and terminator in line:
                break
            if not terminator and not line:
                break
        return lines

    def get_fen(self, position_cmd: str = "position startpos") -> str:
        """Get FEN string for current position."""
        self.send(position_cmd)
        self.send("d")
        fen: Optional[str] = None
        for _ in range(64):
            raw = self.process.stdout.readline()
            if not raw:
                if self.process.poll() is not None:
                    break
                continue
            line = raw.strip()
            if not line and fen:
                break
            if "Fen:" in line:
                fen = line.split("Fen:", 1)[1].strip()
        if fen:
            return fen
        raise RuntimeError(f"Unable to read FEN from engine response for: {position_cmd}")

    def multipv(self, fen: str, depth: int) -> List[Tuple[str, Optional[int], Optional[int]]]:
        """Analyze position with MultiPV."""
        self.send(f"position fen {fen}")
        self.send(f"go depth {depth}")
        pv_data: Dict[int, Tuple[str, Optional[int], Optional[int]]] = {}

        for line in self.receive("bestmove"):
            if line.startswith("info") and "depth" in line and "multipv" in line:
                result = self.parse(line)
                if result:
                    multipv_num, move, eval_cp, mate_in = result
                    pv_data[multipv_num] = (move, eval_cp, mate_in)

        results: List[Tuple[str, Optional[int], Optional[int]]] = []
        for i in range(1, self.multipv_count + 1):
            if i in pv_data:
                results.append(pv_data[i])
            else:
                break
        return results

    def parse(self, line: str) -> Optional[Tuple[int, str, Optional[int], Optional[int]]]:
        """Parse UCI info line."""
        parts = line.split()
        try:
            multipv_idx = parts.index("multipv")
            multipv_num = int(parts[multipv_idx + 1])
            pv_idx = parts.index("pv")
            move = parts[pv_idx + 1]

            eval_cp = None
            mate_in = None

            if "score" in parts:
                score_idx = parts.index("score")
                if parts[score_idx + 1] == "cp":
                    eval_cp = int(parts[score_idx + 2])
                elif parts[score_idx + 1] == "mate":
                    mate_in = int(parts[score_idx + 2])

            return multipv_num, move, eval_cp, mate_in
        except (ValueError, IndexError):
            return None

    def quit(self):
        """Quit engine."""
        if self._terminated:
            return
        self._terminated = True
        if self.process.poll() is None:
            self.send("quit")
            self.process.terminate()
            self.process.wait()

    def close(self):
        self.quit()
class BookBuilder:
    def __init__(self, config: dict):
        """Initialize book builder with configuration."""
        self.config = config
        self.engine = FairyStockfishEngine(
            engine_path=config['engine_path'],
            variant=config['variant'],
            nnue_path=config.get('nnue_path'),
            threads=config['threads'],
            hash_size=config['hash'],
            multipv=config['multipv'],
            variants_ini=config.get('variants_ini')
        )

        self.positions: Dict[str, Position] = {}
        self.analyzed_count = 0
        self.root_fen: Optional[str] = None

        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.output_dir / f"{config['variant']}_book_{config['depth']}.log"
        self.json_path = self.output_dir / f"{config['variant']}_book_{config['depth']}.json"
        self.epd_path = self.output_dir / f"{config['variant']}_book_{config['depth']}.epd"
        self.log = open(self.log_path, 'a')

        self.queue: List[Tuple[int, int, str]] = []
        self.in_queue: Set[str] = set()
        self.queue_order = count()
        self.current_fen: Optional[str] = None

        self.load_existing_data()

    def load_existing_data(self):
        """Load existing analysis data if available."""
        if not self.json_path.exists():
            self.initialize_new_book()
            return

        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
        except Exception:
            self.initialize_new_book()
            return

        self.root_fen = data.get('root_fen')
        self.analyzed_count = data.get('analyzed_count', 0)

        positions_payload = data.get('positions', {})
        for fen, payload in positions_payload.items():
            self.positions[fen] = self._position_from_payload(fen, payload)

        if not self.root_fen:
            self.initialize_new_book()
        else:
            self._synchronise_parent_links()
            queue_payload = data.get('queue') or []
            for fen in queue_payload:
                self.enqueue(fen)
            if not self.queue:
                for fen, pos in self.positions.items():
                    if pos.status == "unknown" and not pos.children_fens:
                        self.enqueue(fen)

    def _position_from_payload(self, fen: str, payload: Dict[str, Any]) -> Position:
        template = Position(fen=fen)
        values: Dict[str, Any] = {}
        for field in fields(Position):
            if field.name == 'fen':
                continue
            values[field.name] = payload.get(field.name, getattr(template, field.name))
        pos = Position(fen=fen, **values)
        pos.moves_to_children = list(pos.moves_to_children)
        pos.children_fens = list(pos.children_fens)
        pos.parent_fens = [p for p in pos.parent_fens if p]
        pos.parent_fens = list(dict.fromkeys(pos.parent_fens))
        return pos

    def _synchronise_parent_links(self):
        for pos in self.positions.values():
            for child_fen in list(pos.children_fens):
                child = self.positions.get(child_fen)
                if not child:
                    continue
                if pos.fen not in child.parent_fens:
                    child.parent_fens.append(pos.fen)
        for pos in self.positions.values():
            valid_parents = []
            for parent_fen in pos.parent_fens:
                parent = self.positions.get(parent_fen)
                if parent and pos.fen in parent.children_fens:
                    valid_parents.append(parent_fen)
            pos.parent_fens = valid_parents

    def initialize_new_book(self):
        """Initialize new book from starting position."""
        self.root_fen = self.engine.get_fen()
        root = Position(fen=self.root_fen)
        self.positions[self.root_fen] = root
        self.enqueue(self.root_fen)

    def enqueue(self, fen: str):
        pos = self.positions.get(fen)
        if not pos or fen in self.in_queue:
            return
        if pos.status != "unknown" or pos.children_fens:
            return
        priority = pos.proof_number if isinstance(pos.proof_number, int) and pos.proof_number is not None else 1
        heapq.heappush(self.queue, (priority, next(self.queue_order), fen))
        self.in_queue.add(fen)

    def select_next_position(self) -> Optional[str]:
        while self.queue:
            _, _, fen = heapq.heappop(self.queue)
            if fen not in self.in_queue:
                continue
            self.in_queue.remove(fen)
            pos = self.positions.get(fen)
            if pos and pos.status == "unknown":
                return fen
        return None

    def analyze(self) -> bool:
        fen = self.select_next_position()
        if fen is None:
            return False
        self.current_fen = fen
        self._analyze_fen(fen)
        self.current_fen = None
        return True

    def _analyze_fen(self, fen: str):
        pos = self.positions[fen]
        pos.visits += 1

        analysis = self.engine.multipv(fen, self.config['depth'])
        self._log_analysis(fen, analysis)

        if not analysis:
            pos.best_move = None
            pos.best_child_fen = None
            if pos.mate_in is None:
                pos.eval_cp = pos.eval_cp or 0
            self.update_proof_numbers(fen)
            self.propagate_to_parents(fen)
            return

        old_children = set(pos.children_fens)
        pos.children_fens = []
        pos.moves_to_children = []
        pos.best_child_fen = None

        new_children: Set[str] = set()
        for move, eval_cp, mate_in in analysis:
            child_fen = self.engine.get_fen(f"position fen {fen} moves {move}")
            pos.children_fens.append(child_fen)
            pos.moves_to_children.append(move)
            new_children.add(child_fen)

            child = self.positions.get(child_fen)
            if not child:
                child = Position(fen=child_fen)
                self.positions[child_fen] = child
            if fen not in child.parent_fens:
                child.parent_fens.append(fen)

            if mate_in is None:
                child.mate_in = None
                child.eval_cp = -eval_cp if eval_cp is not None else child.eval_cp
            else:
                child.eval_cp = None
                if mate_in > 0:
                    child.mate_in = -mate_in + 1
                else:
                    child.mate_in = -mate_in

            if not child.children_fens and child.status == "unknown":
                self.enqueue(child_fen)

        for removed_child in old_children - new_children:
            child = self.positions.get(removed_child)
            if child and fen in child.parent_fens:
                child.parent_fens = [p for p in child.parent_fens if p != fen]

        self.minimax(fen)
        self.update_proof_numbers(fen)
        self.propagate_to_parents(fen)

    def _log_analysis(self, fen: str, analysis: List[Tuple[str, Optional[int], Optional[int]]]):
        header = [
            "",
            f"Analysis #{self.analyzed_count + 1}",
            f"fen {fen}",
            f"pending {len(self.in_queue)}"
        ]
        for i, (move, eval_cp, mate_in) in enumerate(analysis, 1):
            if mate_in is not None:
                header.append(f"alt{i} {move} mate {mate_in}")
            elif eval_cp is not None:
                header.append(f"alt{i} {move} cp {eval_cp}")
            else:
                header.append(f"alt{i} {move}")
        text = '\n'.join(header)
        print(text)
        self.log.write(text + '\n')
        self.log.flush()

    def _child_is_better(self, candidate: Position, current: Position) -> bool:
        ce, be = candidate.eval_cp, current.eval_cp
        cm, bm = candidate.mate_in, current.mate_in

        if cm is not None and bm is not None:
            if cm * bm > 0:
                return cm > bm
            return cm < 0 <= bm
        if cm is not None and bm is None:
            return cm < 0
        if cm is None and bm is not None:
            return bm > 0
        if ce is not None and be is not None:
            return ce < be
        if ce is not None and be is None:
            return True
        return False

    def minimax(self, fen: str):
        pos = self.positions[fen]
        if not pos.children_fens:
            pos.best_move = None
            pos.best_child_fen = None
            return

        best_index = 0
        best_child_fen = pos.children_fens[0]
        best_child = self.positions[best_child_fen]

        for idx, child_fen in enumerate(pos.children_fens[1:], 1):
            child = self.positions[child_fen]
            if self._child_is_better(child, best_child):
                best_child = child
                best_child_fen = child_fen
                best_index = idx

        pos.best_move = pos.moves_to_children[best_index]
        pos.best_child_fen = best_child_fen

        if best_child.mate_in is None:
            pos.mate_in = None
            if best_child.eval_cp is not None:
                if best_child.eval_cp > 0:
                    pos.eval_cp = -best_child.eval_cp + 1
                elif best_child.eval_cp < 0:
                    pos.eval_cp = -best_child.eval_cp - 1
                else:
                    pos.eval_cp = 0
            else:
                pos.eval_cp = None
        else:
            pos.eval_cp = None
            if best_child.mate_in > 0:
                pos.mate_in = -best_child.mate_in
            else:
                pos.mate_in = -best_child.mate_in + 1

    def update_proof_numbers(self, fen: str):
        pos = self.positions[fen]
        if pos.mate_in is not None:
            if pos.mate_in < 0:
                pos.status = "proven_win"
                pos.proof_number = 0
                pos.disproof_number = None
            elif pos.mate_in > 0:
                pos.status = "proven_loss"
                pos.proof_number = None
                pos.disproof_number = 0
            else:
                pos.status = "proven_draw"
                pos.proof_number = None
                pos.disproof_number = None
            return

        if not pos.children_fens:
            pos.status = "unknown"
            pos.proof_number = 1
            pos.disproof_number = 1
            return

        child_proofs = []
        child_disproofs = []
        infinite_proof = False
        infinite_disproof = False
        for child_fen in pos.children_fens:
            child = self.positions[child_fen]
            if child.proof_number is None:
                infinite_proof = True
            else:
                child_proofs.append(child.proof_number)
            if child.disproof_number is None:
                infinite_disproof = True
            else:
                child_disproofs.append(child.disproof_number)

        pos.status = "unknown"
        if infinite_proof and not child_proofs:
            pos.proof_number = None
        else:
            pos.proof_number = min(child_proofs) if child_proofs else 1
        if infinite_disproof and not child_disproofs:
            pos.disproof_number = None
        else:
            pos.disproof_number = sum(child_disproofs) if child_disproofs else 1

    def propagate_to_parents(self, fen: str):
        pos = self.positions[fen]
        stack = list(pos.parent_fens)
        seen: Set[str] = set()
        while stack:
            parent_fen = stack.pop()
            if parent_fen in seen:
                continue
            seen.add(parent_fen)
            if parent_fen not in self.positions:
                continue
            self.minimax(parent_fen)
            self.update_proof_numbers(parent_fen)
            stack.extend(self.positions[parent_fen].parent_fens)

    def pending_queue_snapshot(self) -> List[str]:
        return [fen for _, _, fen in self.queue if fen in self.in_queue]

    def export(self):
        """Export book to JSON and EPD formats."""
        with open(self.json_path, 'w') as f:
            positions_dict = {
                fen: {k: v for k, v in asdict(pos).items() if k != 'fen'}
                for fen, pos in self.positions.items()
            }
            json.dump({
                'root_fen': self.root_fen,
                'analyzed_count': self.analyzed_count,
                'positions': positions_dict,
                'queue': self.pending_queue_snapshot()
            }, f, indent=2)

        with open(self.epd_path, 'w') as f:
            for fen, pos in self.positions.items():
                epd = fen.rsplit(' ', 2)[0] if ' ' in fen else fen
                if pos.best_move:
                    epd += f" bm {pos.best_move};"
                if pos.mate_in is not None:
                    epd += f" dm {abs(pos.mate_in)};"
                elif pos.eval_cp is not None:
                    epd += f" ce {pos.eval_cp};"
                f.write(epd + "\n")

    def build(self):
        """Build opening book."""
        try:
            while True:
                if not self.analyze():
                    print("\nQueue exhausted. Nothing left to analyze.")
                    break
                self.analyzed_count += 1
                if self.analyzed_count % 10 == 0:
                    self.export()
        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving current state...")
        except Exception as e:
            print(f"\nError: {e}")
        finally:
            self.export()
            if hasattr(self, 'log') and self.log:
                self.log.close()
            if hasattr(self, 'engine') and self.engine:
                self.engine.quit()
def main():
    parser = argparse.ArgumentParser(description='Build opening book for Fairy-Stockfish variants')
    
    # Required arguments
    parser.add_argument('engine_path', help='Path to Fairy-Stockfish executable')
    parser.add_argument('variant', help='Chess variant (e.g., atomic, crazyhouse, etc.)')
    
    # Optional arguments
    parser.add_argument('--depth', type=int, default=25, help='Search depth (default: 25)')
    parser.add_argument('--multipv', type=int, default=3, help='Number of best moves to consider (default: 3)')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads (default: 4)')
    parser.add_argument('--hash', type=int, default=8192, help='Hash table size in MB (default: 8192)')
    parser.add_argument('--output-dir', default='.', help='Output directory (default: current directory)')
    parser.add_argument('--nnue', dest='nnue_path', help='Path to NNUE file for the variant')
    parser.add_argument('--variants-ini', help='Path to variants.ini configuration file (optional)')
    
    args = parser.parse_args()
    
    # Convert paths to absolute
    engine_path = Path(args.engine_path).resolve()
    
    # Check if engine exists
    if not engine_path.exists():
        print(f"Error: Engine not found at {engine_path}")
        sys.exit(1)
    
    # Check if NNUE file exists (if provided)
    if args.nnue_path:
        nnue_path = Path(args.nnue_path).resolve()
        if nnue_path.exists():
            args.nnue_path = str(nnue_path)
        else:
            args.nnue_path = None
    
    # Check variants.ini (optional)
    if args.variants_ini:
        variants_path = Path(args.variants_ini).resolve()
        if variants_path.exists():
            args.variants_ini = str(variants_path)
        else:
            args.variants_ini = None
    
    # Create configuration
    config = {
        'engine_path': str(engine_path),
        'variant': args.variant,
        'depth': args.depth,
        'multipv': args.multipv,
        'threads': args.threads,
        'hash': args.hash,
        'output_dir': args.output_dir,
        'nnue_path': args.nnue_path,
        'variants_ini': args.variants_ini
    }
    
    # Build book
    builder = BookBuilder(config)
    builder.build()

if __name__ == '__main__':
    main()
