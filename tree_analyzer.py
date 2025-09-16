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
import math
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


@dataclass
class PVLine:
    move: str
    eval_cp: Optional[int]
    mate_in: Optional[int]

@dataclass
class AnalysisResult:
    lines: List[PVLine]
    bestmove: Optional[str]
    ponder: Optional[str]
    raw_bestmove: Optional[str] = None
    legal_moves: Optional[List[str]] = None
    in_check: Optional[bool] = None
class FairyStockfishEngine:
    def __init__(self, engine_path: str, variant: str, nnue_path: Optional[str],
                 threads: int, hash_size: int, multipv: Any, variants_ini: str = None):
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

        self.multipv_mode = multipv
        if isinstance(multipv, int) and multipv > 0:
            self.multipv_count = multipv
        else:
            self.multipv_mode = 'auto'
            self.multipv_count = 1
        self.send(f"setoption name MultiPV value {self.multipv_count}")

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

    def parse_bestmove_line(self, line: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        if not line.startswith('bestmove'):
            return None, None, None
        parts = line.split()
        best = parts[1] if len(parts) > 1 else None
        ponder = None
        if best in {None, '(none)', '0000'}:
            best = None
        if 'ponder' in parts:
            idx = parts.index('ponder')
            if idx + 1 < len(parts):
                ponder = parts[idx + 1]
        return best, ponder, line

    def get_fen(self, position_cmd: str = "position startpos") -> str:
        """Get FEN string for current position."""
        self.send(position_cmd)
        self.send("d")
        fen: Optional[str] = None
        for _ in range(128):
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

    def legal_moves(self, fen: str) -> Tuple[List[str], bool]:
        """Return all legal moves and whether side to move is in check."""
        self.send(f"position fen {fen}")
        self.send("d")
        moves: List[str] = []
        in_check = False
        collecting = False
        for _ in range(256):
            raw = self.process.stdout.readline()
            if not raw:
                if self.process.poll() is not None:
                    break
                continue
            line = raw.strip()
            if line.startswith("Checkers:"):
                in_check = line.split(":", 1)[1].strip() not in {"-", "", "none"}
                continue
            if line.startswith("Legal moves:"):
                collecting = True
                remainder = line.split("Legal moves:", 1)[1].strip()
                if remainder:
                    moves.extend(remainder.split())
                continue
            if collecting:
                if not line:
                    break
                moves.extend(line.split())
        return moves, in_check
    def multipv(self, fen: str, depth: int) -> AnalysisResult:
        """Analyze position with MultiPV."""
        self.send(f"position fen {fen}")
        self.send(f"go depth {depth}")
        pv_data: Dict[int, Tuple[str, Optional[int], Optional[int]]] = {}

        lines = self.receive("bestmove")
        for line in lines:
            if line.startswith("info") and "depth" in line and "pv" in line:
                result = self.parse(line)
                if result:
                    multipv_num, move, eval_cp, mate_in = result
                    pv_data[multipv_num] = (move, eval_cp, mate_in)

        bestmove_line = lines[-1] if lines else ""
        bestmove, ponder, raw = self.parse_bestmove_line(bestmove_line)

        results: List[PVLine] = []
        limit = self.multipv_count if isinstance(self.multipv_count, int) else 1
        for i in range(1, limit + 1):
            if i in pv_data:
                move, eval_cp, mate_in = pv_data[i]
                results.append(PVLine(move=move, eval_cp=eval_cp, mate_in=mate_in))
            else:
                break
        return AnalysisResult(lines=results, bestmove=bestmove, ponder=ponder, raw_bestmove=raw)

    def analyze_move(self, fen: str, move: str, depth: int) -> PVLine:
        """Analyze a specific move using searchmoves."""
        self.send(f"position fen {fen}")
        self.send(f"go depth {depth} searchmoves {move}")
        chosen = PVLine(move=move, eval_cp=None, mate_in=None)
        for line in self.receive("bestmove"):
            if line.startswith("info") and "score" in line:
                result = self.parse(line)
                if result:
                    _, _, eval_cp, mate_in = result
                    chosen = PVLine(move=move, eval_cp=eval_cp, mate_in=mate_in)
        return chosen

    def analyze_all_moves(self, fen: str, depth: int) -> AnalysisResult:
        moves, in_check = self.legal_moves(fen)
        lines: List[PVLine] = []
        best: Optional[PVLine] = None
        for move in moves:
            line = self.analyze_move(fen, move, depth)
            lines.append(line)
            if best is None or self._line_score(line) > self._line_score(best):
                best = line
        best_move = best.move if best else (moves[0] if moves else None)
        return AnalysisResult(lines=lines, bestmove=best_move, ponder=None, legal_moves=moves, in_check=in_check)

    def parse(self, line: str) -> Optional[Tuple[int, Optional[str], Optional[int], Optional[int]]]:
        """Parse UCI info line."""
        parts = line.split()
        try:
            multipv_idx = parts.index("multipv")
            multipv_num = int(parts[multipv_idx + 1])
        except ValueError:
            multipv_num = 1
        try:
            pv_idx = parts.index("pv")
            move = parts[pv_idx + 1]
        except ValueError:
            move = None

        eval_cp = None
        mate_in = None

        if "score" in parts:
            score_idx = parts.index("score")
            if parts[score_idx + 1] == "cp":
                eval_cp = int(parts[score_idx + 2])
            elif parts[score_idx + 1] == "mate":
                mate_in = int(parts[score_idx + 2])

        return multipv_num, move, eval_cp, mate_in

    @staticmethod
    def _line_score(line: PVLine) -> Tuple[int, int]:
        if line.mate_in is not None:
            if line.mate_in > 0:
                return (2, -line.mate_in)
            return (0, line.mate_in)
        if line.eval_cp is not None:
            return (1, line.eval_cp)
        return (-1, -10_000_000)

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
        self.root_player: Optional[str] = None
        self.proof_threshold = config.get('proof_threshold', 0)
        self.disproof_threshold = config.get('disproof_threshold', 0)

        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.output_dir / f"{config['variant']}_book_{config['depth']}.log"
        self.json_path = self.output_dir / f"{config['variant']}_book_{config['depth']}.json"
        self.epd_path = self.output_dir / f"{config['variant']}_book_{config['depth']}.epd"
        self.log = open(self.log_path, 'a')
        self.stop_file = self.output_dir / 'STOP'

        self.queue: List[Tuple[float, float, int, str]] = []
        self.in_queue: Set[str] = set()
        self.queue_order = count()
        self.current_fen: Optional[str] = None
        self.stalemate_loss = bool(config.get('stalemate_loss', False))

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
        self.root_player = self._side_to_move(self.root_fen)
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

    def _side_to_move(self, fen: Optional[str]) -> str:
        if not fen:
            return "w"
        parts = fen.split()
        if len(parts) > 1:
            return parts[1]
        return "w"

    def _is_or_node(self, fen: str) -> bool:
        if not self.root_player:
            return True
        return self._side_to_move(fen) == self.root_player

    def _pn_value(self, value: Optional[int]) -> float:
        if value is None:
            return math.inf
        return float(value)

    def _depth_for(self, pos: Position) -> int:
        base = self.config.get('depth', 1)
        step = self.config.get('depth_step', 0)
        max_depth = self.config.get('max_depth', base)
        visits = max(0, pos.visits - 1)
        depth = base + step * visits
        return max(1, min(depth, max_depth))

    def _analysis_for(self, fen: str, depth: int) -> AnalysisResult:
        multipv = self.config.get('multipv')
        if isinstance(multipv, str) and multipv == "auto":
            return self.engine.analyze_all_moves(fen, depth)
        return self.engine.multipv(fen, depth)


    def _print_progress_start(self, fen: str, depth: int, visits: int):
        board = fen.split(' ', 1)[0] if ' ' in fen else fen
        if len(board) > 40:
            board = board[:40] + '...'
        message = (f"[{self.analyzed_count + 1}] depth {depth} queue {len(self.in_queue)} "
                   f"visits {visits} board {board}")
        print(message, flush=True)

    def _should_stop(self) -> bool:
        return self.stop_file.exists()

    def initialize_new_book(self):
        """Initialize new book from starting position."""
        self.root_fen = self.engine.get_fen()
        self.root_player = self._side_to_move(self.root_fen)
        root = Position(fen=self.root_fen)
        self.positions[self.root_fen] = root
        self.enqueue(self.root_fen)

    def enqueue(self, fen: str):
        pos = self.positions.get(fen)
        if not pos or fen in self.in_queue:
            return
        if pos.status != "unknown" or pos.children_fens:
            return
        proof = pos.proof_number if pos.proof_number is not None else math.inf
        disproof = pos.disproof_number if pos.disproof_number is not None else math.inf
        heapq.heappush(self.queue, (proof, disproof, next(self.queue_order), fen))
        self.in_queue.add(fen)

    def select_next_position(self) -> Optional[str]:
        while self.queue:
            _, _, _, fen = heapq.heappop(self.queue)
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

        depth = self._depth_for(pos)
        self._print_progress_start(fen, depth, pos.visits)
        analysis = self._analysis_for(fen, depth)
        if isinstance(self.config.get('multipv'), str) and self.config['multipv'] == 'auto' and not analysis.lines and analysis.bestmove is None:
            analysis = self.engine.multipv(fen, depth)
        self._log_analysis(fen, depth, analysis)

        if not analysis.lines and analysis.bestmove is None:
            moves = analysis.legal_moves if analysis.legal_moves is not None else None
            in_check = analysis.in_check if analysis.in_check is not None else False
            if moves is None:
                moves, in_check = self.engine.legal_moves(fen)
            if not moves:
                self._handle_no_moves_terminal(fen, pos, in_check)
            else:
                pos.best_move = None
                pos.best_child_fen = None
                pos.children_fens = []
                pos.moves_to_children = []
                pos.status = "unknown"
                if pos.eval_cp is None:
                    pos.eval_cp = 0
                pos.mate_in = None
            self.update_proof_numbers(fen)
            self.propagate_to_parents(fen)
            return

        if analysis.bestmove is None:
            self._handle_terminal(fen, pos, analysis)
            self.update_proof_numbers(fen)
            self.propagate_to_parents(fen)
            return

        lines = analysis.lines or ([] if not analysis.bestmove else [PVLine(move=analysis.bestmove, eval_cp=None, mate_in=None)])

        old_children = set(pos.children_fens)
        pos.children_fens = []
        pos.moves_to_children = []
        pos.best_child_fen = None

        new_children: Set[str] = set()
        for pv in lines:
            move = pv.move
            eval_cp = pv.eval_cp
            mate_in = pv.mate_in
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

        if analysis.bestmove:
            pos.best_move = analysis.bestmove
        elif pos.moves_to_children:
            pos.best_move = pos.moves_to_children[0]
        else:
            pos.best_move = None

        self.minimax(fen)
        self.update_proof_numbers(fen)
        self.propagate_to_parents(fen)

    def _handle_terminal(self, fen: str, pos: Position, analysis: AnalysisResult):
        pos.best_move = None
        pos.best_child_fen = None
        pos.children_fens = []
        pos.moves_to_children = []
        pos.status = "unknown"
        first = analysis.lines[0] if analysis.lines else None
        if first:
            pos.eval_cp = first.eval_cp if first.eval_cp is not None else 0
            if first.mate_in is None:
                pos.mate_in = 0
            else:
                pos.mate_in = first.mate_in
        else:
            if pos.eval_cp is None:
                pos.eval_cp = 0
            pos.mate_in = 0

    def _handle_no_moves_terminal(self, fen: str, pos: Position, in_check: bool):
        pos.best_move = None
        pos.best_child_fen = None
        pos.children_fens = []
        pos.moves_to_children = []
        pos.status = "unknown"
        if in_check or self.stalemate_loss:
            pos.mate_in = -1
            pos.eval_cp = None
        else:
            pos.mate_in = 0
            pos.eval_cp = 0

    def _log_analysis(self, fen: str, depth: int, analysis: AnalysisResult):
        header = [
            "",
            f"Analysis #{self.analyzed_count + 1}",
            f"fen {fen}",
            f"pending {len(self.in_queue)}"
        ]
        if analysis.raw_bestmove:
            header.append(analysis.raw_bestmove)
        lines = analysis.lines or ([] if not analysis.bestmove else [PVLine(move=analysis.bestmove, eval_cp=None, mate_in=None)])
        for i, line in enumerate(lines, 1):
            if line.mate_in is not None:
                header.append(f"alt{i} {line.move} mate {line.mate_in}")
            elif line.eval_cp is not None:
                header.append(f"alt{i} {line.move} cp {line.eval_cp}")
            else:
                header.append(f"alt{i} {line.move}")
        text = "\n".join(header)
        print(text)
        self.log.write(text + "\n")
        self.log.flush()

    def _child_is_better(self, candidate: Position, current: Position) -> bool:
        ce, be = candidate.eval_cp, current.eval_cp
        cm, bm = candidate.mate_in, current.mate_in

        if cm is not None and bm is not None:
            if cm * bm > 0:
                return cm > bm
            return cm <= 0 and 0 < bm
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
        is_or = self._is_or_node(fen)

        if pos.mate_in is not None:
            if pos.mate_in == 0:
                pos.status = "proven_draw"
                pos.proof_number = None
                pos.disproof_number = None
                return
            root_wins = (pos.mate_in > 0) if is_or else (pos.mate_in < 0)
            if root_wins:
                pos.status = "proven_win"
                pos.proof_number = 0
                pos.disproof_number = None
            else:
                pos.status = "proven_loss"
                pos.proof_number = None
                pos.disproof_number = 0
            return

        if not pos.children_fens:
            pos.status = "unknown"
            pos.proof_number = 1
            pos.disproof_number = 1
            return

        proof_values = [self._pn_value(self.positions[child].proof_number) for child in pos.children_fens]
        disproof_values = [self._pn_value(self.positions[child].disproof_number) for child in pos.children_fens]

        if is_or:
            proof_metric = min(proof_values) if proof_values else math.inf
            disproof_metric = sum(disproof_values) if disproof_values else math.inf
        else:
            proof_metric = sum(proof_values) if proof_values else math.inf
            disproof_metric = min(disproof_values) if disproof_values else math.inf

        pos.proof_number = None if math.isinf(proof_metric) else int(proof_metric)
        pos.disproof_number = None if math.isinf(disproof_metric) else int(disproof_metric)
        pos.status = "unknown"

        if self.proof_threshold >= 0 and pos.proof_number is not None and pos.proof_number <= self.proof_threshold:
            pos.status = "proven_win"
            pos.disproof_number = None
            return
        if self.disproof_threshold >= 0 and pos.disproof_number is not None and pos.disproof_number <= self.disproof_threshold:
            pos.status = "proven_loss"
            pos.proof_number = None
            return

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
        return [fen for _, _, _, fen in self.queue if fen in self.in_queue]

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
                if self._should_stop():
                    print("\nStop file detected. Halting after current export.")
                    break
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
    parser.add_argument('--depth', type=int, default=25, help='Base search depth (default: 25)')
    parser.add_argument('--max-depth', type=int, default=None, help='Maximum depth per node (defaults to --depth)')
    parser.add_argument('--depth-step', type=int, default=0, help='Depth increment after each visit (default: 0)')
    parser.add_argument('--multipv', default='auto', help="Number of best moves to consider (int) or 'auto' for every legal move")
    parser.add_argument('--stalemate-loss', action='store_true', help='Treat stalemate/no-move positions as a loss for the side to move')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads (default: 4)')
    parser.add_argument('--hash', type=int, default=8192, help='Hash table size in MB (default: 8192)')
    parser.add_argument('--output-dir', default='.', help='Output directory (default: current directory)')
    parser.add_argument('--nnue', dest='nnue_path', help='Path to NNUE file for the variant')
    parser.add_argument('--variants-ini', help='Path to variants.ini configuration file (optional)')
    parser.add_argument('--proof-threshold', type=int, default=0, help='Auto-mark wins when proof number <= threshold (use -1 to disable)')
    parser.add_argument('--disproof-threshold', type=int, default=0, help='Auto-mark losses when disproof number <= threshold (use -1 to disable)')
    
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
    
    max_depth = args.max_depth if args.max_depth is not None else args.depth
    if max_depth < args.depth:
        max_depth = args.depth
    multipv_arg = args.multipv
    if isinstance(multipv_arg, str):
        try:
            multipv_value = int(multipv_arg)
        except ValueError:
            lower = multipv_arg.lower()
            if lower in ('auto', 'all'):
                multipv_value = 'auto'
            else:
                print(f"Error: invalid multipv value {multipv_arg}")
                sys.exit(1)
    else:
        multipv_value = multipv_arg
    if isinstance(multipv_value, int) and multipv_value <= 0:
        multipv_value = 'auto'
    # Create configuration
    config = {
        'engine_path': str(engine_path),
        'variant': args.variant,
        'depth': args.depth,
        'max_depth': max_depth,
        'depth_step': args.depth_step,
        'multipv': multipv_value,
        'threads': args.threads,
        'hash': args.hash,
        'output_dir': args.output_dir,
        'nnue_path': args.nnue_path,
        'variants_ini': args.variants_ini,
        'proof_threshold': args.proof_threshold,
        'disproof_threshold': args.disproof_threshold,
        'stalemate_loss': args.stalemate_loss
    }
    
    # Build book
    builder = BookBuilder(config)
    builder.build()

if __name__ == '__main__':
    main()

