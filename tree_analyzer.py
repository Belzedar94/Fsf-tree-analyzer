#!/usr/bin/env python3
"""
Fairy-Stockfish Book Builder - Local Version
Builds opening books for chess variants using Fairy-Stockfish engine.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict

@dataclass
class Position:
    fen: str
    best_move: Optional[str] = None
    best_child_fen: Optional[str] = None
    eval_cp: Optional[int] = None
    mate_in: Optional[int] = None
    moves_to_children: List[str] = field(default_factory=list)
    children_fens: List[str] = field(default_factory=list)

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
        
        self.send("uci")
        self.receive("uciok")
        
        # Load variants configuration if provided
        if variants_ini and Path(variants_ini).exists():
            self.send(f"load {os.path.abspath(variants_ini)}")
        
        # Configure engine
        self.send(f"setoption name UCI_Variant value {variant}")
        if nnue_path and Path(nnue_path).exists():
            nnue_abs = os.path.abspath(nnue_path)
            self.send(f"setoption name EvalFile value {nnue_abs}")
            self.send("setoption name Use NNUE value true")
        self.send(f"setoption name Threads value {threads}")
        self.send(f"setoption name Hash value {hash_size}")
        self.send(f"setoption name MultiPV value {multipv}")
        
        self.multipv_count = multipv

    def send(self, command: str):
        """Send command to engine."""
        if self.process.poll() is None:  # Check if process is still running
            self.process.stdin.write(f"{command}\n")
            self.process.stdin.flush()

    def receive(self, terminator: str) -> List[str]:
        """Receive output from engine until terminator is found."""
        lines = []
        while True:
            line = self.process.stdout.readline().strip()
            if not line and self.process.poll() is not None:
                break
            lines.append(line)
            if terminator in line:
                break
        return lines

    def get_fen(self, position_cmd: str = "position startpos") -> str:
        """Get FEN string for current position."""
        self.send(position_cmd)
        self.send("d")
        lines = self.receive("Checkers:")
        for line in lines:
            if "Fen: " in line:
                return line.split("Fen: ", 1)[1].strip()
        # Try alternative format (standard UCI)
        for line in lines:
            if line.startswith("Fen:"):
                return line.split("Fen:", 1)[1].strip()
        # Fallback for standard chess
        return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    def multipv(self, fen: str, depth: int) -> List[Tuple[str, Optional[int], Optional[int]]]:
        """Analyze position with MultiPV."""
        self.send(f"position fen {fen}")
        self.send(f"go depth {depth}")
        pv_data = {}
        
        for line in self.receive("bestmove"):
            if line.startswith("info") and "depth" in line and "multipv" in line:
                result = self.parse(line)
                if result:
                    multipv_num, move, eval_cp, mate_in = result
                    pv_data[multipv_num] = (move, eval_cp, mate_in)
        
        results = []
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
        if self.process.poll() is None:  # Check if process is still running
            self.send("quit")
            self.process.terminate()
            self.process.wait()  # Wait for process to actually terminate

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
        self.root_fen = None
        
        # Setup output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file paths
        self.log_path = self.output_dir / f"{config['variant']}_book_{config['depth']}.log"
        self.json_path = self.output_dir / f"{config['variant']}_book_{config['depth']}.json"
        self.epd_path = self.output_dir / f"{config['variant']}_book_{config['depth']}.epd"
        
        # Open log file
        self.log = open(self.log_path, 'a')
        
        # Load existing data if available
        self.load_existing_data()

    def load_existing_data(self):
        """Load existing analysis data if available."""
        if self.json_path.exists():
            try:
                with open(self.json_path, 'r') as f:
                    data = json.load(f)
                    self.root_fen = data['root_fen']
                    self.analyzed_count = data['analyzed_count']
                    
                    for fen, pos_dict in data['positions'].items():
                        pos_dict['fen'] = fen
                        self.positions[fen] = Position(**pos_dict)
            except:
                self.initialize_new_book()
        else:
            self.initialize_new_book()

    def initialize_new_book(self):
        """Initialize new book from starting position."""
        self.root_fen = self.engine.get_fen()
        self.positions[self.root_fen] = Position(fen=self.root_fen)

    def analyze(self):
        """Analyze one position in the book."""
        # Find path to leaf node
        path = []
        moves = []
        current_fen = self.root_fen
        path.append(current_fen)
        pos = self.positions[current_fen]
        
        while pos.best_child_fen:
            moves.append(pos.best_move)
            current_fen = pos.best_child_fen
            path.append(current_fen)
            pos = self.positions[current_fen]
        
        leaf = current_fen
        
        # Analyze leaf position
        analysis = self.engine.multipv(leaf, self.config['depth'])
        
        # Log and display analysis
        msg = f"\nAnalysis #{self.analyzed_count + 1}"
        msg += f"\ncp {pos.eval_cp or 0}"
        msg += f"\npv {' '.join(moves)}"
        print(msg)
        self.log.write(msg + "\n")
        
        for i, (move, eval_cp, mate_in) in enumerate(analysis, 1):
            if mate_in is None:
                msg = f"alt{i} {move} cp {eval_cp}"
            else:
                msg = f"alt{i} {move} mate {mate_in}"
            print(msg)
            self.log.write(msg + "\n")
        self.log.flush()
        
        # Update position with best move
        if analysis:
            best_move, best_cp, best_mate = analysis[0]
            pos.best_move = best_move
            
            if best_mate is None:
                pos.eval_cp = best_cp
                pos.mate_in = None
            else:
                pos.eval_cp = None
                pos.mate_in = best_mate
            
            # Add children positions
            for move, eval_cp, mate_in in analysis:
                child_fen = self.engine.get_fen(f"position fen {leaf} moves {move}")
                pos.children_fens.append(child_fen)
                pos.moves_to_children.append(move)
                
                if child_fen not in self.positions:
                    # Calculate child evaluation from parent's perspective
                    if mate_in is None:
                        child_cp = -eval_cp
                        child_mate = None
                    else:
                        child_cp = None
                        if mate_in > 0:
                            child_mate = -mate_in + 1
                        else:
                            child_mate = -mate_in
                    
                    self.positions[child_fen] = Position(
                        fen=child_fen,
                        eval_cp=child_cp,
                        mate_in=child_mate
                    )
            
            # Update evaluations along the path
            for fen in reversed(path):
                if self.positions[fen].children_fens:
                    self.minimax(fen)

    def minimax(self, fen: str):
        """Update position evaluation using minimax."""
        pos = self.positions[fen]
        
        if not pos.children_fens:
            return
        
        best_index = 0
        best_child_fen = pos.children_fens[0]
        best_child = self.positions[best_child_fen]
        
        # Find best child (from opponent's perspective, so worst for us)
        for i, child_fen in enumerate(pos.children_fens[1:], 1):
            child = self.positions[child_fen]
            ce, be = child.eval_cp, best_child.eval_cp
            cm, bm = child.mate_in, best_child.mate_in
            
            # Determine if child is better (for opponent)
            better = (
                (cm is not None and bm is not None and ((cm * bm > 0 and cm > bm) or (cm < 0))) or
                (cm is not None and bm is None and cm < 0) or
                (cm is None and bm is not None and bm > 0) or
                (cm is None and bm is None and ce and be and ce < be)
            )
            
            if better:
                best_child = child
                best_child_fen = child_fen
                best_index = i
        
        pos.best_move = pos.moves_to_children[best_index]
        pos.best_child_fen = best_child_fen
        
        # Update parent evaluation (negate child's)
        if best_child.mate_in is None:
            if best_child.eval_cp is not None:
                if best_child.eval_cp > 0:
                    pos.eval_cp = -best_child.eval_cp + 1
                elif best_child.eval_cp < 0:
                    pos.eval_cp = -best_child.eval_cp - 1
                else:
                    pos.eval_cp = 0
            pos.mate_in = None
        else:
            if best_child.mate_in > 0:
                pos.mate_in = -best_child.mate_in
            else:
                pos.mate_in = -best_child.mate_in + 1
            pos.eval_cp = None

    def export(self):
        """Export book to JSON and EPD formats."""
        # Export JSON
        with open(self.json_path, 'w') as f:
            positions_dict = {
                fen: {k: v for k, v in asdict(pos).items() if k != 'fen'}
                for fen, pos in self.positions.items()
            }
            json.dump({
                'root_fen': self.root_fen,
                'analyzed_count': self.analyzed_count,
                'positions': positions_dict
            }, f, indent=2)
        
        # Export EPD
        with open(self.epd_path, 'w') as f:
            for fen, pos in self.positions.items():
                # Extract board position (remove move counters)
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
                self.analyze()
                self.analyzed_count += 1
                
                # Export periodically
                if self.analyzed_count % 10 == 0:
                    self.export()
                
        except KeyboardInterrupt:
            pass
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