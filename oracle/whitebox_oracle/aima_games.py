"""A typed port of the game-tree search of aima-python (games.py, github.com/aimacode/aima-python): `minimax_decision`,
`alpha_beta_search` and `alpha_beta_cutoff_search`, with counters of the nodes each visits (calls of max_value and
min_value, plus the root), and the games `TicTacToe`, `ConnectFour` and `Fig52Game` exactly as aima-python defines
them (move order, utilities, terminal tests)."""

import math
from collections.abc import Callable
from dataclasses import dataclass, field

type Move = tuple[int, int] | str
"""A move: a board square (x, y) counted from 1, or a named edge of figure 5.2."""


@dataclass(frozen=True)
class GameState:
    """aima-python's `GameState` namedtuple: the player to move, the utility, the board and the remaining moves."""

    to_move: str
    utility: int
    board: dict[tuple[int, int], str] = field(default_factory=dict[tuple[int, int], str])
    moves: list[tuple[int, int]] = field(default_factory=list[tuple[int, int]])


class TicTacToe:
    """aima-python's `TicTacToe(h, v, k)`: X moves first; utility +1 if X has k in a row, -1 if O does."""

    def __init__(self, h: int = 3, v: int = 3, k: int = 3) -> None:
        self.h = h
        self.v = v
        self.k = k
        moves = [(x, y) for x in range(1, h + 1) for y in range(1, v + 1)]
        self.initial = GameState(to_move="X", utility=0, board={}, moves=moves)

    def actions(self, state: GameState) -> list[tuple[int, int]]:
        """The empty squares in order."""
        return state.moves

    def result(self, state: GameState, move: tuple[int, int]) -> GameState:
        """Mark a square."""
        if move not in state.moves:
            return state
        board = dict(state.board)
        board[move] = state.to_move
        moves = list(state.moves)
        moves.remove(move)
        return GameState(
            to_move="O" if state.to_move == "X" else "X",
            utility=self.compute_utility(board, move, state.to_move),
            board=board,
            moves=moves,
        )

    def utility(self, state: GameState, player: str) -> int:
        """The utility for a player."""
        return state.utility if player == "X" else -state.utility

    def terminal_test(self, state: GameState) -> bool:
        """A win or a full board."""
        return state.utility != 0 or len(state.moves) == 0

    def to_move(self, state: GameState) -> str:
        """The player to move."""
        return state.to_move

    def compute_utility(self, board: dict[tuple[int, int], str], move: tuple[int, int], player: str) -> int:
        """+1 or -1 if the move completes k in a row for X or O, else 0."""
        if any(self.k_in_row(board, move, player, d) for d in ((0, 1), (1, 0), (1, -1), (1, 1))):
            return 1 if player == "X" else -1
        return 0

    def k_in_row(
        self, board: dict[tuple[int, int], str], move: tuple[int, int], player: str, d: tuple[int, int]
    ) -> bool:
        """Whether the move is part of k in a row along direction d."""
        dx, dy = d
        x, y = move
        n = 0
        while board.get((x, y)) == player:
            n += 1
            x, y = x + dx, y + dy
        x, y = move
        while board.get((x, y)) == player:
            n += 1
            x, y = x - dx, y - dy
        n -= 1
        return n >= self.k


class ConnectFour(TicTacToe):
    """aima-python's `ConnectFour(h, v, k)`: a square is playable when it is on the bottom row or above a filled one."""

    def actions(self, state: GameState) -> list[tuple[int, int]]:
        """The playable squares in order."""
        return [(x, y) for (x, y) in state.moves if y == 1 or (x, y - 1) in state.board]


class Counter:
    """Counts nodes visited."""

    def __init__(self) -> None:
        self.nodes = 1


def minimax_decision(state: GameState, game: TicTacToe, counter: Counter) -> tuple[tuple[int, int], float]:
    """aima-python's `minimax_decision`, also returning the root value."""
    player = game.to_move(state)

    def max_value(s: GameState) -> float:
        counter.nodes += 1
        if game.terminal_test(s):
            return game.utility(s, player)
        v = -math.inf
        for a in game.actions(s):
            v = max(v, min_value(game.result(s, a)))
        return v

    def min_value(s: GameState) -> float:
        counter.nodes += 1
        if game.terminal_test(s):
            return game.utility(s, player)
        v = math.inf
        for a in game.actions(s):
            v = min(v, max_value(game.result(s, a)))
        return v

    values = [(a, min_value(game.result(state, a))) for a in game.actions(state)]
    best = max(values, key=lambda av: av[1])
    return best


def alpha_beta_cutoff_search(
    state: GameState,
    game: TicTacToe,
    counter: Counter,
    d: float = math.inf,
    eval_fn: Callable[[GameState], float] | None = None,
) -> tuple[tuple[int, int] | None, float]:
    """aima-python's `alpha_beta_cutoff_search` (with d infinite it is `alpha_beta_search`), also returning the root
    value."""
    player = game.to_move(state)

    def utility(s: GameState) -> float:
        return game.utility(s, player)

    evaluate = eval_fn or utility

    def cutoff(s: GameState, depth: int) -> bool:
        return depth > d or game.terminal_test(s)

    def max_value(s: GameState, alpha: float, beta: float, depth: int) -> float:
        counter.nodes += 1
        if cutoff(s, depth):
            return evaluate(s)
        v = -math.inf
        for a in game.actions(s):
            v = max(v, min_value(game.result(s, a), alpha, beta, depth + 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(s: GameState, alpha: float, beta: float, depth: int) -> float:
        counter.nodes += 1
        if cutoff(s, depth):
            return evaluate(s)
        v = math.inf
        for a in game.actions(s):
            v = min(v, max_value(game.result(s, a), alpha, beta, depth + 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    best_score = -math.inf
    beta = math.inf
    best_action: tuple[int, int] | None = None
    for a in game.actions(state):
        v = min_value(game.result(state, a), best_score, beta, 1)
        if v > best_score:
            best_score = v
            best_action = a
    return best_action, best_score


FIG52_SUCCS: dict[str, dict[str, str]] = {
    "A": {"a1": "B", "a2": "C", "a3": "D"},
    "B": {"b1": "B1", "b2": "B2", "b3": "B3"},
    "C": {"c1": "C1", "c2": "C2", "c3": "C3"},
    "D": {"d1": "D1", "d2": "D2", "d3": "D3"},
}
"""aima-python's `Fig52Game.succs`."""

FIG52_UTILS: dict[str, int] = {"B1": 3, "B2": 12, "B3": 8, "C1": 2, "C2": 4, "C3": 6, "D1": 14, "D2": 5, "D3": 2}
"""aima-python's `Fig52Game.utils` (utilities for MAX)."""


def play(game: TicTacToe, moves: list[tuple[int, int]]) -> GameState:
    """The state after a sequence of moves from the initial state."""
    s = game.initial
    for m in moves:
        s = game.result(s, m)
    return s
