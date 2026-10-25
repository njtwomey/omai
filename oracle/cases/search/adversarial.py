"""Game-tree search against aima-python (a typed port in mdml_oracle.aima_games: minimax_decision,
alpha_beta_search, alpha_beta_cutoff_search) on tic-tac-toe positions and small connect-four boards, comparing the
chosen move, the root value and the number of nodes visited (aima-python's move order, first best move kept)."""

import numpy as np
from mdml_oracle.aima_games import (
    ConnectFour,
    Counter,
    TicTacToe,
    alpha_beta_cutoff_search,
    minimax_decision,
    play,
)
from mdml_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"aima-python": "port of games.py (2024)"}

TIC_TAC_TOE: list[list[tuple[int, int]]] = [
    [(2, 2)],
    [(1, 1), (2, 2)],
    [(1, 1), (1, 2), (2, 2)],
    [(2, 2), (1, 1), (3, 3), (1, 3)],
    [(1, 2), (2, 2), (3, 1)],
    [(1, 1), (2, 2), (3, 3), (1, 3), (3, 1)],
]

CONNECT: list[tuple[int, int, int, list[tuple[int, int]], int]] = [
    (4, 4, 3, [], 3),
    (4, 4, 3, [(2, 1)], 4),
    (5, 4, 4, [(3, 1), (3, 2)], 3),
    (4, 4, 3, [(1, 1), (2, 1), (1, 2)], 5),
]


def moves_json(moves: list[tuple[int, int]]) -> list[list[int]]:
    """Moves as lists."""
    return [[x, y] for x, y in moves]


def cases() -> list[Case]:
    out: list[Case] = []
    ttt = TicTacToe()
    for moves in TIC_TAC_TOE:
        s = play(ttt, moves)
        name = "-".join(f"{x}{y}" for x, y in moves)
        c = Counter()
        action, value = minimax_decision(s, ttt, c)
        out.append(
            case(
                f"tic-tac-toe/{name}/minimax",
                "minimax",
                [],
                np.array(0.0),
                game="tic-tac-toe",
                h=3,
                v=3,
                k=3,
                moves=moves_json(moves),
                action=list(action),
                value=value,
                nodes=c.nodes,
            )
        )
        c = Counter()
        best, value = alpha_beta_cutoff_search(s, ttt, c)
        out.append(
            case(
                f"tic-tac-toe/{name}/alphaBeta",
                "alphaBeta",
                [],
                np.array(0.0),
                game="tic-tac-toe",
                h=3,
                v=3,
                k=3,
                moves=moves_json(moves),
                action=list(best) if best is not None else None,
                value=value,
                nodes=c.nodes,
            )
        )
    c = Counter()
    best, value = alpha_beta_cutoff_search(ttt.initial, ttt, c)
    out.append(
        case(
            "tic-tac-toe/empty/alphaBeta",
            "alphaBeta",
            [],
            np.array(0.0),
            game="tic-tac-toe",
            h=3,
            v=3,
            k=3,
            moves=[],
            action=list(best) if best is not None else None,
            value=value,
            nodes=c.nodes,
        )
    )
    for h, v, k, moves, d in CONNECT:
        game = ConnectFour(h, v, k)
        s = play(game, moves)
        c = Counter()
        best, value = alpha_beta_cutoff_search(s, game, c, d=d)
        name = "-".join(f"{x}{y}" for x, y in moves) or "empty"
        out.append(
            case(
                f"connect-four/{h}x{v}k{k}/{name}/cutoff{d}",
                "alphaBetaCutoff",
                [],
                np.array(0.0),
                game="connect-four",
                h=h,
                v=v,
                k=k,
                d=d,
                moves=moves_json(moves),
                action=list(best) if best is not None else None,
                value=value,
                nodes=c.nodes,
            )
        )
    return out
