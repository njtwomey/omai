"""Two-player games by exhaustive search in plain Python (OpenSpiel is not installed, so the references are exact
computations from the rules): tic-tac-toe minimax values, misere Nim outcomes, Kuhn poker's Nash value and Leduc
poker's expected payoffs under uniform random play for every deal."""

from functools import cache
from itertools import permutations

import numpy as np
from mdml_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"numpy": np.__version__, "method": "exhaustive game-tree search"}

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


@cache
def ttt_value(board: tuple[int, ...], mover: int) -> int:
    """Negamax value of a tic-tac-toe position for the player to move (pieces 1 and 2)."""
    best = -2
    for i in range(9):
        if board[i] != 0:
            continue
        nxt = board[:i] + (mover,) + board[i + 1 :]
        if any(all(nxt[j] == mover for j in line) for line in LINES):
            return 1
        v = 0 if all(c != 0 for c in nxt) else -ttt_value(nxt, 3 - mover)
        best = max(best, v)
    return best


@cache
def misere_wins(heaps: tuple[int, ...]) -> bool:
    """Whether the mover wins misere Nim (taking the last object loses), by search."""
    for i, h in enumerate(heaps):
        for take in range(1, h + 1):
            nxt = heaps[:i] + (h - take,) + heaps[i + 1 :]
            if sum(nxt) == 0:
                continue  # taking the last object loses
            if not misere_wins(nxt):
                return True
    return False


def kuhn_value(alpha: float) -> float:
    """Expected payoff of the first player when both play Kuhn's Nash strategies (first player's alpha family)."""

    def bet(p: float) -> tuple[float, float]:
        return (1 - p, p)

    def strategy(player: int, card: int, history: str) -> tuple[float, float]:
        if player == 0:
            return bet([alpha, 0.0, 3 * alpha][card] if history == "" else [0.0, alpha + 1 / 3, 1.0][card])
        return bet([1 / 3, 0.0, 1.0][card] if history == "p" else [0.0, 1 / 3, 1.0][card])

    def payoff(cards: tuple[int, int], history: str) -> float:
        if history == "pbp":
            return -1.0
        if history == "bp":
            return 1.0
        stake = 1.0 if history == "pp" else 2.0
        return stake if cards[0] > cards[1] else -stake

    def value(cards: tuple[int, int], history: str) -> float:
        if history in ("pp", "pbp", "pbb", "bp", "bb"):
            return payoff(cards, history)
        player = len(history) % 2
        probs = strategy(player, cards[player], history)
        return probs[0] * value(cards, history + "p") + probs[1] * value(cards, history + "b")

    deals = list(permutations(range(3), 2))
    return sum(value((a, b), "") for a, b in deals) / len(deals)


def leduc_uniform(cards: tuple[int, int, int]) -> float:
    """Expected payoff of the first player in Leduc poker when both choose uniformly among legal actions."""
    raise_size = (2, 4)

    def showdown(contrib: float) -> float:
        a, b, pub = (c // 2 for c in cards)
        if a == pub and b != pub:
            return contrib
        if b == pub and a != pub:
            return -contrib
        if a == b:
            return 0.0
        return contrib if a > b else -contrib

    def value(rnd: int, actions: tuple[int, ...], contrib: tuple[int, int], raises: int, mover: int) -> float:
        facing = contrib[mover] < contrib[1 - mover]
        legal = ([0] if facing else []) + [1] + ([2] if raises < 2 else [])
        total = 0.0
        for a in legal:
            c = list(contrib)
            if a == 0:
                total += -c[0] if mover == 0 else c[1]
                continue
            if a == 2:
                c[mover] = c[1 - mover] + raise_size[rnd]
                r = raises + 1
            else:
                c[mover] = c[1 - mover]
                r = raises
            hist = (*actions, a)
            if a == 1 and len(hist) >= 2:
                if rnd == 0:
                    total += value(1, (), (c[0], c[1]), 0, 0)
                else:
                    total += showdown(c[0])
            else:
                total += value(rnd, hist, (c[0], c[1]), r, 1 - mover)
        return total / len(legal)

    return value(0, (), (1, 1), 0, 0)


def cases() -> list[Case]:
    out: list[Case] = []
    empty: tuple[int, ...] = (0,) * 9
    out.append(case("tictactoe/empty", "ticTacToeValue", [], np.array([float(ttt_value(empty, 1))]), moves=[]))
    first = [float(-ttt_value(empty[:i] + (1,) + empty[i + 1 :], 2)) for i in range(9)]
    out.append(case("tictactoe/first-moves", "ticTacToeFirst", [], np.array(first)))
    replies: list[float] = []
    for i in range(9):
        for j in range(9):
            if i == j:
                continue
            b: list[int] = list(empty)
            b[i] = 1
            b[j] = 2
            replies.append(float(ttt_value(tuple(b), 1)))
    out.append(case("tictactoe/replies", "ticTacToeReplies", [], np.array(replies)))
    heaps = [(a, b, c) for a in range(4) for b in range(4) for c in range(5) if a + b + c > 0]
    wins = [1.0 if misere_wins(h) else -1.0 for h in heaps]
    out.append(case("nim/misere", "nimMisere", [], np.array(wins), heaps=[list(h) for h in heaps]))
    for alpha in (0.0, 1 / 6, 1 / 3):
        out.append(case(f"kuhn/nash/{alpha:.4f}", "kuhnNash", [], np.array([kuhn_value(alpha)]), alpha=alpha))
    deals = list(permutations(range(6), 3))
    values = [leduc_uniform((a, b, c)) for a, b, c in deals]
    out.append(case("leduc/uniform", "leducUniform", [], np.array(values), deals=[list(d) for d in deals]))
    return out
