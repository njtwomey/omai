"""A typed port of the state-space search of aima-python (search.py and utils.py, github.com/aimacode/aima-python),
the reference implementation of Russell and Norvig's AIMA, which is not packaged for installation.

The port keeps aima-python's behaviour line for line: `Node.__lt__` compares states, so its priority queue breaks f
ties by state order; breadth-first graph search tests for the goal when a node is generated; depth-first graph search
pushes every child and pops the last; best-first graph search replaces a frontier node when a child of its state has a
lower f; `InstrumentedProblem` counts successor calls (expansions), goal tests and result calls (generated states).
The problem instances are aima-python's: the Romania map with its locations, the eight puzzle and n-queens.
"""

import heapq
import math
from collections import deque
from collections.abc import Callable

type State = str | tuple[int, ...]
"""A state of the ported problems: a city name or a tuple of tiles or queen rows."""

type Action = str | int
"""An action of the ported problems: a city, a blank move or a queen row."""


class Problem:
    """aima-python's `Problem`: actions, result, goal test, path cost (default 1 per step) and heuristic h."""

    def __init__(self, initial: State, goal: State | None = None) -> None:
        self.initial = initial
        self.goal = goal

    def actions(self, state: State) -> list[Action]:
        """The actions available in a state."""
        raise NotImplementedError

    def result(self, state: State, action: Action) -> State:
        """The state an action leads to."""
        raise NotImplementedError

    def goal_test(self, state: State) -> bool:
        """Whether a state is the goal."""
        return state == self.goal

    def path_cost(self, c: float, state1: State, action: Action, state2: State) -> float:
        """The cost of a path that arrives at state2 from state1 by action, given cost c to state1."""
        return c + 1

    def h(self, node: "Node") -> float:
        """The heuristic estimate of a node's distance to the goal."""
        return 0


class Node:
    """aima-python's `Node`: ordered by state (`__lt__`), equal when the states are."""

    def __init__(
        self, state: State, parent: "Node | None" = None, action: Action | None = None, path_cost: float = 0
    ) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0 if parent is None else parent.depth + 1
        self.f: float = 0.0
        self.memo: float | None = None

    def __lt__(self, other: "Node") -> bool:
        return self.state < other.state  # pyright: ignore[reportOperatorIssue]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self) -> int:
        return hash(self.state)

    def expand(self, problem: Problem) -> list["Node"]:
        """The child nodes, one per action."""
        return [self.child_node(problem, action) for action in problem.actions(self.state)]

    def child_node(self, problem: Problem, action: Action) -> "Node":
        """The child reached by an action."""
        next_state = problem.result(self.state, action)
        return Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))

    def path(self) -> list["Node"]:
        """The nodes from the root to this node."""
        node: Node | None = self
        back: list[Node] = []
        while node is not None:
            back.append(node)
            node = node.parent
        return list(reversed(back))

    def solution(self) -> list[Action]:
        """The actions from the root to this node."""
        return [n.action for n in self.path()[1:] if n.action is not None]


class InstrumentedProblem(Problem):
    """aima-python's `InstrumentedProblem`: counts successor calls, goal tests and result calls."""

    def __init__(self, problem: Problem) -> None:
        super().__init__(problem.initial, problem.goal)
        self.problem = problem
        self.succs = 0
        self.goal_tests = 0
        self.states = 0

    def actions(self, state: State) -> list[Action]:
        """Count and delegate."""
        self.succs += 1
        return self.problem.actions(state)

    def result(self, state: State, action: Action) -> State:
        """Count and delegate."""
        self.states += 1
        return self.problem.result(state, action)

    def goal_test(self, state: State) -> bool:
        """Count and delegate."""
        self.goal_tests += 1
        return self.problem.goal_test(state)

    def path_cost(self, c: float, state1: State, action: Action, state2: State) -> float:
        """Delegate."""
        return self.problem.path_cost(c, state1, action, state2)

    def h(self, node: Node) -> float:
        """Delegate."""
        return self.problem.h(node)


class PriorityQueue:
    """aima-python's min priority queue of (f(item), item) on heapq, with membership by state equality."""

    def __init__(self, f: Callable[[Node], float]) -> None:
        self.heap: list[tuple[float, Node]] = []
        self.f = f

    def append(self, item: Node) -> None:
        """Push an item."""
        heapq.heappush(self.heap, (self.f(item), item))

    def pop(self) -> Node:
        """Pop the item of lowest f (ties: lowest state)."""
        return heapq.heappop(self.heap)[1]

    def __len__(self) -> int:
        return len(self.heap)

    def __contains__(self, key: Node) -> bool:
        return any(item == key for _, item in self.heap)

    def __getitem__(self, key: Node) -> float:
        for value, item in self.heap:
            if item == key:
                return value
        raise KeyError(key)

    def __delitem__(self, key: Node) -> None:
        del self.heap[[item == key for _, item in self.heap].index(True)]
        heapq.heapify(self.heap)


def memoize_f(f: Callable[[Node], float]) -> Callable[[Node], float]:
    """aima-python's `memoize(f, 'f')`: cache f on the node (in a slot of its own, set on first evaluation)."""

    def g(node: Node) -> float:
        if node.memo is None:
            node.memo = f(node)
            node.f = node.memo
        return node.memo

    return g


def breadth_first_graph_search(problem: Problem) -> Node | None:
    """aima-python's `breadth_first_graph_search`."""
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier: deque[Node] = deque([node])
    explored: set[State] = set()
    while frontier:
        node = frontier.popleft()
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return child
                frontier.append(child)
    return None


def depth_first_graph_search(problem: Problem) -> Node | None:
    """aima-python's `depth_first_graph_search`."""
    frontier = [Node(problem.initial)]
    explored: set[State] = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        frontier.extend(
            child for child in node.expand(problem) if child.state not in explored and child not in frontier
        )
    return None


def depth_limited_search(problem: Problem, limit: int = 50) -> Node | str | None:
    """aima-python's `depth_limited_search` (recursive, returns 'cutoff' when the limit pruned a branch)."""

    def recursive_dls(node: Node, limit: int) -> Node | str | None:
        if problem.goal_test(node.state):
            return node
        if limit == 0:
            return "cutoff"
        cutoff_occurred = False
        for child in node.expand(problem):
            result = recursive_dls(child, limit - 1)
            if result == "cutoff":
                cutoff_occurred = True
            elif result is not None:
                return result
        return "cutoff" if cutoff_occurred else None

    return recursive_dls(Node(problem.initial), limit)


def iterative_deepening_search(problem: Problem) -> Node | str | None:
    """aima-python's `iterative_deepening_search`."""
    for depth in range(10_000):
        result = depth_limited_search(problem, depth)
        if result != "cutoff":
            return result
    return None


def best_first_graph_search(problem: Problem, f: Callable[[Node], float]) -> Node | None:
    """aima-python's `best_first_graph_search`."""
    f = memoize_f(f)
    node = Node(problem.initial)
    frontier = PriorityQueue(f)
    frontier.append(node)
    explored: set[State] = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier and f(child) < frontier[child]:
                del frontier[child]
                frontier.append(child)
    return None


def uniform_cost_search(problem: Problem) -> Node | None:
    """aima-python's `uniform_cost_search`."""
    return best_first_graph_search(problem, lambda node: node.path_cost)


def astar_search(problem: Problem) -> Node | None:
    """aima-python's `astar_search` with the problem's h."""
    return best_first_graph_search(problem, lambda n: n.path_cost + problem.h(n))


def greedy_best_first_graph_search(problem: Problem) -> Node | None:
    """aima-python's greedy best-first search (best-first on h)."""
    return best_first_graph_search(problem, problem.h)


def recursive_best_first_search(problem: Problem) -> Node | None:
    """aima-python's `recursive_best_first_search`."""

    def rbfs(node: Node, flimit: float) -> tuple[Node | None, float]:
        if problem.goal_test(node.state):
            return node, 0
        successors = node.expand(problem)
        if len(successors) == 0:
            return None, math.inf
        for s in successors:
            s.f = max(s.path_cost + problem.h(s), node.f)
        while True:
            successors.sort(key=lambda x: x.f)
            best = successors[0]
            if best.f > flimit:
                return None, best.f
            alternative = successors[1].f if len(successors) > 1 else math.inf
            result, best.f = rbfs(best, min(flimit, alternative))
            if result is not None:
                return result, best.f

    node = Node(problem.initial)
    node.f = problem.h(node)
    return rbfs(node, math.inf)[0]


ROMANIA: dict[str, dict[str, int]] = {
    "Arad": {"Zerind": 75, "Sibiu": 140, "Timisoara": 118},
    "Bucharest": {"Urziceni": 85, "Pitesti": 101, "Giurgiu": 90, "Fagaras": 211},
    "Craiova": {"Drobeta": 120, "Rimnicu": 146, "Pitesti": 138},
    "Drobeta": {"Mehadia": 75},
    "Eforie": {"Hirsova": 86},
    "Fagaras": {"Sibiu": 99},
    "Hirsova": {"Urziceni": 98},
    "Iasi": {"Vaslui": 92, "Neamt": 87},
    "Lugoj": {"Timisoara": 111, "Mehadia": 70},
    "Oradea": {"Zerind": 71, "Sibiu": 151},
    "Pitesti": {"Rimnicu": 97},
    "Rimnicu": {"Sibiu": 80},
    "Urziceni": {"Vaslui": 142},
}
"""aima-python's `romania_map` before `make_undirected`."""

LOCATIONS: dict[str, tuple[int, int]] = {
    "Arad": (91, 492),
    "Bucharest": (400, 327),
    "Craiova": (253, 288),
    "Drobeta": (165, 299),
    "Eforie": (562, 293),
    "Fagaras": (305, 449),
    "Giurgiu": (375, 270),
    "Hirsova": (534, 350),
    "Iasi": (473, 506),
    "Lugoj": (165, 379),
    "Mehadia": (168, 339),
    "Neamt": (406, 537),
    "Oradea": (131, 571),
    "Pitesti": (320, 368),
    "Rimnicu": (233, 410),
    "Sibiu": (207, 457),
    "Timisoara": (94, 410),
    "Urziceni": (456, 350),
    "Vaslui": (509, 444),
    "Zerind": (108, 531),
}
"""aima-python's `romania_map.locations`."""


def undirected(graph: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    """aima-python's `UndirectedGraph`: the dict with every link added in reverse (neighbour order as it builds)."""
    g = {a: dict(b) for a, b in graph.items()}
    for a in list(g.keys()):
        for b, dist in list(g[a].items()):
            g.setdefault(b, {})[a] = dist
    return g


class GraphProblem(Problem):
    """aima-python's `GraphProblem`: actions are neighbours in dict order, h the truncated straight-line distance."""

    def __init__(self, initial: str, goal: str, graph: dict[str, dict[str, int]]) -> None:
        super().__init__(initial, goal)
        self.graph = graph

    def actions(self, state: State) -> list[Action]:
        """The neighbouring cities."""
        return list(self.graph[str(state)].keys())

    def result(self, state: State, action: Action) -> State:
        """Go to the city."""
        return str(action)

    def path_cost(self, c: float, state1: State, action: Action, state2: State) -> float:
        """Add the road length."""
        return c + self.graph[str(state1)][str(state2)]

    def h(self, node: Node) -> float:
        """aima-python's `int(distance(locs[node.state], locs[self.goal]))`."""
        a = LOCATIONS[str(node.state)]
        b = LOCATIONS[str(self.goal)]
        return int(math.hypot(a[0] - b[0], a[1] - b[1]))


class EightPuzzle(Problem):
    """aima-python's `EightPuzzle` (blank is 0; actions UP, DOWN, LEFT, RIGHT move the blank; h counts misplaced
    squares including the blank)."""

    def __init__(self, initial: tuple[int, ...], goal: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 0)) -> None:
        super().__init__(initial, goal)

    def actions(self, state: State) -> list[Action]:
        """The blank's legal moves."""
        moves: list[Action] = ["UP", "DOWN", "LEFT", "RIGHT"]
        blank = tuple(state).index(0)
        if blank % 3 == 0:
            moves.remove("LEFT")
        if blank < 3:
            moves.remove("UP")
        if blank % 3 == 2:
            moves.remove("RIGHT")
        if blank > 5:
            moves.remove("DOWN")
        return moves

    def result(self, state: State, action: Action) -> State:
        """Swap the blank with its neighbour."""
        tiles = [int(x) for x in state]
        blank = tiles.index(0)
        neighbour = blank + {"UP": -3, "DOWN": 3, "LEFT": -1, "RIGHT": 1}[str(action)]
        tiles[blank], tiles[neighbour] = tiles[neighbour], tiles[blank]
        return tuple(tiles)

    def h(self, node: Node) -> float:
        """Misplaced squares (the blank included, as in aima-python)."""
        goal = tuple(self.goal or ())
        return sum(s != g for s, g in zip(tuple(node.state), goal, strict=True))


class ManhattanEightPuzzle(EightPuzzle):
    """The eight puzzle with the Manhattan distance of the tiles (blank excluded), an admissible heuristic."""

    def h(self, node: Node) -> float:
        """Sum of tile distances to their goal squares."""
        goal = tuple(self.goal or ())
        total = 0
        for i, t in enumerate(tuple(node.state)):
            if t == 0:
                continue
            j = goal.index(t)
            total += abs(i // 3 - j // 3) + abs(i % 3 - j % 3)
        return total


class NQueensProblem(Problem):
    """aima-python's incremental `NQueensProblem`: place queens column by column on non-attacked rows."""

    def __init__(self, n: int) -> None:
        super().__init__(tuple([-1] * n))
        self.n = n

    def actions(self, state: State) -> list[Action]:
        """Rows of the next column not attacked by the queens placed."""
        s = tuple(int(x) for x in state)
        if s[-1] != -1:
            return []
        col = s.index(-1)
        return [row for row in range(self.n) if not self.conflicted(s, row, col)]

    def result(self, state: State, action: Action) -> State:
        """Place a queen in the next column."""
        s = [int(x) for x in state]
        s[s.index(-1)] = int(action)
        return tuple(s)

    def conflicted(self, state: tuple[int, ...], row: int, col: int) -> bool:
        """Whether a queen at (row, col) is attacked by the queens of earlier columns."""
        return any(self.conflict(row, col, state[c], c) for c in range(col))

    @staticmethod
    def conflict(row1: int, col1: int, row2: int, col2: int) -> bool:
        """Whether two queens attack each other."""
        return row1 == row2 or col1 == col2 or row1 - col1 == row2 - col2 or row1 + col1 == row2 + col2

    def goal_test(self, state: State) -> bool:
        """All queens placed and none attacked."""
        s = tuple(int(x) for x in state)
        if s[-1] == -1:
            return False
        return not any(self.conflicted(s, s[col], col) for col in range(len(s)))


def run(search: Callable[[Problem], Node | str | None], problem: Problem) -> dict[str, object]:
    """Run a search on an instrumented problem: the solution and aima-python's counts."""
    p = InstrumentedProblem(problem)
    node = search(p)
    solved = isinstance(node, Node)
    return {
        "status": "solved" if solved else ("cutoff" if node == "cutoff" else "failure"),
        "actions": node.solution() if isinstance(node, Node) else [],
        "cost": node.path_cost if isinstance(node, Node) else None,
        "depth": node.depth if isinstance(node, Node) else -1,
        "expanded": p.succs,
        "goalTests": p.goal_tests,
        "generated": p.states,
    }
