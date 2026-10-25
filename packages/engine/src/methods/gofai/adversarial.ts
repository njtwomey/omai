/**
 * Adversarial search played node by node over a whole game tree, from the core's traced game searches
 * (`mdml-engine/search`): minimax and alpha-beta, showing the node visited, the values backed up and the branches
 * alpha-beta prunes. Games include textbook trees (AIMA figure 5.2), random trees and tic-tac-toe.
 */

import { MdmlError } from '../../lib/base/index.ts'
import { type BoardSteps, boardSteps, type Explained } from '../../lib/exposition/index.ts'
import { type Key, randint } from '../../lib/random/index.ts'
import { alphaBetaAlgorithm, type Game, type GameSearchState, minimaxAlgorithm } from '../../lib/search/index.ts'
import { iterate } from '../../lib/trace/index.ts'

/** A position of a game whose states remember how they were reached, so every node of the game tree is distinct. */
export interface TreeState {
  /** The moves from the root, joined (`''` at the root). */
  readonly path: string
  readonly depth: number
}

/** A game tree given by its leaf values: `branching` moves per node, `depth` plies, MAX to move at the root. */
export interface ExplicitTree extends Game<TreeState, string> {
  readonly leaves: readonly number[]
  readonly branching: number
  readonly depth: number
}

const MOVES = 'abcdefghij'

/**
 * A uniform game tree with given leaf utilities (for MAX), as in textbook figures of minimax and alpha-beta. Moves are
 * named `a`, `b`, `c`, ...; MAX moves at even depths.
 *
 * @param leaves - The utilities of the leaves, left to right (`branching ^ depth` of them).
 * @param branching - Moves per node (2 to 10).
 * @returns The game.
 * @throws MdmlError `value` when the leaf count is not a power of the branching factor.
 */
export function explicitTree(leaves: readonly number[], branching: number): ExplicitTree {
  if (!(Number.isInteger(branching) && branching >= 2 && branching <= MOVES.length))
    throw new MdmlError('value', `explicitTree: the branching ${branching} must be an integer from 2 to 10`)
  const depth = Math.round(Math.log(leaves.length) / Math.log(branching))
  if (branching ** depth !== leaves.length)
    throw new MdmlError('value', `explicitTree: ${leaves.length} leaves is not a power of ${branching}`)
  const index = (path: string): number => [...path].reduce((i, m) => i * branching + MOVES.indexOf(m), 0)
  return {
    leaves,
    branching,
    depth,
    initial: { path: '', depth: 0 },
    player: (s) => (s.depth % 2 === 0 ? 'MAX' : 'MIN'),
    actions: (s) => (s.depth >= depth ? [] : [...MOVES.slice(0, branching)]),
    result: (s, a) => ({ path: s.path + a, depth: s.depth + 1 }),
    isTerminal: (s) => s.depth >= depth,
    utility: (s, p) => {
      const v = leaves[index(s.path)] ?? 0
      return p === 'MAX' ? v : -v
    },
    key: (s) => s.path,
    label: (s) => s.path,
  }
}

/**
 * The two-ply game of AIMA figure 5.2: MAX chooses among three moves, MIN among three replies.
 *
 * @returns The game (minimax value 3).
 */
export function textbookTree(): ExplicitTree {
  return explicitTree([3, 12, 8, 2, 4, 6, 14, 5, 2], 3)
}

/**
 * A uniform game tree with random integer leaf utilities from -9 to 9.
 *
 * @param key - The key.
 * @param branching - Moves per node.
 * @param depth - Plies.
 * @returns The game.
 */
export function randomTree(key: Key, branching = 3, depth = 3): ExplicitTree {
  const n = branching ** depth
  return explicitTree(Array.from(randint(key, [n], -9, 10).data), branching)
}

/** A tic-tac-toe position: the board as nine characters (`X`, `O` or `.`, row by row) and how it was reached. */
export interface TicTacToeState extends TreeState {
  readonly board: string
  readonly toMove: 'X' | 'O'
}

const LINES = [
  [0, 1, 2],
  [3, 4, 5],
  [6, 7, 8],
  [0, 3, 6],
  [1, 4, 7],
  [2, 5, 8],
  [0, 4, 8],
  [2, 4, 6],
] as const

function winner(board: string): 'X' | 'O' | null {
  for (const [a, b, c] of LINES) {
    const p = board[a]
    if ((p === 'X' || p === 'O') && p === board[b] && p === board[c]) return p
  }
  return null
}

/**
 * Tic-tac-toe from a position: moves are the free squares, numbered 1 to 9 row by row and named by mover and square
 * (`X5`); a win is worth 1 to the winner and -1 to the loser.
 *
 * @param board - The position, nine characters (default the empty board); the player with fewer marks moves (X on
 *   ties).
 * @returns The game.
 * @throws MdmlError `value` for a malformed board.
 */
export function ticTacToe(board = '.........'): Game<TicTacToeState, string> {
  if (!/^[XO.]{9}$/.test(board)) throw new MdmlError('value', `ticTacToe: '${board}' is not a board`)
  const xs = [...board].filter((c) => c === 'X').length
  const os = [...board].filter((c) => c === 'O').length
  return {
    initial: { board, toMove: xs <= os ? 'X' : 'O', path: '', depth: 0 },
    player: (s) => s.toMove,
    actions: (s) =>
      winner(s.board) !== null ? [] : [...s.board].flatMap((c, i) => (c === '.' ? [`${s.toMove}${i + 1}`] : [])),
    result: (s, a) => {
      const i = Number(a.slice(1)) - 1
      return {
        board: s.board.slice(0, i) + s.toMove + s.board.slice(i + 1),
        toMove: s.toMove === 'X' ? 'O' : 'X',
        path: s.path === '' ? a : `${s.path} ${a}`,
        depth: s.depth + 1,
      }
    },
    isTerminal: (s) => winner(s.board) !== null || !s.board.includes('.'),
    utility: (s, p) => {
      const w = winner(s.board)
      return w === null ? 0 : w === p ? 1 : -1
    },
    key: (s) => s.path,
    label: (s) => s.path,
  }
}

/** Options of {@link playGameSearch}. */
export interface GameSearchPlayOptions {
  /** Minimax or alpha-beta (default `alpha-beta`). */
  readonly algorithm?: 'minimax' | 'alpha-beta'
  /** Try the moves that look best first (by the utility of terminal children), so alpha-beta prunes more. */
  readonly ordered?: boolean
  /** The largest tree drawn (default 600 nodes). */
  readonly maxNodes?: number
}

/** What a played game search found. */
export interface GameSearchPlay {
  readonly algorithm: 'minimax' | 'alpha-beta'
  /** The best move at the root and its value for the player to move there. */
  readonly action: string | undefined
  readonly value: number
  readonly principalVariation: readonly string[]
  /** Nodes visited and nodes in the whole tree. */
  readonly visited: number
  readonly treeSize: number
  readonly cutoffs: number
}

interface Laid {
  readonly labels: string[]
  readonly parent: number[]
  readonly children: number[][]
  readonly kinds: ('max' | 'min' | 'leaf')[]
  readonly moves: string[]
  readonly x: number[]
  readonly depth: number[]
}

// The whole tree, depth first, children in the game's move order, leaves spread evenly and parents over their children.
function layTree<S>(game: Game<S, string>, root: S, cap: number): Laid {
  const t: Laid = { labels: [], parent: [], children: [], kinds: [], moves: [], x: [], depth: [] }
  const rootPlayer = game.player(root)
  const label = (s: S): string => game.label?.(s) ?? String(game.key?.(s) ?? JSON.stringify(s))
  let leaf = 0
  const visit = (s: S, parent: number, move: string, depth: number): number => {
    if (t.labels.length >= cap)
      throw new MdmlError('value', `playGameSearch: the game tree has more than ${cap} nodes; search a smaller one`)
    const i = t.labels.length
    t.labels.push(label(s))
    t.parent.push(parent)
    t.children.push([])
    t.moves.push(move)
    t.depth.push(depth)
    t.x.push(0)
    const terminal = game.isTerminal(s)
    t.kinds.push(terminal ? 'leaf' : game.player(s) === rootPlayer ? 'max' : 'min')
    if (terminal) t.x[i] = leaf++
    else {
      for (const a of game.actions(s)) (t.children[i] as number[]).push(visit(game.result(s, a), i, a, depth + 1))
      const xs = (t.children[i] as number[]).map((c) => t.x[c] ?? 0)
      t.x[i] = xs.length === 0 ? leaf++ : (Math.min(...xs) + Math.max(...xs)) / 2
    }
    return i
  }
  visit(root, -1, '', 0)
  return t
}

const CATEGORIES = ['unvisited', 'MAX', 'MIN', 'visited', 'current', 'pruned', 'path'] as const
const C = Object.fromEntries(CATEGORIES.map((c, i) => [c, i])) as Record<(typeof CATEGORIES)[number], number>

/**
 * Minimax or alpha-beta search over a whole (small) game tree, recording every node visit for playback: the node
 * visited, the values backed up so far (for the player to move at the root), the subtrees alpha-beta prunes, and at the
 * end the principal variation. Unvisited leaves show their utilities and inner nodes whether MAX (▲) or MIN (▼) moves.
 *
 * @param game - The game (states with distinct labels per tree node, as the games of this module).
 * @param options - The search, move ordering and the size limit.
 * @returns The best move, the value and the effort, with a `boardSteps` record of the search over the tree.
 */
export function playGameSearch<S>(
  game: Game<S, string>,
  options: GameSearchPlayOptions = {},
): Explained<GameSearchPlay> {
  const algorithm = options.algorithm ?? 'alpha-beta'
  const root = game.initial
  const tree = layTree(game, root, options.maxNodes ?? 600)
  const n = tree.labels.length
  const id = new Map(tree.labels.map((l, i) => [l, i]))
  const rootPlayer = game.player(root)
  const order = options.ordered === true ? ('evaluate' as const) : ('given' as const)
  const search =
    algorithm === 'minimax' ? minimaxAlgorithm(game, root, { order }) : alphaBetaAlgorithm(game, root, { order })
  // The static text: leaf utilities for the root player, ▲ and ▼ for the movers.
  const leafValue = (i: number): number => {
    let s = root
    const path: string[] = []
    for (let j = i; j > 0; j = tree.parent[j] ?? 0) path.unshift(tree.moves[j] ?? '')
    for (const m of path) s = game.result(s, m)
    return game.utility(s, rootPlayer)
  }
  const fixed = tree.kinds.map((k, i) => (k === 'leaf' ? String(leafValue(i)) : k === 'max' ? '▲' : '▼'))
  const status = new Array<number>(n).fill(C.unvisited)
  const values = new Array<string>(n).fill('')
  const pruneBelow = (i: number): void => {
    status[i] = C.pruned
    for (const c of tree.children[i] ?? []) pruneBelow(c)
  }
  const steps: { cells: number[]; labels: string[]; caption: string }[] = []
  const visitedSeries: number[] = []
  const cutoffSeries: number[] = []
  let last: GameSearchState<string> | undefined
  for (const { state: s } of iterate(search, { maxSteps: 4 * n + 8 })) {
    const notes: string[] = []
    for (const b of s.backedUp) {
      const i = id.get(b.node)
      if (i === undefined) continue
      values[i] = String(b.value)
    }
    for (const p of s.pruned) {
      const i = id.get(p.node)
      if (i === undefined) continue
      for (const c of tree.children[i] ?? []) if (status[c] === C.unvisited) pruneBelow(c)
      notes.push(`prune ${p.skipped} below ${p.node === '' ? 'the root' : p.node}`)
    }
    const v = s.visited === null ? undefined : id.get(s.visited.node)
    if (v !== undefined) status[v] = tree.kinds[v] === 'leaf' ? C.visited : tree.kinds[v] === 'max' ? C.MAX : C.MIN
    const cells = [...status]
    if (v !== undefined) cells[v] = C.current
    if (s.status === 'solved') {
      let at = 0
      cells[0] = C.path
      for (const m of s.principalVariation) {
        const next = (tree.children[at] ?? []).find((c) => tree.moves[c] === m)
        if (next === undefined) break
        cells[next] = C.path
        at = next
      }
    }
    const where = v === undefined ? '' : `visit ${tree.labels[v] === '' ? 'the root' : tree.moves[v]}`
    const caption =
      s.status === 'solved'
        ? `value ${s.value}: play ${s.principalVariation.join(' ') || '(none)'}`
        : [where, ...notes].filter((x) => x !== '').join('; ')
    steps.push({ cells, labels: values.map((x, i) => (x === '' ? (fixed[i] ?? '') : x)), caption })
    visitedSeries.push(s.nodes)
    cutoffSeries.push(s.cutoffs)
    last = s
  }
  if (last === undefined) throw new MdmlError('value', 'playGameSearch: the search did not start')
  const leaves = Math.max(1, ...tree.x) + 1
  const depth = Math.max(1, ...tree.depth)
  const dy = Math.max(1, (0.6 * leaves) / depth)
  const record: BoardSteps = boardSteps({
    method: algorithm,
    layout: {
      type: 'graph',
      nodes: tree.labels.map((_, i) => ({ x: tree.x[i] ?? 0, y: (tree.depth[i] ?? 0) * dy })),
      edges: tree.parent.flatMap((p, i) => (p < 0 ? [] : [[p, i] as [number, number]])),
    },
    categories: CATEGORIES,
    steps,
    metrics: [
      { name: 'nodes visited', values: visitedSeries },
      ...(algorithm === 'alpha-beta' ? [{ name: 'cutoffs', values: cutoffSeries }] : []),
    ],
    stepName: 'node',
    summary: `value ${last.value} after visiting ${last.nodes} of ${n} nodes`,
  })
  return {
    algorithm,
    action: last.principalVariation[0],
    value: last.value,
    principalVariation: last.principalVariation,
    visited: last.nodes,
    treeSize: n,
    cutoffs: last.cutoffs,
    exposition: [record],
  }
}
