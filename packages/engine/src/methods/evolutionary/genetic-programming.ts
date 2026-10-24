// Symbolic regression by genetic programming, played generation by generation: the best program of each generation as
// a simplified symbolic expression and its curve over the data, from the core's genetic programming
// (whitebox-engine/search) and computer algebra (whitebox-engine/symbolic).
import { WhiteboxError } from '../../lib/base/index.ts'
import { type Explained, type Generation, populationSteps } from '../../lib/exposition/index.ts'
import { type Key, key as makeKey } from '../../lib/random/index.ts'
import {
  arithmeticPrimitives,
  type ExprTree,
  evaluateTree,
  formatTree,
  geneticProgrammingAlgorithm,
  type Primitive,
  type PrimitiveSet,
  regressionFitness,
  treeSize,
} from '../../lib/search/index.ts'
import { add, cos, div, type Expr, mul, num, simplify, sin, sub, sym, toText } from '../../lib/symbolic/index.ts'
import { iterate } from '../../lib/trace/index.ts'

/** The target functions symbolic regression can be set. */
export const SYMBOLIC_TARGETS: Readonly<Record<'quartic' | 'cubic' | 'sine' | 'rational', (x: number) => number>> = {
  // Koza's quartic.
  quartic: (x) => x ** 4 + x ** 3 + x ** 2 + x,
  cubic: (x) => x ** 3 - x,
  sine: (x) => Math.sin(2 * x) + x / 2,
  rational: (x) => 1 / (1 + x * x),
}

/** The names of the targets. */
export type SymbolicTarget = keyof typeof SYMBOLIC_TARGETS

const TRIG: readonly Primitive[] = [
  { name: 'sin', arity: 1, apply: ([a = 0]) => Math.sin(a) },
  { name: 'cos', arity: 1, apply: ([a = 0]) => Math.cos(a) },
]

/**
 * An expression tree as a symbolic expression (protected division reads as plain division), simplified.
 *
 * @param tree - The tree over `add`, `sub`, `mul`, `div`, `sin` and `cos`.
 * @returns The expression.
 */
export function treeToExpr(tree: ExprTree): Expr {
  const go = (t: ExprTree): Expr => {
    if ('constant' in t) return num(Number(t.constant.toFixed(2)))
    if ('variable' in t) return sym(t.variable)
    const [a, b] = t.children.map(go)
    const x = a ?? num(0)
    const y = b ?? num(0)
    switch (t.op) {
      case 'add':
        return add(x, y)
      case 'sub':
        return sub(x, y)
      case 'mul':
        return mul(x, y)
      case 'div':
        return div(x, y)
      case 'sin':
        return sin(x)
      case 'cos':
        return cos(x)
      default:
        throw new WhiteboxError('value', `treeToExpr: unknown function ${t.op}`)
    }
  }
  return go(tree)
}

function textOf(tree: ExprTree): string {
  try {
    return toText(simplify(treeToExpr(tree)))
  } catch {
    return formatTree(tree)
  }
}

/** Options of {@link symbolicRegression}. */
export interface SymbolicRegressionOptions {
  /** The function to recover (default the quartic). */
  readonly target?: SymbolicTarget
  /** Points sampled on [-1, 1] (default 20). */
  readonly points?: number
  /** Add sine and cosine to the arithmetic functions (default false). */
  readonly trig?: boolean
  readonly populationSize?: number
  readonly generations?: number
  /** The fitness penalty per node, against bloat (default 0.001). */
  readonly parsimony?: number
  readonly key?: Key
}

/** What symbolic regression found. */
export interface SymbolicRegressionResult {
  /** The best program as text, simplified, and as a tree. */
  readonly expression: string
  readonly tree: ExprTree
  /** Its mean squared error on the data. */
  readonly mse: number
  readonly size: number
  readonly generations: number
}

/**
 * Symbolic regression by genetic programming: evolve expression trees over `x`, the arithmetic functions (and
 * optionally sine and cosine) and constants to fit points of a target function, recording every generation: each
 * program's error, and the best program as a simplified expression with its curve over the data.
 *
 * @param options - Target, points, functions, population, generations, parsimony and key.
 * @returns The best expression and its error, with a `populationSteps` record.
 */
export function symbolicRegression(options: SymbolicRegressionOptions = {}): Explained<SymbolicRegressionResult> {
  const target = SYMBOLIC_TARGETS[options.target ?? 'quartic']
  if (target === undefined) throw new WhiteboxError('value', `symbolicRegression: unknown target '${options.target}'`)
  const m = options.points ?? 20
  const xs = Array.from({ length: m }, (_, i) => -1 + (2 * i) / Math.max(1, m - 1))
  const ys = xs.map(target)
  const functions = options.trig === true ? [...arithmeticPrimitives(), ...TRIG] : arithmeticPrimitives()
  const set: PrimitiveSet = { functions, variables: ['x'], constants: { low: -1, high: 1 } }
  const parsimony = options.parsimony ?? 0.001
  const fitness = regressionFitness(
    functions,
    xs.map((x) => ({ x })),
    ys,
    parsimony,
  )
  const algorithm = geneticProgrammingAlgorithm(set, fitness, {
    populationSize: options.populationSize ?? 200,
    maxDepth: 8,
    initialDepth: 4,
  })
  const grid = Array.from({ length: 101 }, (_, i) => -1.1 + (2.2 * i) / 100)
  const mseOf = (t: ExprTree): number => {
    let s = 0
    for (const [i, x] of xs.entries()) s += (evaluateTree(t, functions, { x }) - (ys[i] ?? 0)) ** 2
    return s / m
  }
  const gens: Generation[] = []
  let best: ExprTree | undefined
  let generation = 0
  let lastText = ''
  for (const { state } of iterate(algorithm, { key: options.key ?? makeKey(0), maxSteps: options.generations ?? 40 })) {
    const changed = best !== state.best
    best = state.best
    generation = state.generation
    if (changed) lastText = textOf(state.best)
    const err = mseOf(state.best)
    gens.push({
      // Each program's error, its fitness without the parsimony penalty.
      fitness: state.fitness.map((f, i) => -f - parsimony * treeSize(state.population[i] as ExprTree)),
      best: {
        fitness: err,
        label: `${lastText}  (mse ${Number(err.toPrecision(3))})`,
        curve: grid.map((x) => evaluateTree(state.best, functions, { x })),
      },
    })
    if (err < 1e-12) break
  }
  if (best === undefined) throw new WhiteboxError('value', 'symbolicRegression: no generations')
  const record = populationSteps({
    method: 'genetic programming',
    lowerIsBetter: true,
    data: { x: xs, y: ys, grid },
    generations: gens,
  })
  return {
    expression: textOf(best),
    tree: best,
    mse: mseOf(best),
    size: treeSize(best),
    generations: generation,
    exposition: [record],
  }
}
