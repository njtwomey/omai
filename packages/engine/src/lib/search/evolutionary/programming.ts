/**
 * Genetic programming (Koza): evolving programs represented as expression trees. Primitive sets with the arithmetic
 * primitives of symbolic regression, random trees by the grow and full methods and ramped half-and-half initialisation,
 * evaluation and printing, subtree crossover and subtree and point mutation, a regression fitness with a parsimony
 * penalty, and the traced evolutionary loop.
 */

import { OmaiError } from '../../base/index.ts'
import type { Key } from '../../random/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import { type Draw, draws } from '../_draw.ts'
import {
  type EvolutionResult,
  type GeneticOptions,
  type GeneticState,
  geneticAlgorithm,
  geneticAlgorithmAlgorithm,
  type Representation,
} from './genetic.ts'

/** An expression tree for genetic programming: a function node, a variable or a constant. Plain, serialisable data. */
export type ExprTree =
  | { readonly op: string; readonly children: readonly ExprTree[] }
  | { readonly variable: string }
  | { readonly constant: number }

/** A function of the primitive set: its name, arity and meaning. */
export interface GpPrimitive {
  readonly name: string
  readonly arity: number
  readonly apply: (args: readonly number[]) => number
}

/** The language of a genetic program: functions, variables, and constants (a list, or a range for ephemeral ones). */
export interface PrimitiveSet {
  readonly functions: readonly GpPrimitive[]
  readonly variables: readonly string[]
  readonly constants?: readonly number[] | { readonly low: number; readonly high: number }
}

/**
 * The arithmetic primitives of Koza's symbolic regression: add, sub, mul and protected division (1 when the divisor
 * is within 1e-9 of 0).
 *
 * @returns The four primitives.
 * @concept genetic-programming
 * @example
 * ```ts
 * import { arithmeticPrimitives } from 'omai-engine/search'
 * arithmeticPrimitives().map((p) => p.apply([6, 0])) // => [6, 6, 0, 1]
 * ```
 */
export function arithmeticPrimitives(): GpPrimitive[] {
  return [
    { name: 'add', arity: 2, apply: ([a = 0, b = 0]) => a + b },
    { name: 'sub', arity: 2, apply: ([a = 0, b = 0]) => a - b },
    { name: 'mul', arity: 2, apply: ([a = 0, b = 0]) => a * b },
    { name: 'div', arity: 2, apply: ([a = 0, b = 0]) => (Math.abs(b) < 1e-9 ? 1 : a / b) },
  ]
}

function terminal(d: Draw, set: PrimitiveSet): ExprTree {
  const c = set.constants
  const nConst = c === undefined ? 0 : Array.isArray(c) ? c.length : 1
  const k = d.int(set.variables.length + nConst)
  if (k < set.variables.length) return { variable: set.variables[k] as string }
  if (Array.isArray(c)) return { constant: c[k - set.variables.length] as number }
  const range = c as { low: number; high: number }
  return { constant: range.low + d.uniform() * (range.high - range.low) }
}

function grow(d: Draw, set: PrimitiveSet, depth: number, full: boolean): ExprTree {
  const terminals = set.variables.length + (set.constants === undefined ? 0 : 1)
  const pickTerminal = depth <= 0 || (!full && d.uniform() < terminals / (terminals + set.functions.length))
  if (pickTerminal || set.functions.length === 0) return terminal(d, set)
  const f = set.functions[d.int(set.functions.length)] as GpPrimitive
  return { op: f.name, children: Array.from({ length: f.arity }, () => grow(d, set, depth - 1, full)) }
}

function checkSet(set: PrimitiveSet, name: string): void {
  if (set.variables.length === 0 && set.constants === undefined) {
    throw new OmaiError('value', `${name}: the primitive set needs a variable or constants`)
  }
}

/**
 * A random expression tree by Koza's grow method (any node may be a terminal before the depth limit) or full method
 * (functions all the way to the depth limit).
 *
 * @param key - The random key.
 * @param set - The primitive set.
 * @param options - The depth (default 4) and the method (default grow).
 * @returns The tree.
 * @throws OmaiError with code `value` when the set has no terminals.
 * @concept genetic-programming
 * @example
 * ```ts
 * import { key } from 'omai-engine/random'
 * import { arithmeticPrimitives, randomTree, treeDepth } from 'omai-engine/search'
 * treeDepth(randomTree(key(0), { functions: arithmeticPrimitives(), variables: ['x'] }, { depth: 3, method: 'full' })) // => 3
 * ```
 */
export function randomTree(
  key: Key,
  set: PrimitiveSet,
  options: { readonly depth?: number; readonly method?: 'grow' | 'full' } = {},
): ExprTree {
  checkSet(set, 'randomTree')
  return grow(draws(key), set, options.depth ?? 4, options.method === 'full')
}

/**
 * Ramped half-and-half initialisation (Koza 1992): trees of depths spread evenly over [minDepth, maxDepth], half by
 * the full method and half by grow.
 *
 * @param key - The random key.
 * @param set - The primitive set.
 * @param n - The number of trees.
 * @param options - The depth range (default 2 to 6).
 * @returns The trees.
 * @throws OmaiError with code `value` when the set has no terminals or the range is empty.
 * @concept genetic-programming
 * @example
 * ```ts
 * import { key } from 'omai-engine/random'
 * import { arithmeticPrimitives, rampedHalfAndHalf } from 'omai-engine/search'
 * rampedHalfAndHalf(key(0), { functions: arithmeticPrimitives(), variables: ['x'] }, 10).length // => 10
 * ```
 */
export function rampedHalfAndHalf(
  key: Key,
  set: PrimitiveSet,
  n: number,
  options: { readonly minDepth?: number; readonly maxDepth?: number } = {},
): ExprTree[] {
  checkSet(set, 'rampedHalfAndHalf')
  const lo = options.minDepth ?? 2
  const hi = options.maxDepth ?? 6
  if (!(hi >= lo && lo >= 0)) throw new OmaiError('value', 'rampedHalfAndHalf: need 0 <= minDepth <= maxDepth')
  const d = draws(key)
  return Array.from({ length: n }, (_, i) => grow(d, set, lo + (Math.floor(i / 2) % (hi - lo + 1)), i % 2 === 0))
}

/**
 * Evaluate an expression tree.
 *
 * @param tree - The tree.
 * @param functions - The functions its nodes name.
 * @param env - The variables' values.
 * @returns The value.
 * @throws OmaiError with code `value` when a function or variable is unknown.
 * @concept genetic-programming
 * @example
 * ```ts
 * import { arithmeticPrimitives, evaluateTree } from 'omai-engine/search'
 * const t = { op: 'add', children: [{ variable: 'x' }, { op: 'mul', children: [{ variable: 'x' }, { constant: 2 }] }] }
 * evaluateTree(t, arithmeticPrimitives(), { x: 3 }) // => 9
 * ```
 */
export function evaluateTree(
  tree: ExprTree,
  functions: readonly GpPrimitive[],
  env: Readonly<Record<string, number>>,
): number {
  if ('constant' in tree) return tree.constant
  if ('variable' in tree) {
    const v = env[tree.variable]
    if (v === undefined) throw new OmaiError('value', `evaluateTree: variable ${tree.variable} has no value`)
    return v
  }
  const f = functions.find((p) => p.name === tree.op)
  if (f === undefined) throw new OmaiError('value', `evaluateTree: unknown function ${tree.op}`)
  return f.apply(tree.children.map((c) => evaluateTree(c, functions, env)))
}

/**
 * The number of nodes of a tree.
 *
 * @param tree - The tree.
 * @returns Its size.
 * @example
 * ```ts
 * import { treeSize } from 'omai-engine/search'
 * treeSize({ op: 'add', children: [{ variable: 'x' }, { constant: 1 }] }) // => 3
 * ```
 */
export function treeSize(tree: ExprTree): number {
  return 'op' in tree ? 1 + tree.children.reduce((a, c) => a + treeSize(c), 0) : 1
}

/**
 * The depth of a tree (a terminal has depth 0).
 *
 * @param tree - The tree.
 * @returns Its depth.
 * @example
 * ```ts
 * import { treeDepth } from 'omai-engine/search'
 * treeDepth({ op: 'add', children: [{ variable: 'x' }, { constant: 1 }] }) // => 1
 * ```
 */
export function treeDepth(tree: ExprTree): number {
  return 'op' in tree ? 1 + tree.children.reduce((a, c) => Math.max(a, treeDepth(c)), 0) : 0
}

/**
 * A tree as an s-expression.
 *
 * @param tree - The tree.
 * @returns The text, such as `(add x 1)`.
 * @example
 * ```ts
 * import { formatTree } from 'omai-engine/search'
 * formatTree({ op: 'add', children: [{ variable: 'x' }, { constant: 1 }] }) // => '(add x 1)'
 * ```
 */
export function formatTree(tree: ExprTree): string {
  if ('constant' in tree) return String(Number(tree.constant.toPrecision(6)))
  if ('variable' in tree) return tree.variable
  return `(${[tree.op, ...tree.children.map(formatTree)].join(' ')})`
}

/** The subtree at a preorder index, and a copy of the tree with that subtree replaced. */
function nodeAt(tree: ExprTree, index: number): ExprTree {
  let i = index
  const walk = (t: ExprTree): ExprTree | undefined => {
    if (i === 0) return t
    i -= 1
    if (!('op' in t)) return undefined
    for (const c of t.children) {
      const found = walk(c)
      if (found !== undefined) return found
    }
    return undefined
  }
  return walk(tree) ?? tree
}

function replaceAt(tree: ExprTree, index: number, by: ExprTree): ExprTree {
  let i = index
  const walk = (t: ExprTree): ExprTree => {
    if (i === 0) {
      i -= 1
      return by
    }
    i -= 1
    if (!('op' in t)) return t
    return { op: t.op, children: t.children.map(walk) }
  }
  return walk(tree)
}

/**
 * Subtree crossover (Koza 1992): swap a random subtree of each parent, keeping the parent when a child would exceed the
 * depth limit.
 *
 * @param key - The random key.
 * @param a - A parent.
 * @param b - Another.
 * @param options - The depth limit (default 17).
 * @returns Two children.
 * @concept genetic-programming
 * @example
 * ```ts
 * import { key } from 'omai-engine/random'
 * import { subtreeCrossover, treeSize } from 'omai-engine/search'
 * const a = { op: 'add', children: [{ variable: 'x' }, { constant: 1 }] }
 * const b = { op: 'mul', children: [{ variable: 'y' }, { variable: 'y' }] }
 * const [c, d] = subtreeCrossover(key(0), a, b)
 * treeSize(c) + treeSize(d) // => 6
 * ```
 */
export function subtreeCrossover(
  key: Key,
  a: ExprTree,
  b: ExprTree,
  options: { readonly maxDepth?: number } = {},
): [ExprTree, ExprTree] {
  const maxDepth = options.maxDepth ?? 17
  const d = draws(key)
  const i = d.int(treeSize(a))
  const j = d.int(treeSize(b))
  const c = replaceAt(a, i, nodeAt(b, j))
  const e = replaceAt(b, j, nodeAt(a, i))
  return [treeDepth(c) <= maxDepth ? c : a, treeDepth(e) <= maxDepth ? e : b]
}

/**
 * Subtree mutation: replace a random subtree by a fresh random tree (grow method), keeping the parent when the result
 * would exceed the depth limit.
 *
 * @param key - The random key.
 * @param tree - The tree.
 * @param set - The primitive set.
 * @param options - The depth of new subtrees (default 3) and the depth limit (default 17).
 * @returns The mutated tree.
 * @concept genetic-programming
 * @example
 * ```ts
 * import { key } from 'omai-engine/random'
 * import { arithmeticPrimitives, subtreeMutation, treeDepth } from 'omai-engine/search'
 * const t = subtreeMutation(key(1), { variable: 'x' }, { functions: arithmeticPrimitives(), variables: ['x'] }, { depth: 2 })
 * treeDepth(t) <= 2 // => true
 * ```
 */
export function subtreeMutation(
  key: Key,
  tree: ExprTree,
  set: PrimitiveSet,
  options: { readonly depth?: number; readonly maxDepth?: number } = {},
): ExprTree {
  checkSet(set, 'subtreeMutation')
  const d = draws(key)
  const out = replaceAt(tree, d.int(treeSize(tree)), grow(d, set, options.depth ?? 3, false))
  return treeDepth(out) <= (options.maxDepth ?? 17) ? out : tree
}

/**
 * Point mutation: each node is replaced, with probability `rate`, by another primitive of the same arity (a function
 * by a function, a terminal by a terminal).
 *
 * @param key - The random key.
 * @param tree - The tree.
 * @param set - The primitive set.
 * @param rate - The per-node probability (default 1 / size).
 * @returns The mutated tree (same shape).
 * @concept genetic-programming
 * @example
 * ```ts
 * import { key } from 'omai-engine/random'
 * import { arithmeticPrimitives, pointMutation, treeSize } from 'omai-engine/search'
 * const t = { op: 'add', children: [{ variable: 'x' }, { constant: 1 }] }
 * treeSize(pointMutation(key(0), t, { functions: arithmeticPrimitives(), variables: ['x'], constants: [1, 2] }, 1)) // => 3
 * ```
 */
export function pointMutation(
  key: Key,
  tree: ExprTree,
  set: PrimitiveSet,
  rate: number = 1 / treeSize(tree),
): ExprTree {
  checkSet(set, 'pointMutation')
  const d = draws(key)
  const walk = (t: ExprTree): ExprTree => {
    if (!('op' in t)) return d.uniform() < rate ? terminal(d, set) : t
    const same = set.functions.filter((f) => f.arity === t.children.length)
    const op = d.uniform() < rate && same.length > 0 ? (same[d.int(same.length)] as GpPrimitive).name : t.op
    return { op, children: t.children.map(walk) }
  }
  return walk(tree)
}

/**
 * The tree representation for a genetic algorithm: ramped half-and-half random trees, subtree crossover and subtree
 * mutation within a depth limit.
 *
 * @param set - The primitive set.
 * @param options - The initial depth range (default 2 to 6) and the depth limit (default 17).
 * @returns The representation.
 * @concept genetic-programming
 * @example
 * ```ts
 * import { key } from 'omai-engine/random'
 * import { arithmeticPrimitives, treeDepth, trees } from 'omai-engine/search'
 * treeDepth(trees({ functions: arithmeticPrimitives(), variables: ['x'] }, { maxDepth: 4 }).random(key(0))) <= 6 // => true
 * ```
 */
export function trees(
  set: PrimitiveSet,
  options: { readonly minDepth?: number; readonly initialDepth?: number; readonly maxDepth?: number } = {},
): Representation<ExprTree> {
  checkSet(set, 'trees')
  const lo = options.minDepth ?? 2
  const hi = options.initialDepth ?? 6
  const maxDepth = options.maxDepth ?? 17
  return {
    random: (k) => {
      const d = draws(k)
      return grow(d, set, lo + d.int(hi - lo + 1), d.uniform() < 0.5)
    },
    crossover: (k, a, b) => subtreeCrossover(k, a, b, { maxDepth }),
    mutate: (k, t) => subtreeMutation(k, t, set, { maxDepth }),
  }
}

/**
 * The fitness of a tree for symbolic regression: minus the mean squared error on the data, minus a parsimony
 * penalty per node (against bloat). Non-finite predictions score -Infinity.
 *
 * @param functions - The functions trees may use.
 * @param inputs - One environment of variable values per sample.
 * @param targets - The target per sample.
 * @param parsimony - The penalty per node (default 0).
 * @returns The fitness function.
 * @throws OmaiError with code `shape` when inputs and targets differ in length.
 * @concept genetic-programming
 * @example
 * ```ts
 * import { arithmeticPrimitives, regressionFitness } from 'omai-engine/search'
 * const f = regressionFitness(arithmeticPrimitives(), [{ x: 1 }, { x: 2 }], [1, 4])
 * f({ op: 'mul', children: [{ variable: 'x' }, { variable: 'x' }] }) // => 0
 * ```
 */
export function regressionFitness(
  functions: readonly GpPrimitive[],
  inputs: readonly Readonly<Record<string, number>>[],
  targets: readonly number[],
  parsimony = 0,
): (tree: ExprTree) => number {
  if (inputs.length !== targets.length) throw new OmaiError('shape', 'regressionFitness: inputs and targets differ')
  return (tree) => {
    let sse = 0
    for (let i = 0; i < inputs.length; i++) {
      const e = evaluateTree(tree, functions, inputs[i] ?? {}) - (targets[i] ?? 0)
      sse += e * e
    }
    const mse = sse / Math.max(1, inputs.length)
    return Number.isFinite(mse) ? -mse - parsimony * treeSize(tree) : Number.NEGATIVE_INFINITY
  }
}

/** Options of {@link geneticProgramming}. */
export interface ProgrammingOptions extends GeneticOptions {
  readonly minDepth?: number
  readonly initialDepth?: number
  readonly maxDepth?: number
}

/**
 * Genetic programming (Koza 1992) as a traced algorithm: a genetic algorithm over expression trees with ramped
 * half-and-half initialisation, tournament selection, subtree crossover and (by default rarely) subtree mutation.
 *
 * @param set - The primitive set.
 * @param fitness - The fitness of a tree (maximised).
 * @param options - Genetic-algorithm options (mutation rate default 0.1) and depth limits.
 * @returns The algorithm.
 * @concept genetic-programming
 * @example
 * ```ts
 * import { arithmeticPrimitives, geneticProgrammingAlgorithm } from 'omai-engine/search'
 * geneticProgrammingAlgorithm({ functions: arithmeticPrimitives(), variables: ['x'] }, () => 0).name // => 'genetic-algorithm'
 * ```
 */
export function geneticProgrammingAlgorithm(
  set: PrimitiveSet,
  fitness: (tree: ExprTree) => number,
  options: ProgrammingOptions = {},
): Algorithm<GeneticState<ExprTree>> {
  const { minDepth, initialDepth, maxDepth, ...ga } = options
  const representation = trees(set, {
    ...(minDepth === undefined ? {} : { minDepth }),
    ...(initialDepth === undefined ? {} : { initialDepth }),
    ...(maxDepth === undefined ? {} : { maxDepth }),
  })
  return geneticAlgorithmAlgorithm(representation, fitness, { mutationRate: 0.1, ...ga })
}

/**
 * Genetic programming.
 *
 * @param set - The primitive set.
 * @param fitness - The fitness of a tree (maximised), such as {@link regressionFitness}.
 * @param options - Population size (default 50), generations (default 100), rates, depth limits, the target, a seed
 * or key, and tracing.
 * @returns The best tree, its fitness and the best fitness per generation.
 * @concept genetic-programming
 * @example
 * ```ts
 * import { arithmeticPrimitives, evaluateTree, geneticProgramming, regressionFitness } from 'omai-engine/search'
 * const xs = [-2, -1, 0, 1, 2, 3].map((x) => ({ x }))
 * const fitness = regressionFitness(arithmeticPrimitives(), xs, xs.map(({ x }) => x * x + x))
 * const r = geneticProgramming({ functions: arithmeticPrimitives(), variables: ['x'] }, fitness, { target: -1e-9, populationSize: 200, generations: 50, seed: 1 })
 * evaluateTree(r.best, arithmeticPrimitives(), { x: 5 }) // => 30
 * ```
 */
export function geneticProgramming(
  set: PrimitiveSet,
  fitness: (tree: ExprTree) => number,
  options: ProgrammingOptions = {},
): EvolutionResult<ExprTree> {
  const { minDepth, initialDepth, maxDepth, ...ga } = options
  const representation = trees(set, {
    ...(minDepth === undefined ? {} : { minDepth }),
    ...(initialDepth === undefined ? {} : { initialDepth }),
    ...(maxDepth === undefined ? {} : { maxDepth }),
  })
  return geneticAlgorithm(representation, fitness, { mutationRate: 0.1, ...ga })
}
