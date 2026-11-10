/**
 * MAP by graph cuts for pairwise discrete models: the exact minimum of a binary model with submodular pair terms by one
 * minimum s-t cut (with roof duality, QPBO, as the fallback for terms that are not submodular), and the move-making
 * algorithms alpha-expansion and alpha-beta swap for models with more labels, which reduce each move to a binary cut
 * (Boykov, Veksler and Zabih 2001). Each move-making run is a traced algorithm with the energy after every cycle.
 */

import { OmaiError } from '../../base/index.ts'
import type { Json } from '../../base/index.ts'
import { type Algorithm, channels, type RunStatus, runAlgorithm, type TracePolicy, type Tracer } from '../../trace/index.ts'
import { type BinaryTerm, minimiseSubmodular, roofDuality, submodularMargin } from './_binary-cut.ts'
import {
  descendOver,
  energyOf,
  linkEnergy,
  mapResultOf,
  type PairwiseEnergy,
  pairEnergy,
  pairwiseEnergy,
  unaryArgmin,
} from './_pairwise.ts'
import type { DiscreteModel } from '../models/factor-graph.ts'
import type { MapResult } from './map.ts'

/** Tolerance for the inequalities that make a pair term submodular or regular, relative to the table's scale. */
function slack(table: Float64Array): number {
  let scale = 0
  for (const v of table) scale = Math.max(scale, Math.abs(v))
  return 1e-9 * Math.max(1, scale)
}

/** Options of {@link graphCutMap}. */
export interface GraphCutMapOptions {
  readonly evidence?: Readonly<Record<string, string | number>>
  /**
   * What to do with a pair term that is not submodular: `error` (default) throws, `qpbo` labels what roof duality can
   * decide, completes the rest by coordinate descent and reports the lower bound.
   */
  readonly nonSubmodular?: 'error' | 'qpbo'
}

/** The result of {@link graphCutMap}. */
export interface GraphCutMapResult extends MapResult {
  /** The energy (minus the log value) of the assignment. */
  readonly energy: number
  /** A lower bound on the minimum energy (equal to `energy` when the cut is exact). */
  readonly lowerBound: number
  /** True when the assignment is a certified minimum of the energy. */
  readonly exact: boolean
  /** Variables that roof duality left undecided (always empty for a submodular model). */
  readonly undecided: readonly string[]
}

/**
 * Exact MAP of a binary pairwise model by one minimum cut (Greig, Porteous and Seheult 1989; Kolmogorov and Zabih
 * 2004). Every pair term must be submodular, E(0,0) + E(1,1) <= E(0,1) + E(1,0), where E is minus the log value, and
 * every variable has two states; the energy is then the capacity of a cut of an s-t graph (unary terms on the source
 * and sink edges, the non-negative excess of each pair as an edge between the variables) and the minimum cut is the
 * MAP assignment. With `nonSubmodular: 'qpbo'` a model with other terms is handled by roof duality (QPBO): the doubled
 * graph of the variables and their complements is cut, a variable whose two copies agree on a label is persistent (some
 * MAP assignment has it), the rest are completed by coordinate descent and the cut value is a lower bound.
 *
 * @param model - A binary model of unary and pairwise factors (log values must be finite).
 * @param options - Evidence and what to do with a term that is not submodular.
 * @returns The assignment, its log value and energy, the lower bound, whether it is certified optimal and the
 *   variables roof duality could not decide.
 * @throws OmaiError with code `value` for a variable with other than two states, a factor over three or more variables, a
 *   non-finite log value, or (by default) a pair term that is not submodular, naming the pair.
 * @concept graph-cuts
 * @concept map-inference
 * @index function Exact MAP of a binary submodular pairwise model by a minimum cut; QPBO fallback.
 * @example
 * ```ts
 * import { graphCutMap, isingModel } from 'omai-engine/pgm'
 * const r = graphCutMap(isingModel({ rows: 2, cols: 2, coupling: 1, field: { s_0_0: 0.5 } }))
 * [r.exact, r.assignment.s_1_1, r.energy] // => [true, '+1', -4.5]
 * ```
 */
export function graphCutMap(model: DiscreteModel, options: GraphCutMapOptions = {}): GraphCutMapResult {
  const p = pairwiseEnergy(model, options.evidence ?? {}, 'graphCutMap')
  const n = p.names.length
  for (const [i, k] of p.card.entries())
    if (k !== 2) throw new OmaiError('value', `graphCutMap: ${p.names[i]} has ${k} states; binary models only (try alphaExpansion)`)
  const u0 = Float64Array.from(p.unary, (u) => u[0] ?? 0)
  const u1 = Float64Array.from(p.unary, (u) => u[1] ?? 0)
  const terms: BinaryTerm[] = p.pairs.map((t) => ({
    i: t.i,
    j: t.j,
    a: t.table[0] ?? 0,
    b: t.table[1] ?? 0,
    c: t.table[2] ?? 0,
    d: t.table[3] ?? 0,
  }))
  const bad = terms.findIndex((t, k) => submodularMargin(t) < -slack(p.pairs[k]?.table ?? new Float64Array(1)))
  if (bad < 0 || n === 0) {
    const { labels } = minimiseSubmodular(n, u0, u1, terms)
    const x = Int32Array.from(labels)
    const energy = energyOf(p, x)
    return { ...mapResultOf(model, p, x), energy, lowerBound: energy, exact: true, undecided: [] }
  }
  const t = terms[bad] as BinaryTerm
  if ((options.nonSubmodular ?? 'error') === 'error')
    throw new OmaiError(
      'value',
      `graphCutMap: the term between ${p.names[t.i]} and ${p.names[t.j]} is not submodular (E00 + E11 = ${t.a + t.d} > E01 + E10 = ${t.b + t.c}); use nonSubmodular: 'qpbo' or another MAP method`,
    )
  const { labels, lowerBound } = roofDuality(n, u0, u1, terms)
  const x = Int32Array.from(labels, (l) => Math.max(l, 0))
  const open: number[] = []
  for (const [i, l] of labels.entries()) if (l < 0) open.push(i)
  descendOver(p, x, open)
  const energy = energyOf(p, x)
  const exact = open.length === 0 || energy <= lowerBound + 1e-9 * Math.max(1, Math.abs(energy))
  return {
    ...mapResultOf(model, p, x),
    energy,
    lowerBound,
    exact,
    undecided: open.map((i) => p.names[i] ?? ''),
  }
}

/** Which conditions the pair terms of a model meet: see {@link pairwiseMoveConditions}. */
export interface PairwiseConditions {
  /** Every table is symmetric, non-negative and zero on the diagonal. */
  readonly semiMetric: boolean
  /** A semi-metric table that also obeys the triangle inequality. */
  readonly metric: boolean
  /** E(p, q) + E(a, a) <= E(p, a) + E(a, q) for every pair and states p, q, a: alpha-expansion moves are submodular. */
  readonly expansion: boolean
  /** E(a, a) + E(b, b) <= E(a, b) + E(b, a) for every pair and states a, b: alpha-beta swap moves are submodular. */
  readonly swap: boolean
  /** The first violation of each condition, naming the pair and the states. */
  readonly violations: {
    readonly semiMetric?: string
    readonly metric?: string
    readonly expansion?: string
    readonly swap?: string
  }
}

function conditionsOf(p: PairwiseEnergy): PairwiseConditions {
  const violations: { semiMetric?: string; metric?: string; expansion?: string; swap?: string } = {}
  for (const [index, t] of p.pairs.entries()) {
    const K = p.card[t.i] ?? 1
    if (K !== (p.card[t.j] ?? 1)) throw new OmaiError('value', 'pairwise conditions: the variables of a pair need the same states')
    const tol = slack(t.table)
    const at = (a: number, b: number): number => pairEnergy(p, index, a, b)
    const name = `${p.names[t.i]}-${p.names[t.j]}`
    for (let a = 0; a < K; a++) {
      if (Math.abs(at(a, a)) > tol) violations.semiMetric ??= `${name}: E(${a},${a}) = ${at(a, a)} is not zero`
      for (let b = 0; b < K; b++) {
        if (Math.abs(at(a, b) - at(b, a)) > tol) violations.semiMetric ??= `${name}: E(${a},${b}) != E(${b},${a})`
        if (at(a, b) < -tol) violations.semiMetric ??= `${name}: E(${a},${b}) = ${at(a, b)} is negative`
        if (at(a, a) + at(b, b) > at(a, b) + at(b, a) + tol)
          violations.swap ??= `${name}: E(${a},${a}) + E(${b},${b}) > E(${a},${b}) + E(${b},${a})`
        for (let c = 0; c < K; c++) {
          if (at(a, b) + at(c, c) > at(a, c) + at(c, b) + tol)
            violations.expansion ??= `${name}: E(${a},${b}) + E(${c},${c}) > E(${a},${c}) + E(${c},${b})`
          if (at(a, c) > at(a, b) + at(b, c) + tol) violations.metric ??= `${name}: E(${a},${c}) > E(${a},${b}) + E(${b},${c})`
        }
      }
    }
  }
  return {
    semiMetric: violations.semiMetric === undefined,
    metric: violations.semiMetric === undefined && violations.metric === undefined,
    expansion: violations.expansion === undefined,
    swap: violations.swap === undefined,
    violations,
  }
}

/**
 * Which of the conditions of the move-making methods a pairwise model meets, with the first violation of each. Energies
 * are minus the log values. A table is a semi-metric when it is symmetric, non-negative and zero on the diagonal, a
 * metric when it also obeys the triangle inequality. Alpha-expansion needs every move to be submodular, which
 * E(p, q) + E(a, a) <= E(p, a) + E(a, q) says for every pair (a metric satisfies it); alpha-beta swap needs
 * E(a, a) + E(b, b) <= E(a, b) + E(b, a) (a semi-metric satisfies it). Both are unchanged by adding a function of one
 * variable to a table, so a Potts or truncated-linear table plus unary parts passes though it is not zero on the
 * diagonal.
 *
 * @param model - A model of unary and pairwise factors with the same number of states on each variable of a pair.
 * @param options - Evidence.
 * @returns The four flags and the first violations.
 * @throws OmaiError with code `value` for a factor over three or more variables or a non-finite value.
 * @concept graph-cuts
 * @example
 * ```ts
 * import { pairwiseMoveConditions, pottsModel } from 'omai-engine/pgm'
 * const c = pairwiseMoveConditions(pottsModel({ rows: 2, cols: 2, q: 3, coupling: 1 }))
 * [c.expansion, c.swap] // => [true, true]
 * ```
 */
export function pairwiseMoveConditions(
  model: DiscreteModel,
  options: Pick<GraphCutMapOptions, 'evidence'> = {},
): PairwiseConditions {
  return conditionsOf(pairwiseEnergy(model, options.evidence ?? {}, 'pairwiseMoveConditions'))
}

/** Options of {@link alphaExpansionAlgorithm} and {@link alphaBetaSwapAlgorithm}. */
export interface MoveMakingOptions {
  readonly evidence?: Readonly<Record<string, string | number>>
  /** The starting state (indices) by variable; default each variable's lowest unary energy. */
  readonly init?: Readonly<Record<string, number>>
  /** The most cycles (default 50). A cycle is one expansion for every label, or one swap for every pair of labels. */
  readonly maxCycles?: number
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
}

/** The state of a move-making run. */
export interface MoveMakingState {
  readonly x: Int32Array
  readonly energy: number
  /** Moves accepted in the last cycle. */
  readonly moves: number
  /** The energy at the start and after every cycle. */
  readonly history: readonly number[]
}

function requireSameStates(p: PairwiseEnergy, caller: string): number {
  const K = p.card[0] ?? 2
  for (const [i, k] of p.card.entries())
    if (k !== K) throw new OmaiError('value', `${caller}: ${p.names[i]} has ${k} states but ${p.names[0]} has ${K}`)
  return K
}

function startOf(p: PairwiseEnergy, init: Readonly<Record<string, number>> | undefined): Int32Array {
  const x = unaryArgmin(p)
  for (const [i, v] of p.names.entries()) if (init?.[v] !== undefined) x[i] = init[v] as number
  return x
}

/** The tolerance below which a move does not count as lowering the energy. */
const gain = (energy: number): number => 1e-12 * Math.max(1, Math.abs(energy))

/** One alpha-expansion: every variable keeps its state or takes alpha, chosen by a minimum cut; null if no gain. */
function expansionMove(p: PairwiseEnergy, x: Int32Array, alpha: number, energy: number): Int32Array | null {
  const n = x.length
  const u0 = new Float64Array(n)
  const u1 = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    u0[i] = p.unary[i]?.[x[i] ?? 0] ?? 0
    u1[i] = p.unary[i]?.[alpha] ?? 0
  }
  const terms: BinaryTerm[] = p.pairs.map((t, k) => ({
    i: t.i,
    j: t.j,
    a: pairEnergy(p, k, x[t.i] ?? 0, x[t.j] ?? 0),
    b: pairEnergy(p, k, x[t.i] ?? 0, alpha),
    c: pairEnergy(p, k, alpha, x[t.j] ?? 0),
    d: pairEnergy(p, k, alpha, alpha),
  }))
  const { labels } = minimiseSubmodular(n, u0, u1, terms)
  const next = Int32Array.from(x, (s, i) => (labels[i] === 1 ? alpha : s))
  return energyOf(p, next) < energy - gain(energy) ? next : null
}

/** One alpha-beta swap: the variables in state alpha or beta may exchange them, chosen by a minimum cut. */
function swapMove(p: PairwiseEnergy, x: Int32Array, alpha: number, beta: number, energy: number): Int32Array | null {
  const active: number[] = []
  const slot = new Int32Array(x.length).fill(-1)
  for (let i = 0; i < x.length; i++)
    if (x[i] === alpha || x[i] === beta) {
      slot[i] = active.length
      active.push(i)
    }
  if (active.length === 0) return null
  const u0 = new Float64Array(active.length)
  const u1 = new Float64Array(active.length)
  for (const [m, i] of active.entries()) {
    u0[m] = p.unary[i]?.[alpha] ?? 0
    u1[m] = p.unary[i]?.[beta] ?? 0
    for (const link of p.neighbours[i] ?? []) {
      if ((slot[link.other] ?? -1) >= 0) continue
      u0[m] = (u0[m] ?? 0) + linkEnergy(p, link, alpha, x[link.other] ?? 0)
      u1[m] = (u1[m] ?? 0) + linkEnergy(p, link, beta, x[link.other] ?? 0)
    }
  }
  const terms: BinaryTerm[] = []
  for (const [k, t] of p.pairs.entries()) {
    const i = slot[t.i] ?? -1
    const j = slot[t.j] ?? -1
    if (i < 0 || j < 0) continue
    terms.push({
      i,
      j,
      a: pairEnergy(p, k, alpha, alpha),
      b: pairEnergy(p, k, alpha, beta),
      c: pairEnergy(p, k, beta, alpha),
      d: pairEnergy(p, k, beta, beta),
    })
  }
  const { labels } = minimiseSubmodular(active.length, u0, u1, terms)
  const next = Int32Array.from(x)
  for (const [m, i] of active.entries()) next[i] = labels[m] === 1 ? beta : alpha
  return energyOf(p, next) < energy - gain(energy) ? next : null
}

function moveMaking(
  name: string,
  model: DiscreteModel,
  options: MoveMakingOptions,
  condition: 'expansion' | 'swap',
  cycle: (p: PairwiseEnergy, x: Int32Array, energy: number, K: number) => { x: Int32Array; energy: number; moves: number },
): Algorithm<MoveMakingState> {
  const p = pairwiseEnergy(model, options.evidence ?? {}, name)
  const K = requireSameStates(p, name)
  const violation = conditionsOf(p).violations[condition]
  if (violation !== undefined)
    throw new OmaiError(
      'value',
      `${name}: the pair terms must satisfy the ${condition} condition (${violation}); see pairwiseMoveConditions`,
    )
  return {
    name,
    config: { variables: [...p.names], labels: K } as Json,
    schema: channels({
      metrics: {
        energy: { description: 'energy (minus the log value) after the cycle', direction: 'lower' },
        moves: { description: 'moves accepted in the cycle' },
      },
    }),
    init: () => {
      const x = startOf(p, options.init)
      const energy = energyOf(p, x)
      return { x, energy, moves: -1, history: [energy] }
    },
    step: (s) => {
      const r = cycle(p, s.x, s.energy, K)
      return { x: r.x, energy: r.energy, moves: r.moves, history: [...s.history, r.energy] }
    },
    done: (s, t) => (t > 0 && s.moves === 0 ? 'converged' : undefined),
    observe: (s) => ({ metrics: { energy: s.energy, moves: s.moves } }),
  }
}

/**
 * Alpha-expansion (Boykov, Veksler and Zabih 2001) as an {@link Algorithm}: each cycle visits every label alpha in turn
 * and lets every variable keep its state or switch to alpha, the best such move being the minimum cut of a binary
 * problem, accepted when it lowers the energy. The energy never rises, the run stops at a cycle with no move (a local
 * minimum with respect to expansions), and for a Potts model that minimum is within a factor of two of the global one.
 * The pair terms must satisfy the expansion condition (see {@link pairwiseMoveConditions}; a metric does).
 *
 * @param model - A model of unary and pairwise factors, every variable with the same number of states.
 * @param options - Evidence, start, cycle limit and tracing.
 * @returns The algorithm; its state holds the assignment, the energy and the energy after each cycle.
 * @throws OmaiError with code `value` when the expansion condition fails, naming the pair and states.
 * @concept graph-cuts
 * @concept map-inference
 * @example
 * ```ts
 * import { alphaExpansionAlgorithm, pottsModel } from 'omai-engine/pgm'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const r = runAlgorithm(alphaExpansionAlgorithm(pottsModel({ rows: 2, cols: 2, q: 3, coupling: 1 })), { maxSteps: 20 })
 * r.status // => 'converged'
 * ```
 */
export function alphaExpansionAlgorithm(model: DiscreteModel, options: MoveMakingOptions = {}): Algorithm<MoveMakingState> {
  return moveMaking('alphaExpansion', model, options, 'expansion', (p, x0, e0, K) => {
    let x = x0
    let energy = e0
    let moves = 0
    for (let alpha = 0; alpha < K; alpha++) {
      const next = expansionMove(p, x, alpha, energy)
      if (next === null) continue
      x = next
      energy = energyOf(p, x)
      moves++
    }
    return { x, energy, moves }
  })
}

/**
 * Alpha-beta swap (Boykov, Veksler and Zabih 2001) as an {@link Algorithm}: each cycle visits every pair of labels and
 * lets the variables currently in one of the two exchange them, the best exchange being a minimum cut, accepted when it
 * lowers the energy. It needs the weaker swap condition E(a, a) + E(b, b) <= E(a, b) + E(b, a) (a semi-metric
 * satisfies it) and gives a local minimum with respect to swaps, which has no approximation guarantee.
 *
 * @param model - A model of unary and pairwise factors, every variable with the same number of states.
 * @param options - Evidence, start, cycle limit and tracing.
 * @returns The algorithm; its state holds the assignment, the energy and the energy after each cycle.
 * @throws OmaiError with code `value` when the swap condition fails, naming the pair and states.
 * @concept graph-cuts
 * @concept map-inference
 * @example
 * ```ts
 * import { alphaBetaSwapAlgorithm, pottsModel } from 'omai-engine/pgm'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const r = runAlgorithm(alphaBetaSwapAlgorithm(pottsModel({ rows: 2, cols: 2, q: 3, coupling: 1 })), { maxSteps: 20 })
 * r.status // => 'converged'
 * ```
 */
export function alphaBetaSwapAlgorithm(model: DiscreteModel, options: MoveMakingOptions = {}): Algorithm<MoveMakingState> {
  return moveMaking('alphaBetaSwap', model, options, 'swap', (p, x0, e0, K) => {
    let x = x0
    let energy = e0
    let moves = 0
    for (let alpha = 0; alpha < K; alpha++)
      for (let beta = alpha + 1; beta < K; beta++) {
        const next = swapMove(p, x, alpha, beta, energy)
        if (next === null) continue
        x = next
        energy = energyOf(p, x)
        moves++
      }
    return { x, energy, moves }
  })
}

/** The result of a move-making run. */
export interface MoveMakingResult extends MapResult {
  /** The energy (minus the log value) of the assignment. */
  readonly energy: number
  /** The energy at the start and after every cycle. */
  readonly energyTrace: readonly number[]
  readonly cycles: number
  readonly status: RunStatus
  /** True when the last cycle found no move, a local minimum with respect to the moves. */
  readonly converged: boolean
}

function runMoves(
  algorithm: Algorithm<MoveMakingState>,
  model: DiscreteModel,
  options: MoveMakingOptions,
  caller: string,
): MoveMakingResult {
  const p = pairwiseEnergy(model, options.evidence ?? {}, caller)
  const run = runAlgorithm(algorithm, {
    maxSteps: options.maxCycles ?? 50,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
  const s = run.state
  return {
    ...mapResultOf(model, p, s.x),
    energy: s.energy,
    energyTrace: s.history,
    cycles: s.history.length - 1,
    status: run.status,
    converged: run.status === 'converged',
  }
}

/**
 * MAP by alpha-expansion ({@link alphaExpansionAlgorithm} run to a local minimum): the standard multi-label graph-cut
 * method for metric and Potts-like smoothness (stereo, segmentation, denoising), far better than iterated conditional
 * modes and within a factor of two of the optimum on a Potts model.
 *
 * @param model - A model of unary and pairwise factors, every variable with the same number of states.
 * @param options - Evidence, start, cycle limit and tracing.
 * @returns The assignment, its log value and energy, the energy after each cycle and whether it converged.
 * @throws OmaiError with code `value` when the expansion condition fails.
 * @concept graph-cuts
 * @concept map-inference
 * @index function MAP of a multi-label pairwise model by alpha-expansion graph cuts.
 * @example
 * ```ts
 * import { alphaExpansion, pottsModel } from 'omai-engine/pgm'
 * const r = alphaExpansion(pottsModel({ rows: 1, cols: 3, q: 3, coupling: 1, field: { s_0_0: 0.5 } }))
 * [r.converged, r.cycles >= 1] // => [true, true]
 * ```
 */
export function alphaExpansion(model: DiscreteModel, options: MoveMakingOptions = {}): MoveMakingResult {
  return runMoves(alphaExpansionAlgorithm(model, options), model, options, 'alphaExpansion')
}

/**
 * MAP by alpha-beta swap ({@link alphaBetaSwapAlgorithm} run to a local minimum): the move-making method for
 * semi-metric smoothness terms that violate the triangle inequality.
 *
 * @param model - A model of unary and pairwise factors, every variable with the same number of states.
 * @param options - Evidence, start, cycle limit and tracing.
 * @returns The assignment, its log value and energy, the energy after each cycle and whether it converged.
 * @throws OmaiError with code `value` when the swap condition fails.
 * @concept graph-cuts
 * @concept map-inference
 * @index function MAP of a multi-label pairwise model by alpha-beta swap graph cuts.
 * @example
 * ```ts
 * import { alphaBetaSwap, pottsModel } from 'omai-engine/pgm'
 * alphaBetaSwap(pottsModel({ rows: 1, cols: 3, q: 3, coupling: 1 })).converged // => true
 * ```
 */
export function alphaBetaSwap(model: DiscreteModel, options: MoveMakingOptions = {}): MoveMakingResult {
  return runMoves(alphaBetaSwapAlgorithm(model, options), model, options, 'alphaBetaSwap')
}
