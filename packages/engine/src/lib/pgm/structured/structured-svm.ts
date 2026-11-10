/**
 * Max-margin structured learning over any `StructuredOracle`: the structured SVM with margin rescaling (Taskar et al.
 * 2003; Tsochantaridis et al. 2005), minimising the penalised structured hinge loss
 * `c2 ||w||^2 + sum_i max_y [ loss(y_i, y) + w . phi(x_i, y) - w . phi(x_i, y_i) ]`. The inner maximum is the oracle's
 * loss-augmented decoding (`decode(w, x, { against: y })`), so the same four solvers train a chain CRF, a graph CRF or
 * any later oracle: the stochastic subgradient method (Ratliff, Bagnell and Zinkevich 2007, with Pegasos' step size),
 * block-coordinate Frank-Wolfe (Lacoste-Julien et al. 2013) with its duality gap, and the n-slack (Tsochantaridis et al.
 * 2005) and one-slack (Joachims, Finley and Yu 2009) cutting-plane methods, whose dual QP over the working set of
 * most-violated constraints is solved by the active-set or interior-point solver of `optim`. They are modes of a solver
 * slot beside the other learners of `structuredSolvers` (`structuredSvmModes`, `structuredSolversWith(structuredSvmModes)`),
 * and `structuredSvmSolvers` is the table of just these four, with `trainStructuredSvm` to run one to the end.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import { ddot, dnrm2 } from '../../numerics/linalg/index.ts'
import {
  qpActiveSet,
  qpArrays,
  qpInteriorPoint,
  type SlotState,
  type SolverReached,
  slotAlgorithm,
  type VectorParams,
} from '../../optim/index.ts'
import { foldIn, key as makeKey, permutation } from '../../random/index.ts'
import { bool, choice, dict, real, type SpaceValue } from '../../space/index.ts'
import { toFlat } from '../../tensor/index.ts'
import {
  type Algorithm,
  type ModeSpec,
  type ModeTable,
  modes,
  runAlgorithm,
  stateFromJson,
  stateToJson,
} from '../../trace/index.ts'
import type { StructuredExample, StructuredOracle, StructuredPenalty, StructuredProblem } from './oracle.ts'
import {
  type AnyStructuredProblem,
  type StructuredMode,
  type StructuredTrainView,
  structuredTrainMetrics,
  structuredTrainView,
} from './solvers.ts'

/**
 * The penalty of a structured SVM written in pystruct's (and SVM-struct's) regularisation parameter C. Their objective
 * is `1/2 ||w||^2 + C sum_i H_i(w)`; ours is `c2 ||w||^2 + sum_i H_i(w)`, the same up to the factor 2 c2, so
 * `c2 = 1 / (2 C)`.
 *
 * @param C - The weight of the summed hinge losses against `1/2 ||w||^2`.
 * @returns The penalty to give a {@link StructuredProblem}.
 * @throws OmaiError with code `value` when `C` is not positive.
 * @concept structured-svm
 * @example
 * ```ts
 * import { structuredSvmPenalty } from 'omai-engine/pgm'
 * structuredSvmPenalty(0.5) // => { c2: 1 }
 * ```
 */
export function structuredSvmPenalty(C: number): StructuredPenalty {
  if (!(C > 0)) throw new OmaiError('value', 'structuredSvmPenalty: C must be positive')
  return { c2: 1 / (2 * C) }
}

/** The most violated constraint of one example at the weights: the feature-count difference, its loss and its hinge. */
interface Separation {
  /** phi(x, y) - phi(x, y_hat): the constraint says w . psi >= loss - slack. */
  readonly psi: Float64Array
  readonly loss: number
  /** loss - w . psi, the example's hinge loss (zero when the gold labelling is the best answer). */
  readonly hinge: number
}

/**
 * The separation oracle of one example: the loss-augmented decoding at `w` and the constraint it yields. The gold
 * labelling is always a candidate (psi 0, loss 0), so an inexact decoder never reports a negative hinge loss.
 */
function separate<X, Y>(oracle: StructuredOracle<X, Y>, example: StructuredExample<X, Y>, w: Float64Array): Separation {
  const { x, y } = example
  const found = oracle.decode(w, x, { against: y })
  const loss = oracle.loss(y, found.y)
  const psi = Float64Array.from(oracle.features(x, y))
  const guess = oracle.features(x, found.y)
  for (let q = 0; q < psi.length; q++) psi[q] = (psi[q] ?? 0) - (guess[q] ?? 0)
  const hinge = loss - ddot(w, psi)
  if (!(hinge > 0)) return { psi: new Float64Array(psi.length), loss: 0, hinge: 0 }
  return { psi, loss, hinge }
}

interface HingePass {
  readonly value: number
  readonly hinge: number
  readonly grad: Float64Array
  readonly violated: number
  readonly items: readonly Separation[]
}

function hingePass<X, Y>(
  oracle: StructuredOracle<X, Y>,
  data: readonly StructuredExample<X, Y>[],
  w: Float64Array,
  c2: number,
): HingePass {
  const grad = new Float64Array(oracle.size)
  const items: Separation[] = []
  let hinge = 0
  let violated = 0
  for (const example of data) {
    const s = separate(oracle, example, w)
    items.push(s)
    hinge += s.hinge
    if (s.hinge > 0) violated++
    for (let q = 0; q < grad.length; q++) grad[q] = (grad[q] ?? 0) - (s.psi[q] ?? 0)
  }
  for (let q = 0; q < grad.length; q++) grad[q] = (grad[q] ?? 0) + 2 * c2 * (w[q] ?? 0)
  return { value: c2 * ddot(w, w) + hinge, hinge, grad, violated, items }
}

/**
 * The structured hinge objective (margin rescaling) of labelled examples under an oracle, `c2 ||w||^2` plus the sum
 * over the examples of `max_y [loss(y_i, y) + w . phi(x_i, y)] - w . phi(x_i, y_i)` (the loss-augmented decoding of the
 * oracle), and a subgradient of it.
 *
 * @param oracle - The structured model.
 * @param data - Labelled examples.
 * @param w - The weights (the point of evaluation).
 * @param penalty - The L2 coefficient c2 (the L1 term is not part of the structured SVM).
 * @returns The objective, a subgradient (`2 c2 w` minus the summed constraint vectors) and the summed hinge losses.
 * @concept structured-svm
 * @index function Structured hinge loss (margin rescaling) of an oracle with its subgradient.
 * @example
 * ```ts
 * import { chainOracle, linearChainCrf, structuredHingeObjective } from 'omai-engine/pgm'
 * const oracle = chainOracle(linearChainCrf(['A', 'B'], ['f']))
 * // at zero weights every wrong label costs 1, so the hinge loss of a two-item sequence is 2
 * structuredHingeObjective(oracle, [{ x: [{ f: 1 }, { f: 1 }], y: ['A', 'A'] }], new Float64Array(6), { c2: 1 }).value // => 2
 * ```
 */
export function structuredHingeObjective<X, Y>(
  oracle: StructuredOracle<X, Y>,
  data: readonly StructuredExample<X, Y>[],
  w: ArrayLike<number>,
  penalty: StructuredPenalty = {},
): { readonly value: number; readonly gradient: Float64Array; readonly hinge: number } {
  const p = hingePass(oracle, data, w instanceof Float64Array ? w : Float64Array.from(w), penalty.c2 ?? 0)
  return { value: p.value, gradient: p.grad, hinge: p.hinge }
}

/** What every structured-SVM state reports, whichever solver: the reported weights and the hinge objective at them. */
export interface StructuredSvmReport {
  /** The reported weights (the average of the iterates when the mode averages). */
  readonly x: Float64Array
  /** The penalised structured hinge objective at `x`. */
  readonly value: number
  /** A subgradient of the objective at `x`. */
  readonly grad: Float64Array
  /** The distance moved by the last step. */
  readonly step: number
  /** The summed hinge losses at `x`. */
  readonly hinge: number
  /** Examples whose margin constraint is violated at `x`. */
  readonly violated: number
}

/** The state of the stochastic subgradient mode. */
export interface StructuredSubgradientState extends StructuredSvmReport {
  /** The iterate (equal to `x` unless averaging). */
  readonly w: Float64Array
  /** The running weighted average of the iterates. */
  readonly average: Float64Array
  /** Updates made so far (the step size is 1 / (lambda t)). */
  readonly t: number
  /** The radius of the ball the optimum lies in (the iterate is projected onto it). */
  readonly radius: number
}

/** The state of the block-coordinate Frank-Wolfe mode. */
export interface StructuredBcfwState extends StructuredSvmReport {
  /** The iterate, the sum of the blocks `blocks`. */
  readonly w: Float64Array
  /** The running weighted average of the iterates. */
  readonly average: Float64Array
  /** The blocks w_i of the iterate, one per example, flat. */
  readonly blocks: Float64Array
  /** The loss part l_i of each block. */
  readonly losses: Float64Array
  /** The sum of the l_i. */
  readonly loss: number
  /** Block updates made so far. */
  readonly updates: number
  /** The dual objective at the dual point (in the units of the primal objective). */
  readonly dual: number
  /** The duality gap of the iterate `w` and the dual point: an upper bound on its suboptimality. */
  readonly gap: number
}

/** The state of the two cutting-plane modes. */
export interface StructuredCuttingPlaneState extends StructuredSvmReport {
  /** The working set: the constraint vectors, flat (one row of the length of the weights per constraint). */
  readonly psi: Float64Array
  /** The loss of each working-set constraint. */
  readonly losses: Float64Array
  /** The slack variable (the example, or 0 for one slack) each constraint belongs to. */
  readonly owner: Int32Array
  /** The dual variable of each constraint. */
  readonly alpha: Float64Array
  /** The constraints found violated at `x` that the next step adds (empty at convergence), same layout as the working set. */
  readonly pendingPsi: Float64Array
  readonly pendingLosses: Float64Array
  readonly pendingOwner: Int32Array
  /** The dual objective of the working-set QP: a lower bound of the optimal objective. */
  readonly dual: number
  /** The primal objective at `x` minus the dual: at most `epsilon` once no constraint is violated by more. */
  readonly gap: number
  /** The size of the working set. */
  readonly constraints: number
  /** The step number. */
  readonly iterations: number
}

type AnySvmState = StructuredSubgradientState | StructuredBcfwState | StructuredCuttingPlaneState

interface Setup {
  readonly oracle: StructuredOracle<unknown, unknown>
  readonly data: readonly StructuredExample<unknown, unknown>[]
  readonly n: number
  readonly P: number
  readonly c2: number
}

function setup(problem: AnyStructuredProblem, who: string): Setup {
  const c2 = problem.penalty?.c2 ?? 0
  if (!(c2 > 0)) throw new OmaiError('value', `${who}: a structured SVM needs penalty.c2 > 0 (c2 = 1 / (2 C))`)
  if ((problem.penalty?.c1 ?? 0) > 0) throw new OmaiError('value', `${who}: the structured SVM has no L1 penalty`)
  return { oracle: problem.oracle, data: problem.data, n: problem.data.length, P: problem.oracle.size, c2 }
}

const knob = (knobs: Readonly<Record<string, SpaceValue>>, name: string, fallback: number): number =>
  (knobs[name] as number | undefined) ?? fallback

const flag = (knobs: Readonly<Record<string, SpaceValue>>, name: string, fallback: boolean): boolean =>
  (knobs[name] as boolean | undefined) ?? fallback

const orderOf = (n: number, shuffle: boolean, ctxKey: Parameters<typeof permutation>[0]): number[] =>
  shuffle ? Array.from(toFlat(permutation(ctxKey, n))) : Array.from({ length: n }, (_, i) => i)

/** The report of the weights `x`: one separation pass. */
function reportOf(
  s: Setup,
  x: Float64Array,
  previous: Float64Array | undefined,
): StructuredSvmReport & { items: readonly Separation[] } {
  const pass = hingePass(s.oracle, s.data, x, s.c2)
  const moved = previous === undefined ? 0 : dnrm2(x.map((v, q) => v - (previous[q] ?? 0)))
  return {
    x,
    value: pass.value,
    grad: pass.grad,
    step: moved,
    hinge: pass.hinge,
    violated: pass.violated,
    items: pass.items,
  }
}

function withoutItems<T extends StructuredSvmReport>(r: T & { readonly items?: unknown }): StructuredSvmReport {
  return { x: r.x, value: r.value, grad: r.grad, step: r.step, hinge: r.hinge, violated: r.violated }
}

const SHUFFLE = { shuffle: bool({ default: true, label: 'shuffle each epoch' }) }

/** The weighted average of the iterates with weights proportional to the step number (Lacoste-Julien et al.). */
function averageInto(average: Float64Array, w: Float64Array, before: number): void {
  const rho = 2 / (before + 2)
  for (let q = 0; q < average.length; q++) average[q] = (1 - rho) * (average[q] ?? 0) + rho * (w[q] ?? 0)
}

const subgradientMode: ModeSpec<AnyStructuredProblem, StructuredSubgradientState, VectorParams> = {
  description:
    'stochastic subgradient SSVM (Ratliff et al. 2007) with the Pegasos step 1 / (lambda t), optional projection and weighted averaging: one loss-augmented decode per example',
  space: dict({
    ...SHUFFLE,
    learningRate: real(1e-2, 1e2, { log: true, default: 1, label: 'step-size multiplier' }),
    average: bool({ default: true, label: 'report the weighted average of the iterates' }),
    project: bool({ default: true, label: 'project onto the ball the optimum lies in' }),
  }),
  make: (problem, knobs): Algorithm<StructuredSubgradientState> => {
    const s = setup(problem, 'subgradient')
    const { oracle, data, n, P, c2 } = s
    const rate = knob(knobs, 'learningRate', 1)
    const average = flag(knobs, 'average', true)
    const project = flag(knobs, 'project', true)
    const shuffle = flag(knobs, 'shuffle', true)
    // the optimum has c2 ||w||^2 <= objective(0), the summed largest losses
    const lambda = (2 * c2) / n
    const state = (w: Float64Array, avg: Float64Array, t: number, radius: number, previous?: Float64Array) => ({
      ...withoutItems(reportOf(s, average ? avg : w, previous)),
      w,
      average: avg,
      t,
      radius,
    })
    return {
      name: 'structured-svm-subgradient',
      config: { rule: 'subgradient', learningRate: rate, average, project, shuffle, examples: n, c2 },
      init: () => {
        const zero = new Float64Array(P)
        const radius = Math.sqrt(hingePass(oracle, data, zero, c2).hinge / c2)
        return state(zero, new Float64Array(P), 0, radius)
      },
      step: (st, ctx) => {
        const w = Float64Array.from(st.w)
        const avg = Float64Array.from(st.average)
        let t = st.t
        for (const i of orderOf(n, shuffle, ctx.key)) {
          const example = data[i]
          if (example === undefined) continue
          const sep = separate(oracle, example, w)
          const eta = rate / (lambda * (t + 1))
          const shrink = Math.max(0, 1 - eta * lambda)
          for (let q = 0; q < P; q++) w[q] = shrink * (w[q] ?? 0) + eta * (sep.psi[q] ?? 0)
          if (project) {
            const norm = dnrm2(w)
            if (norm > st.radius && norm > 0) for (let q = 0; q < P; q++) w[q] = ((w[q] ?? 0) * st.radius) / norm
          }
          averageInto(avg, w, t)
          t++
        }
        return state(w, avg, t, st.radius, st.x)
      },
      serialise: (st) => stateToJson(st) as Json,
      deserialise: (j) => stateFromJson(j) as StructuredSubgradientState,
    }
  },
  params: (st) => ({ x: Array.from(st.x) }),
}

const bcfwMode: ModeSpec<AnyStructuredProblem, StructuredBcfwState, VectorParams> = {
  description:
    'block-coordinate Frank-Wolfe SSVM (Lacoste-Julien et al. 2013): one loss-augmented decode and an exact line search per example, optional weighted averaging, the duality gap as the stopping certificate',
  space: dict({
    ...SHUFFLE,
    lineSearch: bool({ default: true, label: 'exact line search (else the step 2n / (k + 2n))' }),
    average: bool({ default: false, label: 'report the weighted average of the iterates' }),
    gapTol: real(1e-10, 1, {
      log: true,
      default: 1e-4,
      label: 'stop when the gap is below this times max(1, objective)',
    }),
  }),
  make: (problem, knobs): Algorithm<StructuredBcfwState> => {
    const s = setup(problem, 'bcfw')
    const { oracle, data, n, P, c2 } = s
    const lineSearch = flag(knobs, 'lineSearch', true)
    const average = flag(knobs, 'average', false)
    const shuffle = flag(knobs, 'shuffle', true)
    const gapTol = knob(knobs, 'gapTol', 1e-4)
    // the primal is lambda/2 ||w||^2 + mean hinge with lambda = 2 c2 / n; its dual is -lambda/2 ||w||^2 + sum_i l_i,
    // with w = sum_i w_i, w_i = (1 / (2 c2)) sum_y alpha_i(y) psi_i(y) and l_i = (1 / n) sum_y alpha_i(y) loss_i(y)
    const lambda = (2 * c2) / n
    const state = (
      w: Float64Array,
      avg: Float64Array,
      blocks: Float64Array,
      losses: Float64Array,
      loss: number,
      updates: number,
      previous?: Float64Array,
    ): StructuredBcfwState => {
      const at = hingePass(oracle, data, w, c2)
      const reported = average ? reportOf(s, avg, previous) : { ...at, x: w, step: 0, items: at.items }
      const moved = average
        ? reported.step
        : previous === undefined
          ? 0
          : dnrm2(w.map((v, q) => v - (previous[q] ?? 0)))
      const dual = n * loss - c2 * ddot(w, w)
      return {
        ...withoutItems(reported),
        step: moved,
        w,
        average: avg,
        blocks,
        losses,
        loss,
        updates,
        dual,
        gap: at.value - dual,
      }
    }
    return {
      name: 'structured-svm-bcfw',
      config: { rule: 'bcfw', lineSearch, average, shuffle, gapTol, examples: n, c2 },
      init: () => {
        const zero = new Float64Array(P)
        return state(zero, new Float64Array(P), new Float64Array(n * P), new Float64Array(n), 0, 0)
      },
      step: (st, ctx) => {
        const w = Float64Array.from(st.w)
        const avg = Float64Array.from(st.average)
        const blocks = Float64Array.from(st.blocks)
        const losses = Float64Array.from(st.losses)
        let loss = st.loss
        let k = st.updates
        for (const i of orderOf(n, shuffle, ctx.key)) {
          const example = data[i]
          if (example === undefined) continue
          const sep = separate(oracle, example, w)
          const base = i * P
          const ls = sep.loss / n
          // the step along w_s - w_i, with w_s = psi / (2 c2): gamma = ((w_i - w_s) . w - (l_i - l_s) / lambda) / |w_i - w_s|^2
          const d = new Float64Array(P)
          for (let q = 0; q < P; q++) d[q] = (blocks[base + q] ?? 0) - (sep.psi[q] ?? 0) / (2 * c2)
          let gamma = (2 * n) / (k + 2 * n)
          if (lineSearch) {
            const den = ddot(d, d)
            gamma = den > 0 ? Math.min(1, Math.max(0, (ddot(d, w) - ((losses[i] ?? 0) - ls) / lambda) / den)) : 0
          }
          for (let q = 0; q < P; q++) {
            const move = -gamma * (d[q] ?? 0)
            blocks[base + q] = (blocks[base + q] ?? 0) + move
            w[q] = (w[q] ?? 0) + move
          }
          const nextLoss = (1 - gamma) * (losses[i] ?? 0) + gamma * ls
          loss += nextLoss - (losses[i] ?? 0)
          losses[i] = nextLoss
          averageInto(avg, w, k)
          k++
        }
        return state(w, avg, blocks, losses, loss, k, st.x)
      },
      done: (st) => (st.gap <= gapTol * Math.max(1, Math.abs(st.value)) ? 'converged' : undefined),
      serialise: (st) => stateToJson(st) as Json,
      deserialise: (j) => stateFromJson(j) as StructuredBcfwState,
    }
  },
  params: (st) => ({ x: Array.from(st.x) }),
}

/** The constraint rows of the working set, appended (a duplicate of a row of the same owner is skipped). */
function appended(
  st: StructuredCuttingPlaneState,
  P: number,
): { psi: Float64Array; losses: Float64Array; owner: Int32Array; alpha: Float64Array } {
  const add = st.pendingLosses.length
  const m = st.losses.length
  const psi = new Float64Array((m + add) * P)
  psi.set(st.psi)
  psi.set(st.pendingPsi, m * P)
  const losses = new Float64Array(m + add)
  losses.set(st.losses)
  losses.set(st.pendingLosses, m)
  const owner = new Int32Array(m + add)
  owner.set(st.owner)
  owner.set(st.pendingOwner, m)
  const alpha = new Float64Array(m + add)
  alpha.set(st.alpha)
  return { psi, losses, owner, alpha }
}

/**
 * The dual of the working-set QP, `min 1/2 alpha' Q alpha - loss' alpha` with `Q_ab = psi_a . psi_b / (2 c2)`, one
 * simplex-like row `sum alpha <= 1` per slack variable and `alpha >= 0`, solved by the QP solvers of `optim`. The
 * weights are `w = sum alpha psi / (2 c2)` and the dual objective `sum alpha loss - c2 ||w||^2`.
 */
function solveWorkingSet(
  working: { psi: Float64Array; losses: Float64Array; owner: Int32Array; alpha: Float64Array },
  P: number,
  c2: number,
  method: string,
): { alpha: Float64Array; w: Float64Array; dual: number } {
  const m = working.losses.length
  if (m === 0) return { alpha: new Float64Array(0), w: new Float64Array(P), dual: 0 }
  const row = (a: number): Float64Array => working.psi.subarray(a * P, (a + 1) * P)
  const owners = [...new Set(working.owner)]
  // a feasible warm start: the previous duals, each owner's total scaled into its row of the constraint matrix
  const x0 = Float64Array.from(working.alpha, (v) => Math.max(0, v))
  for (const g of owners) {
    let total = 0
    for (let a = 0; a < m; a++) if (working.owner[a] === g) total += x0[a] ?? 0
    if (total > 1 - 1e-12)
      for (let a = 0; a < m; a++) if (working.owner[a] === g) x0[a] = ((x0[a] ?? 0) * (1 - 1e-9)) / total
  }
  const qp = qpArrays({
    P: Array.from({ length: m }, (_, a) => Array.from({ length: m }, (_, b) => ddot(row(a), row(b)) / (2 * c2))),
    q: Array.from(working.losses, (l) => -l),
    G: owners.map((g) => Array.from({ length: m }, (_, a) => (working.owner[a] === g ? 1 : 0))),
    h: owners.map(() => 1),
    lb: new Array<number>(m).fill(0),
  })
  // the active-set method can cycle on the degenerate rows of a large working set: fall back to the interior point
  let solution =
    method === 'active-set' ? qpActiveSet(qp, { x0, maxIterations: 10 * (m + owners.length + 1) }) : undefined
  if (solution === undefined || solution.status !== 'optimal') solution = qpInteriorPoint(qp)
  // dual feasible exactly: the interior-point method leaves each row of sum alpha <= 1 violated by its tolerance
  const alpha = Float64Array.from(solution.x, (v) => Math.max(0, v))
  for (const g of owners) {
    let total = 0
    for (let a = 0; a < m; a++) if (working.owner[a] === g) total += alpha[a] ?? 0
    if (total > 1) for (let a = 0; a < m; a++) if (working.owner[a] === g) alpha[a] = (alpha[a] ?? 0) / total
  }
  const w = new Float64Array(P)
  for (let a = 0; a < m; a++) {
    const weight = (alpha[a] ?? 0) / (2 * c2)
    if (weight !== 0) for (let q = 0; q < P; q++) w[q] = (w[q] ?? 0) + weight * (working.psi[a * P + q] ?? 0)
  }
  let dual = -c2 * ddot(w, w)
  for (let a = 0; a < m; a++) dual += (alpha[a] ?? 0) * (working.losses[a] ?? 0)
  return { alpha, w, dual }
}

function cuttingPlaneMode(
  kind: 'one-slack' | 'n-slack',
  description: string,
): ModeSpec<AnyStructuredProblem, StructuredCuttingPlaneState, VectorParams> {
  return {
    description,
    space: dict({
      epsilon: real(1e-10, 10, {
        log: true,
        default: 1e-4,
        label: 'stop when no constraint is violated by more (summed over the examples)',
      }),
      qp: choice(['interior-point', 'active-set'], {
        default: 'interior-point',
        label: 'solver of the dual QP (the active-set method falls back to the interior point if it cycles)',
      }),
    }),
    make: (problem, knobs): Algorithm<StructuredCuttingPlaneState> => {
      const s = setup(problem, kind)
      const { n, P, c2 } = s
      const epsilon = knob(knobs, 'epsilon', 1e-4)
      const method = (knobs['qp'] as string | undefined) ?? 'interior-point'
      const single = kind === 'one-slack'
      /** The constraints violated at the report's weights by more than the working set's own slack. */
      const pendingOf = (
        report: StructuredSvmReport & { readonly items: readonly Separation[] },
        working: { psi: Float64Array; losses: Float64Array; owner: Int32Array },
      ): { psi: Float64Array; losses: Float64Array; owner: Int32Array } => {
        const m = working.losses.length
        const slack = new Float64Array(single ? 1 : n)
        for (let a = 0; a < m; a++) {
          const g = single ? 0 : (working.owner[a] ?? 0)
          slack[g] = Math.max(
            slack[g] ?? 0,
            (working.losses[a] ?? 0) - ddot(report.x, working.psi.subarray(a * P, (a + 1) * P)),
          )
        }
        const candidates: { psi: Float64Array; loss: number; owner: number }[] = []
        if (single) {
          const psi = new Float64Array(P)
          let loss = 0
          for (const item of report.items) {
            loss += item.loss
            for (let q = 0; q < P; q++) psi[q] = (psi[q] ?? 0) + (item.psi[q] ?? 0)
          }
          if (report.hinge - (slack[0] ?? 0) > epsilon) candidates.push({ psi, loss, owner: 0 })
        } else {
          report.items.forEach((item, i) => {
            if (item.hinge - (slack[i] ?? 0) > epsilon / n)
              candidates.push({ psi: item.psi, loss: item.loss, owner: i })
          })
        }
        const fresh = candidates.filter((c) => {
          for (let a = 0; a < m; a++)
            if (
              working.owner[a] === c.owner &&
              working.losses[a] === c.loss &&
              working.psi.subarray(a * P, (a + 1) * P).every((v, q) => v === c.psi[q])
            )
              return false
          return true
        })
        const psi = new Float64Array(fresh.length * P)
        fresh.forEach((c, r) => {
          psi.set(c.psi, r * P)
        })
        return {
          psi,
          losses: Float64Array.from(fresh, (c) => c.loss),
          owner: Int32Array.from(fresh, (c) => c.owner),
        }
      }
      const state = (
        working: { psi: Float64Array; losses: Float64Array; owner: Int32Array; alpha: Float64Array },
        w: Float64Array,
        dual: number,
        iterations: number,
        previous?: Float64Array,
      ): StructuredCuttingPlaneState => {
        const report = reportOf(s, w, previous)
        const pending = pendingOf(report, working)
        return {
          ...withoutItems(report),
          psi: working.psi,
          losses: working.losses,
          owner: working.owner,
          alpha: working.alpha,
          pendingPsi: pending.psi,
          pendingLosses: pending.losses,
          pendingOwner: pending.owner,
          dual,
          gap: report.value - dual,
          constraints: working.losses.length,
          iterations,
        }
      }
      return {
        name: `structured-svm-${kind}`,
        config: { rule: kind, epsilon, qp: method, examples: n, c2 },
        init: () =>
          state(
            {
              psi: new Float64Array(0),
              losses: new Float64Array(0),
              owner: new Int32Array(0),
              alpha: new Float64Array(0),
            },
            new Float64Array(P),
            0,
            0,
          ),
        step: (st) => {
          const working = appended(st, P)
          const solved = solveWorkingSet(working, P, c2, method)
          return state({ ...working, alpha: solved.alpha }, solved.w, solved.dual, st.iterations + 1, st.x)
        },
        done: (st) => (st.pendingLosses.length === 0 ? 'converged' : undefined),
        serialise: (st) => stateToJson(st) as Json,
        deserialise: (j) => stateFromJson(j) as StructuredCuttingPlaneState,
      }
    },
    params: (st) => ({ x: Array.from(st.x) }),
  }
}

/**
 * The structured-SVM modes by name, for the extra modes of a structured solver table
 * (`structuredSolversWith(structuredSvmModes)`): `subgradient`, `bcfw`, `one-slack` and `n-slack`. Each needs only the
 * oracle's loss-augmented `decode`, `features` and `loss`, a penalty `c2 > 0` (`c2 = 1 / (2 C)`, see
 * {@link structuredSvmPenalty}) and ignores `start` (they start from zero weights). They report the penalised structured
 * hinge objective as the state's `value`, not the negative log-likelihood of the other modes.
 */
export const structuredSvmModes: Readonly<Record<string, StructuredMode>> = {
  subgradient: subgradientMode,
  bcfw: bcfwMode,
  'one-slack': cuttingPlaneMode(
    'one-slack',
    'one-slack cutting-plane SSVM (Joachims, Finley and Yu 2009): one aggregated most-violated constraint per round, the dual QP over the working set solved exactly',
  ),
  'n-slack': cuttingPlaneMode(
    'n-slack',
    'n-slack cutting-plane SSVM (Tsochantaridis et al. 2005): a working set of most-violated constraints per example, the dual QP over it solved exactly',
  ),
}

/**
 * The table of the four structured-SVM solvers (`bcfw` the default, `subgradient`, `one-slack`, `n-slack`) over a
 * `StructuredProblem`. It is a table of its own, not part of `structuredSolvers`: the generic modes of that one minimise
 * the smooth negative log-likelihood and its tape reports it, while these minimise the non-smooth structured hinge
 * loss and report that, with the duality gap and working-set size where they have them. For one table with both, use
 * `structuredSolversWith(structuredSvmModes)`.
 *
 * @index table Structured-SVM solvers: subgradient, block-coordinate Frank-Wolfe, one-slack and n-slack cutting planes.
 * @concept structured-svm
 * @example
 * ```ts
 * import { structuredSvmSolvers } from 'omai-engine/pgm'
 * [structuredSvmSolvers.modeOf({}).name, structuredSvmSolvers.names] // => ['bcfw', ['subgradient', 'bcfw', 'one-slack', 'n-slack']]
 * ```
 */
export const structuredSvmSolvers: ModeTable<AnyStructuredProblem, VectorParams> = modes<
  AnyStructuredProblem,
  VectorParams
>(structuredSvmModes, {
  label: 'solver',
  description: 'how the structured hinge loss is minimised',
  default: 'bcfw',
})

/** Options of {@link structuredSvmAlgorithm} and {@link trainStructuredSvm}. */
export interface StructuredSvmOptions {
  /** The solver, a value of `structuredSvmSolvers.space` (default block-coordinate Frank-Wolfe). */
  readonly solver?: SpaceValue
  /** Stop when the largest entry of the subgradient falls below this (default 1e-9; each solver has its own criterion). */
  readonly tol?: number
  /** The key for shuffling. */
  readonly key?: ReturnType<typeof makeKey>
}

/**
 * What the traced training of a structured SVM reports: the view of `structuredTrainView` with the objective the
 * penalised structured hinge loss (and `gradient` the largest entry of a subgradient), the summed hinge, the examples
 * with a violated margin, and, for the solvers that have them, the dual objective, duality gap and working-set size.
 */
export interface StructuredSvmTrainState extends StructuredTrainView, SlotState {
  readonly hinge: number
  readonly violated: number
  readonly dual?: number
  readonly gap?: number
  readonly constraints?: number
}

/**
 * Structured-SVM training as a traced {@link Algorithm} over the slot `structuredSvmSolvers`: each step (an epoch of
 * `subgradient` or `bcfw`, a round of a cutting-plane method) puts the structured hinge objective, a subgradient norm,
 * the task loss and error of decoding the training examples, the summed hinge, the number of violated examples and
 * where available the dual objective, the gap and the number of constraints on the tape.
 *
 * @param problem - The oracle, labelled examples and penalty (`c2 > 0`; see {@link structuredSvmPenalty}).
 * @param options - The solver and tolerance.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for no examples, or (when the solver is built) for `c2` not positive or an L1 penalty.
 * @concept structured-svm
 * @example
 * ```ts
 * import { chainOracle, linearChainCrf, structuredSvmAlgorithm } from 'omai-engine/pgm'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const oracle = chainOracle(linearChainCrf(['X', 'Y'], ['w=a', 'w=b']))
 * const data = [{ x: [{ 'w=a': 1 }, { 'w=b': 1 }], y: ['X', 'Y'] }]
 * const solver = { kind: 'one-slack' }
 * runAlgorithm(structuredSvmAlgorithm({ oracle, data, penalty: { c2: 0.1 } }, { solver }), { maxSteps: 50 }).state.error // => 0
 * ```
 */
export function structuredSvmAlgorithm<X, Y>(
  problem: StructuredProblem<X, Y>,
  options: StructuredSvmOptions = {},
): Algorithm<StructuredSvmTrainState> {
  if (problem.data.length === 0) throw new OmaiError('value', 'structuredSvm: no training examples')
  const solver = options.solver ?? { kind: 'bcfw' }
  const P = problem.oracle.size
  const erased = problem as AnyStructuredProblem
  const { mistakes: _mistakes, ...metrics } = structuredTrainMetrics({
    name: 'error',
    description: 'the fraction of training examples not decoded exactly right',
  })
  return slotAlgorithm<StructuredSvmTrainState, AnyStructuredProblem>({
    name: 'structured-svm-training',
    config: { solver, examples: problem.data.length, c2: problem.penalty?.c2 ?? 0 } as Json,
    schema: {
      metrics: {
        ...metrics,
        objective: { description: 'the penalised structured hinge loss (margin rescaling)', direction: 'lower' },
        gradient: { description: 'the largest entry of a subgradient of the objective', direction: 'lower' },
        hinge: { description: 'the summed hinge losses at the weights', direction: 'lower' },
        violated: { description: 'the examples whose margin constraint is violated', direction: 'lower' },
        dual: {
          description: 'the dual objective (a lower bound of the optimum; bcfw and the cutting planes)',
          direction: 'higher',
        },
        gap: { description: 'the primal objective minus the dual objective', direction: 'lower' },
        constraints: { description: 'the size of the working set of a cutting-plane method' },
      },
      tensors: { weights: { role: 'param', description: 'the model weights, in the oracle layout', shape: [P] } },
    },
    problem: erased,
    table: structuredSvmSolvers,
    solver,
    tol: options.tol ?? 1e-9,
    key: options.key ?? makeKey(0),
    view: (r: SolverReached) => {
      const s = r as unknown as AnySvmState & Partial<StructuredBcfwState & StructuredCuttingPlaneState>
      return {
        ...structuredTrainView(erased, r),
        hinge: s.hinge,
        violated: s.violated,
        ...(s.dual === undefined ? {} : { dual: s.dual }),
        ...(s.gap === undefined ? {} : { gap: s.gap }),
        ...(s.constraints === undefined ? {} : { constraints: s.constraints }),
      }
    },
    observe: (s) => ({
      metrics: {
        objective: s.objective,
        gradient: s.gradient,
        loss: s.loss,
        error: s.error,
        nonzeros: s.nonzeros,
        hinge: s.hinge,
        violated: s.violated,
        ...(s.dual === undefined ? {} : { dual: s.dual }),
        ...(s.gap === undefined ? {} : { gap: s.gap }),
        ...(s.constraints === undefined ? {} : { constraints: s.constraints }),
      },
      tensors: { weights: { role: 'param', value: s.weights, shape: [P] } },
    }),
  })
}

/**
 * Train a structured SVM to the end with one of `structuredSvmSolvers`: the weights, the penalised structured hinge
 * objective, the task loss and error on the training examples, and the duality gap and constraint count where the
 * solver has them.
 *
 * @param problem - The oracle, labelled examples and penalty (`c2 > 0`; see {@link structuredSvmPenalty}).
 * @param options - The solver (default `bcfw`), tolerance and key.
 * @param maxSteps - The step budget (epochs for `subgradient` and `bcfw`, rounds for the cutting planes; default 200).
 * @returns The fitted weights and the training summary.
 * @concept structured-svm
 * @index function Train a structured SVM by subgradient, Frank-Wolfe or cutting planes, with duality gap.
 * @example
 * ```ts
 * import { chainOracle, linearChainCrf, trainStructuredSvm } from 'omai-engine/pgm'
 * const oracle = chainOracle(linearChainCrf(['X', 'Y'], ['w=a', 'w=b']))
 * const data = [{ x: [{ 'w=a': 1 }, { 'w=b': 1 }], y: ['X', 'Y'] }]
 * const fit = trainStructuredSvm({ oracle, data, penalty: { c2: 0.1 } }, { solver: { kind: 'n-slack' } })
 * [fit.error, fit.converged, fit.gap !== undefined && fit.gap < 1e-3] // => [0, true, true]
 * ```
 */
export function trainStructuredSvm<X, Y>(
  problem: StructuredProblem<X, Y>,
  options: StructuredSvmOptions = {},
  maxSteps = 200,
): {
  readonly weights: Float64Array
  readonly objective: number
  readonly hinge: number
  readonly loss: number
  readonly error: number
  readonly gap?: number
  readonly dual?: number
  readonly constraints?: number
  readonly converged: boolean
  readonly steps: number
} {
  const run = runAlgorithm(structuredSvmAlgorithm(problem, options), {
    maxSteps,
    key: options.key ?? foldIn(makeKey(0), 1),
  })
  const s = run.state
  return {
    weights: Float64Array.from(s.weights),
    objective: s.objective,
    hinge: s.hinge,
    loss: s.loss,
    error: s.error,
    ...(s.gap === undefined ? {} : { gap: s.gap }),
    ...(s.dual === undefined ? {} : { dual: s.dual }),
    ...(s.constraints === undefined ? {} : { constraints: s.constraints }),
    converged: run.status === 'converged',
    steps: run.steps,
  }
}
