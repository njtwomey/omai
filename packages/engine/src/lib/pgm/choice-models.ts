/**
 * Rankings from strengths: the Plackett-Luce model (a ranking is built by choosing the best of those left with
 * probability proportional to strength) and its two-item case, the Bradley-Terry model of paired comparisons. The
 * log-likelihood in the log strengths is smooth and concave, so the fit takes its solver from the slot: Hunter's
 * minorise-maximise iteration (the model's own mode, a monotone fixed-point update) beside Newton, BFGS, L-BFGS and the
 * rest of `smoothSolvers`, which reach the same strengths. Strengths are identified up to a common factor, fixed here
 * by a zero sum of the log strengths. Every item needs at least one win and one loss for the maximum to exist.
 */

import { MdmlError } from '../base/index.ts'
import {
  analyticObjective,
  familySolvers,
  type SlotState,
  type SmoothProblem,
  slotAlgorithm,
  type VectorParams,
} from '../optim/index.ts'
import { key as makeKey } from '../random/index.ts'
import { dict, type SpaceValue } from '../space/index.ts'
import { type Algorithm, type ModeSpec, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'

/** Rankings of items, each listing some items from best to worst. */
export interface ChoiceProblem {
  readonly nItems: number
  readonly rankings: readonly (readonly number[])[]
}

/**
 * Paired comparisons as rankings: each game `[winner, loser]` is a ranking of two (Bradley-Terry as Plackett-Luce).
 *
 * @param nItems - The number of items.
 * @param games - Winner and loser of each comparison.
 * @returns The problem.
 * @example
 * ```ts
 * import { pairedComparisons } from 'mdml-engine/pgm'
 * pairedComparisons(3, [[0, 1], [1, 2]]).rankings // => [[0, 1], [1, 2]]
 * ```
 */
export function pairedComparisons(nItems: number, games: readonly (readonly [number, number])[]): ChoiceProblem {
  return { nItems, rankings: games.map(([w, l]) => [w, l]) }
}

function check(p: ChoiceProblem): void {
  const wins = new Array<number>(p.nItems).fill(0)
  const losses = new Array<number>(p.nItems).fill(0)
  for (const r of p.rankings) {
    if (r.length < 2) throw new MdmlError('value', 'choice model: a ranking needs at least two items')
    r.forEach((item, pos) => {
      if (!(Number.isInteger(item) && item >= 0 && item < p.nItems))
        throw new MdmlError('value', 'choice model: item out of range')
      if (pos < r.length - 1) wins[item] = (wins[item] ?? 0) + 1
      if (pos > 0) losses[item] = (losses[item] ?? 0) + 1
    })
  }
  for (let i = 0; i < p.nItems; i++)
    if ((wins[i] ?? 0) === 0 || (losses[i] ?? 0) === 0)
      throw new MdmlError('value', `choice model: item ${i} has no win or no loss, so its strength is not finite`)
}

/** The log strengths from the free vector (the last is minus the sum of the others). */
const thetaOf = (u: ArrayLike<number>): number[] => {
  const t = Array.from(u)
  t.push(-t.reduce((a, b) => a + b, 0))
  return t
}

/** The negative log-likelihood per ranking and its gradient in the free vector. */
function evaluate(p: ChoiceProblem, u: ArrayLike<number>): { value: number; grad: Float64Array } {
  const theta = thetaOf(u)
  const n = p.nItems
  const g = new Float64Array(n)
  let ll = 0
  for (const r of p.rankings) {
    for (let s = 0; s < r.length - 1; s++) {
      const rest = r.slice(s)
      const mx = Math.max(...rest.map((i) => theta[i] ?? 0))
      const z = rest.reduce((a, i) => a + Math.exp((theta[i] ?? 0) - mx), 0)
      const chosen = r[s] as number
      ll += (theta[chosen] ?? 0) - mx - Math.log(z)
      g[chosen] = (g[chosen] ?? 0) + 1
      for (const i of rest) g[i] = (g[i] ?? 0) - Math.exp((theta[i] ?? 0) - mx) / z
    }
  }
  const m = p.rankings.length
  const grad = new Float64Array(n - 1)
  for (let i = 0; i < n - 1; i++) grad[i] = -((g[i] ?? 0) - (g[n - 1] ?? 0)) / m
  return { value: -ll / m, grad }
}

/**
 * The negative log-likelihood per ranking as a smooth problem in the free log strengths, with its analytic gradient;
 * the generic solvers difference the gradient for the curvature.
 *
 * @param problem - The rankings.
 * @returns The objective and the start (all strengths equal).
 * @example
 * ```ts
 * import { choiceSmoothProblem, pairedComparisons } from 'mdml-engine/pgm'
 * choiceSmoothProblem(pairedComparisons(3, [[0, 1], [1, 2], [2, 0]])).x0 // => [0, 0]
 * ```
 */
export function choiceSmoothProblem(problem: ChoiceProblem): SmoothProblem {
  check(problem)
  return {
    objective: analyticObjective({
      value: (x) => evaluate(problem, x).value,
      gradient: (x) => evaluate(problem, x).grad,
    }),
    x0: new Array<number>(problem.nItems - 1).fill(0),
  }
}

interface MmState {
  readonly x: Float64Array
  readonly value: number
  readonly grad: Float64Array
  readonly step: number
}

const mm: ModeSpec<ChoiceProblem, MmState, VectorParams> = {
  description:
    'Hunter minorise-maximise: each strength set to its number of wins over the sum, across the stages it is in, of the reciprocal of the strengths still in contention; the likelihood never decreases',
  space: dict({}),
  make: (p): Algorithm<MmState> => {
    check(p)
    const n = p.nItems
    const wins = new Array<number>(n).fill(0)
    for (const r of p.rankings)
      for (let s = 0; s < r.length - 1; s++) wins[r[s] as number] = (wins[r[s] as number] ?? 0) + 1
    const at = (x: Float64Array, step: number): MmState => ({ x, ...evaluate(p, x), step })
    return {
      name: 'choice-mm',
      config: { nItems: n } as never,
      init: () => at(new Float64Array(n - 1), 0),
      step: (s) => {
        const pi = thetaOf(s.x).map(Math.exp)
        const denom = new Array<number>(n).fill(0)
        for (const r of p.rankings)
          for (let st = 0; st < r.length - 1; st++) {
            const rest = r.slice(st)
            const z = rest.reduce((a, i) => a + (pi[i] ?? 0), 0)
            for (const i of rest) denom[i] = (denom[i] ?? 0) + 1 / z
          }
        const next = pi.map((_, i) => (wins[i] ?? 0) / (denom[i] ?? 1))
        const logs = next.map(Math.log)
        const centre = logs.reduce((a, b) => a + b, 0) / n
        const x = Float64Array.from(logs.slice(0, n - 1), (v) => v - centre)
        return at(x, Math.hypot(...Array.from(x, (v, i) => v - (s.x[i] ?? 0))))
      },
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as MmState,
    }
  },
  params: (s) => ({ x: Array.from(s.x) }),
}

/** The ways to fit a choice model: `mm` (Hunter's iteration), then every generic smooth solver on the same objective. */
export const choiceSolvers: ReturnType<typeof familySolvers<ChoiceProblem>> = familySolvers(
  { mm },
  choiceSmoothProblem,
  {
    label: 'solver',
    description: 'how the Plackett-Luce log-likelihood is maximised',
    default: 'mm',
  },
)

/** The state of {@link choiceAlgorithm}. */
export type ChoiceState = SlotState & {
  /** The free log strengths. */
  readonly u: number[]
  readonly objective: number
  readonly gradient: number
}

/** Options of {@link choiceAlgorithm} and {@link fitPlackettLuce}. */
export interface ChoiceOptions {
  /** A value of `choiceSolvers.space` (default `mm`). */
  readonly solver?: SpaceValue
  /** The gradient tolerance (default 1e-9). */
  readonly tol?: number
  /** The iteration cap of `fitPlackettLuce` (default 2000). */
  readonly maxSteps?: number
}

/**
 * The Plackett-Luce fit as a traced algorithm over the solver slot, logging `objective` (the negative log-likelihood
 * per ranking), `gradient`, `gradNorm`, `step` and the tensor `logStrength`.
 *
 * @param problem - The rankings.
 * @param options - The solver and tolerance.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { choiceAlgorithm, pairedComparisons } from 'mdml-engine/pgm'
 * choiceAlgorithm(pairedComparisons(2, [[0, 1], [1, 0], [0, 1]])).name // => 'plackett-luce'
 * ```
 */
export function choiceAlgorithm(problem: ChoiceProblem, options: ChoiceOptions = {}): Algorithm<ChoiceState> {
  const solver = options.solver ?? { kind: 'mm' }
  const tol = options.tol ?? 1e-9
  return slotAlgorithm<ChoiceState, ChoiceProblem>({
    name: 'plackett-luce',
    config: { nItems: problem.nItems, nRankings: problem.rankings.length, solver } as never,
    schema: {
      metrics: {
        objective: { description: 'the negative log-likelihood per ranking', direction: 'lower' },
        gradient: { description: 'the largest gradient entry', direction: 'lower' },
      },
      tensors: { logStrength: { role: 'param', description: 'the free log strengths', shape: [problem.nItems - 1] } },
    },
    problem,
    table: choiceSolvers,
    solver,
    tol,
    view: (r) => ({
      u: Array.from(r.x),
      objective: r.value,
      gradient: r.grad.reduce((a, b) => Math.max(a, Math.abs(b)), 0),
    }),
    observe: (s) => ({
      metrics: { objective: s.objective, gradient: s.gradient },
      tensors: { logStrength: { role: 'param', value: s.u, shape: [problem.nItems - 1] } },
    }),
  })
}

/** A fitted Plackett-Luce or Bradley-Terry model. */
export interface ChoiceFit {
  /** The log strengths, summing to zero. */
  readonly theta: readonly number[]
  /** The strengths exp(theta), with geometric mean one. */
  readonly strength: readonly number[]
  readonly logLikelihood: number
  readonly converged: boolean
  readonly steps: number
  readonly solver: string
}

/**
 * Fit a Plackett-Luce model by maximum likelihood.
 *
 * @param problem - The rankings.
 * @param options - The solver (default Hunter's `mm`), tolerance and cap.
 * @returns The strengths and the likelihood.
 * @example
 * ```ts
 * import { fitPlackettLuce, pairedComparisons } from 'mdml-engine/pgm'
 * // A beats B three times in four: the odds are three to one.
 * const fit = fitPlackettLuce(pairedComparisons(2, [[0, 1], [0, 1], [0, 1], [1, 0]]))
 * Math.round((fit.strength[0] ?? 0) / (fit.strength[1] ?? 1) * 1e6) / 1e6 // => 3
 * ```
 */
export function fitPlackettLuce(problem: ChoiceProblem, options: ChoiceOptions = {}): ChoiceFit {
  const algorithm = choiceAlgorithm(problem, options)
  const run = runAlgorithm(algorithm, { maxSteps: options.maxSteps ?? 2000, key: makeKey(0) })
  const theta = thetaOf(run.state.u)
  return {
    theta,
    strength: theta.map(Math.exp),
    logLikelihood: -run.state.objective * problem.rankings.length,
    converged: run.status === 'converged',
    steps: run.steps,
    solver: choiceSolvers.modeOf(options.solver ?? { kind: 'mm' }).name,
  }
}

/**
 * Fit the Bradley-Terry model of paired comparisons.
 *
 * @param nItems - The number of items.
 * @param games - Winner and loser of each comparison.
 * @param options - The solver and tolerance.
 * @returns The strengths and the likelihood.
 * @example
 * ```ts
 * import { fitBradleyTerry } from 'mdml-engine/pgm'
 * fitBradleyTerry(2, [[0, 1], [1, 0]]).theta // => [0, 0]
 * ```
 */
export function fitBradleyTerry(
  nItems: number,
  games: readonly (readonly [number, number])[],
  options: ChoiceOptions = {},
): ChoiceFit {
  return fitPlackettLuce(pairedComparisons(nItems, games), options)
}

/**
 * The probability that item i beats item j under fitted strengths: pi_i / (pi_i + pi_j).
 *
 * @param fit - The fitted model (or its log strengths).
 * @param i - One item.
 * @param j - The other.
 * @returns The win probability.
 * @example
 * ```ts
 * import { winProbability } from 'mdml-engine/pgm'
 * winProbability({ theta: [Math.log(3), 0] }, 0, 1) // => 0.75
 * ```
 */
export function winProbability(fit: Pick<ChoiceFit, 'theta'>, i: number, j: number): number {
  const a = fit.theta[i] ?? 0
  const b = fit.theta[j] ?? 0
  return 1 / (1 + Math.exp(b - a))
}
