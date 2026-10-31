/**
 * Backfitting (Hastie and Tibshirani) for additive models, as an `Algorithm` with one cycle over the terms per step:
 * each term is re-smoothed against the partial residuals of the others, by the penalised weighted least squares of its own
 * block at its own smoothing parameters. For an exponential family it is local scoring, the working response being
 * refreshed at the start of every cycle. At fixed smoothing parameters the cycles converge to the joint penalised
 * solution of {@link gamAlgorithm}, which is what makes the two checks of each other.
 */

import { MdmlError } from '../base/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'
import { assemblePenalty, gram, solvePenalised } from './penalised.ts'
import type { GamHyper, GamProblem } from './pirls.ts'
import { startingPredictor, workingResponse } from './pirls.ts'

/** The state of {@link backfittingAlgorithm}. */
export type BackfitState = {
  /** Cycles done. */
  readonly cycle: number
  /** The coefficients of every block: the intercept, then the terms. */
  readonly beta: number[]
  /** The linear predictor. */
  readonly eta: number[]
  /** The largest change of the linear predictor in the last cycle. */
  readonly change: number
  readonly converged: boolean
}

/**
 * Backfitting over the terms of a problem at the smoothing parameters `hyper.sp` (default all 1), one cycle per step. It
 * logs `change`, the largest change of the linear predictor in the cycle, and the coefficients; it ends when
 * `change < tol * (1 + max |eta|)`.
 *
 * @param problem - The prepared problem.
 * @param hyper - The family and `sp`; `tol` is the tolerance on the change.
 * @returns The algorithm.
 * @throws MdmlError `shape` for an `sp` of the wrong length.
 * @concept generalized-additive-models
 * @reference T. Hastie and R. Tibshirani, Generalized Additive Models (1990), chapter 5.
 * @example
 * ```ts
 * import { GAM_DEFAULTS, backfittingAlgorithm, gamProblem, s } from 'mdml-engine/gam'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = Array.from({ length: 40 }, (_, i) => [i / 39, ((i * 7) % 40) / 39])
 * const y = x.map((r) => Math.sin(5 * (r[0] ?? 0)) + (r[1] ?? 0))
 * const hyper = { ...GAM_DEFAULTS, terms: [s(0, { k: 6 }), s(1, { k: 6 })], tol: 1e-9 }
 * const problem = gamProblem(x, y, hyper)
 * runAlgorithm(backfittingAlgorithm(problem, hyper), { maxSteps: 500 }).state.converged // => true
 * ```
 */
export function backfittingAlgorithm(problem: GamProblem, hyper: GamHyper): Algorithm<BackfitState> {
  const { design, structure } = problem
  const { n, p } = design
  const nPen = structure.blocks.length
  if (hyper.sp !== undefined && hyper.sp.length !== nPen) {
    throw new MdmlError('shape', `backfitting: sp has ${hyper.sp.length} entries for ${nPen} penalties`)
  }
  const logLambda = Array.from({ length: nPen }, (_, r) => Math.log(Math.max(hyper.sp?.[r] ?? 1, 1e-300)))
  const S = assemblePenalty(structure, logLambda)
  const blocks = [{ offset: 0, width: 1 }, ...design.terms.map((t) => ({ offset: t.offset, width: t.width }))]
  const columns = (offset: number, width: number): Float64Array => {
    const out = new Float64Array(n * width)
    for (let i = 0; i < n; i++) for (let j = 0; j < width; j++) out[i * width + j] = design.X[i * p + offset + j] ?? 0
    return out
  }
  const blockX = blocks.map((b) => columns(b.offset, b.width))
  const blockS = blocks.map((b) => {
    const out = new Float64Array(b.width * b.width)
    for (let i = 0; i < b.width; i++)
      for (let j = 0; j < b.width; j++) out[i * b.width + j] = S[(b.offset + i) * p + b.offset + j] ?? 0
    return out
  })
  return {
    name: 'gam-backfitting',
    config: { family: problem.family.name, link: problem.link.name },
    schema: channels({
      metrics: {
        change: { description: 'the largest change of the linear predictor in the cycle', direction: 'lower' },
      },
      tensors: { coef: { role: 'param', description: 'the coefficients', shape: [p] } },
    }),
    init: () => ({
      cycle: 0,
      beta: new Array<number>(p).fill(0),
      eta: new Array<number>(n).fill(0),
      change: Number.POSITIVE_INFINITY,
      converged: false,
    }),
    step: (s) => {
      // The first working response is formed at the starting predictor, the later ones at the current fit.
      const wz = workingResponse(problem, s.cycle === 0 ? startingPredictor(problem) : s.eta)
      const eta = Float64Array.from(s.eta)
      const beta = Float64Array.from(s.beta)
      let change = 0
      const previous = Float64Array.from(eta)
      blocks.forEach((b, k) => {
        const Xb = blockX[k] as Float64Array
        const fOld = new Float64Array(n)
        for (let i = 0; i < n; i++) {
          let v = 0
          for (let j = 0; j < b.width; j++) v += (Xb[i * b.width + j] ?? 0) * (beta[b.offset + j] ?? 0)
          fOld[i] = v
        }
        const r = new Float64Array(n)
        for (let i = 0; i < n; i++) r[i] = (wz.z[i] ?? 0) - ((eta[i] ?? 0) - (fOld[i] ?? 0))
        const sol = solvePenalised(gram(Xb, n, b.width, wz.w, r), blockS[k] as Float64Array)
        for (let j = 0; j < b.width; j++) beta[b.offset + j] = sol.beta[j] ?? 0
        for (let i = 0; i < n; i++) eta[i] = (eta[i] ?? 0) - (fOld[i] ?? 0) + (sol.eta[i] ?? 0)
      })
      let scale = 1
      for (let i = 0; i < n; i++) {
        change = Math.max(change, Math.abs((eta[i] ?? 0) - (previous[i] ?? 0)))
        scale = Math.max(scale, Math.abs(eta[i] ?? 0))
      }
      return {
        cycle: s.cycle + 1,
        beta: Array.from(beta),
        eta: Array.from(eta),
        change,
        converged: s.cycle > 0 && change < hyper.tol * scale,
      }
    },
    done: (s) => (s.converged ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { change: s.change },
      tensors: { coef: { role: 'param', value: s.beta, shape: [p] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as BackfitState,
  }
}
