// Variable-order BDF/NDF: scipy's BDF ported, with an implicit-function correction for traced steps.
import { WhiteboxError } from '../../base/index.ts'
import type { LuFactorisation } from '../../numerics/linalg/index.ts'
import { isTensor, type Value } from '../../tensor/index.ts'
import {
  applyMass,
  checkMass,
  type ImplicitConfig,
  identityNumbers,
  implicitCorrection,
  isTracedState,
  jacobianAt,
  luApply,
  massTimes,
  realLu,
} from './_newton.ts'
import { atBound, attempt, type Counter, snapshot, statusAfter, tally } from './_step.ts'
import { allFinite, combine, fromNumbers, minStep, numbersOf, scaledRms, type Term, tolAt } from './_vec.ts'
import type { BdfState } from './implicit.ts'
import type { DenseSegment, StepAttempt, VectorField } from './problem.ts'

/** The highest BDF order. */
export const BDF_MAX_ORDER = 5
const NEWTON_MAXITER = 4
const MIN_FACTOR = 0.2
const MAX_FACTOR = 10
const KAPPA = [0, -0.185, -1 / 9, -0.0823, -0.0415, 0]
const GAMMA = KAPPA.map((_, k) => {
  let s = 0
  for (let j = 1; j <= k; j++) s += 1 / j
  return s
})
const ALPHA = KAPPA.map((kappa, i) => (1 - kappa) * (GAMMA[i] ?? 0))
const ERROR_CONST = KAPPA.map((kappa, i) => kappa * (GAMMA[i] ?? 0) + 1 / (i + 1))

function computeR(order: number, factor: number): number[][] {
  const m = Array.from({ length: order + 1 }, (_, i) =>
    Array.from({ length: order + 1 }, (_, j) => (i === 0 ? 1 : j === 0 ? 0 : (i - 1 - factor * j) / i)),
  )
  for (let i = 1; i <= order; i++) {
    const row = m[i] as number[]
    for (let j = 0; j <= order; j++) row[j] = (row[j] ?? 0) * (m[i - 1]?.[j] ?? 0)
  }
  return m
}

/** Rescale the difference array for a step-size change by `factor` (scipy's change_D). */
export function changeDifferences(d: readonly Value[], order: number, factor: number): Value[] {
  const r = computeR(order, factor)
  const u = computeR(order, 1)
  const ru = r.map((row) =>
    Array.from({ length: order + 1 }, (_, j) => row.reduce((acc, rik, k) => acc + rik * (u[k]?.[j] ?? 0), 0)),
  )
  const out = [...d]
  for (let i = 0; i <= order; i++) {
    out[i] = combine(
      null,
      Array.from({ length: order + 1 }, (_, k) => [ru[k]?.[i] ?? 0, d[k] as Value] as const),
    )
  }
  return out
}

function solveSystem(
  f: VectorField,
  tNew: number,
  yPred: Float64Array,
  shape: readonly number[],
  c: number,
  psiM: Float64Array,
  lu: LuFactorisation,
  scale: Float64Array,
  tol: number,
  mass: Float64Array | undefined,
): { converged: boolean; iterations: number; y: Float64Array; d: Float64Array } {
  const n = yPred.length
  const d = new Float64Array(n)
  const y = Float64Array.from(yPred)
  let normOld: number | undefined
  let converged = false
  let k = 0
  for (; k < NEWTON_MAXITER; k++) {
    const fv = numbersOf(f(tNew, fromNumbers(y, shape)))
    if (!allFinite(fv)) break
    const md = applyMass(mass, d, n)
    const dy = luApply(
      lu,
      Float64Array.from(fv, (v, i) => c * v - (psiM[i] ?? 0) - (md[i] ?? 0)),
    )
    const dyNorm = scaledRms(dy, scale)
    const rate = normOld === undefined ? undefined : dyNorm / normOld
    if (rate !== undefined && (rate >= 1 || (rate ** (NEWTON_MAXITER - k) / (1 - rate)) * dyNorm > tol)) break
    for (let i = 0; i < n; i++) {
      y[i] = (y[i] ?? 0) + (dy[i] ?? 0)
      d[i] = (d[i] ?? 0) + (dy[i] ?? 0)
    }
    if (dyNorm === 0 || (rate !== undefined && (rate / (1 - rate)) * dyNorm < tol)) {
      converged = true
      break
    }
    normOld = dyNorm
  }
  return { converged, iterations: k + 1, y, d }
}

function failed(s: BdfState, c: Counter, attempts: StepAttempt[], message: string): BdfState {
  return { ...s, status: 'failed', message, attempts, counts: snapshot(c) }
}

/** One accepted BDF step (with any rejections before it), following scipy's `BDF._step_impl`. */
export function bdfStep(s: BdfState, cfg: ImplicitConfig): BdfState {
  if (s.status !== 'running') throw new WhiteboxError('value', 'bdf: the integration has ended')
  if (atBound(s)) return { ...s, tOld: s.t, t: s.tBound, status: 'finished', attempts: [] }
  const c: Counter = { ...s.counts }
  const f = tally(s.f, c)
  const shape = s.y.shape
  const n = numbersOf(s.y).length
  const mass = checkMass(cfg.mass, n)
  const eyeM = mass ?? identityNumbers(n)
  const t = s.t
  let D = [...s.d]
  const order = s.order
  let nEqual = s.nEqualSteps
  const hmin = minStep(t, s.direction)
  let hAbs: number
  if (s.hAbs > cfg.maxStep) {
    hAbs = cfg.maxStep
    D = changeDifferences(D, order, cfg.maxStep / s.hAbs)
    nEqual = 0
  } else if (s.hAbs < hmin) {
    hAbs = hmin
    D = changeDifferences(D, order, hmin / s.hAbs)
    nEqual = 0
  } else hAbs = s.hAbs
  let J: Float64Array = s.jacobian
  let lu = s.lu
  let currentJac = false
  const attempts: StepAttempt[] = []
  for (let tries = 0; tries < 60; tries++) {
    if (hAbs < hmin) return failed(s, c, attempts, 'bdf: required step size is less than spacing between numbers')
    let h = hAbs * s.direction
    let tNew = t + h
    if (s.direction * (tNew - s.tBound) > 0) {
      tNew = s.tBound
      D = changeDifferences(D, order, Math.abs(tNew - t) / hAbs)
      nEqual = 0
      lu = undefined
    }
    h = tNew - t
    hAbs = Math.abs(h)
    const yPred = combine(
      null,
      D.slice(0, order + 1).map((v) => [1, v] as const),
    )
    const yPredN = numbersOf(yPred)
    let scale: Float64Array = Float64Array.from(yPredN, (v, i) => tolAt(cfg.atol, i) + tolAt(cfg.rtol, i) * Math.abs(v))
    const alpha = ALPHA[order] ?? 1
    const psi = combine(
      null,
      D.slice(1, order + 1).map((v, i) => [(GAMMA[i + 1] ?? 0) / alpha, v] as const),
    )
    const psiM = applyMass(mass, numbersOf(psi), n)
    const cc = h / alpha
    let result: ReturnType<typeof solveSystem> | undefined
    for (;;) {
      if (lu === undefined) {
        const m = new Float64Array(n * n)
        for (let k = 0; k < n * n; k++) m[k] = (eyeM[k] ?? 0) - cc * (J[k] ?? 0)
        lu = realLu(m, n)
        c.nlu += 1
      }
      result = solveSystem(f, tNew, yPredN, shape, cc, psiM, lu, scale, cfg.newtonTol, mass)
      if (result.converged || currentJac) break
      J = jacobianAt(s.f, tNew, yPred, cfg.jac)
      c.njev += 1
      lu = undefined
      currentJac = true
    }
    if (!result.converged) {
      attempts.push(attempt(h, Number.POSITIVE_INFINITY, false))
      hAbs *= 0.5
      D = changeDifferences(D, order, 0.5)
      nEqual = 0
      lu = undefined
      continue
    }
    const { iterations, y: yNew, d: dN } = result
    const safety = (0.9 * (2 * NEWTON_MAXITER + 1)) / (2 * NEWTON_MAXITER + iterations)
    scale = Float64Array.from(yNew, (v, i) => tolAt(cfg.atol, i) + tolAt(cfg.rtol, i) * Math.abs(v))
    const err = (ERROR_CONST[order] ?? 1) * scaledRms(dN, scale)
    if (!(err <= 1)) {
      attempts.push(attempt(h, err, false))
      const factor = Number.isFinite(err) ? Math.max(MIN_FACTOR, safety * err ** (-1 / (order + 1))) : MIN_FACTOR
      hAbs *= factor
      D = changeDifferences(D, order, factor)
      nEqual = 0
      continue
    }
    attempts.push(attempt(h, err, true))
    nEqual += 1
    const dValue = isTracedState(yPred, psi, s.y)
      ? implicitCorrection(
          dN,
          shape,
          (dv) =>
            combine(massTimes(mass, dv, n), [
              [-cc, f(tNew, combine(yPred, [[1, dv]]))],
              [1, massTimes(mass, psi, n)],
            ]),
          () => {
            const jn = jacobianAt(s.f, tNew, fromNumbers(yNew, shape), cfg.jac)
            return Float64Array.from(jn, (v, k) => (eyeM[k] ?? 0) - cc * v)
          },
        )
      : fromNumbers(dN, shape)
    const yValue = isTensor(dValue) && isTensor(yPred) ? fromNumbers(yNew, shape) : combine(yPred, [[1, dValue]])
    D[order + 2] = combine(dValue, [[-1, D[order + 1] as Value]])
    D[order + 1] = dValue
    for (let i = order; i >= 0; i--) D[i] = combine(D[i] as Value, [[1, D[i + 1] as Value]])
    let newHAbs = hAbs
    let newOrder = order
    let newLu: LuFactorisation | undefined = lu
    if (nEqual >= order + 1) {
      const errM =
        order > 1
          ? (ERROR_CONST[order - 1] ?? 1) * scaledRms(numbersOf(D[order] as Value), scale)
          : Number.POSITIVE_INFINITY
      const errP =
        order < BDF_MAX_ORDER
          ? (ERROR_CONST[order + 1] ?? 1) * scaledRms(numbersOf(D[order + 2] as Value), scale)
          : Number.POSITIVE_INFINITY
      const factors = [errM, err, errP].map((e, i) => e ** (-1 / (order + i)))
      let best = 0
      for (let i = 1; i < 3; i++) if ((factors[i] ?? 0) > (factors[best] ?? 0)) best = i
      newOrder = order + best - 1
      const factor = Math.min(MAX_FACTOR, safety * (factors[best] ?? 1))
      newHAbs = hAbs * factor
      D = changeDifferences(D, newOrder, factor)
      nEqual = 0
      newLu = undefined
    }
    return {
      ...s,
      tOld: t,
      yOld: s.y,
      t: tNew,
      y: yValue,
      d: D,
      order: newOrder,
      hAbs: newHAbs,
      nEqualSteps: nEqual,
      jacobian: J,
      lu: newLu,
      attempts,
      status: statusAfter(tNew, s.tBound, s.direction),
      counts: snapshot(c),
    }
  }
  return failed(s, c, attempts, 'bdf: too many step rejections')
}

/** The dense output of the last BDF step: the interpolating polynomial through the difference array. */
export function bdfDense(s: BdfState): DenseSegment {
  const order = s.order
  const h = s.hAbs * s.direction
  const D = s.d.slice(0, order + 1)
  return {
    t0: s.tOld,
    t1: s.t,
    at: (t) => {
      const terms: Term[] = []
      let p = 1
      for (let j = 0; j < order; j++) {
        p *= (t - (s.t - h * j)) / (h * (j + 1))
        terms.push([p, D[j + 1] as Value])
      }
      return combine(D[0] as Value, terms)
    },
  }
}
