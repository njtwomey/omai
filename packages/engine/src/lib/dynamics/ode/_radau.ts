// Radau IIA (order 5): scipy's Radau ported, with an implicit-function correction for traced steps.
import { MdmlError } from '../../base/index.ts'
import type { LuFactorisation } from '../../numerics/linalg/index.ts'
import { matmul, reshape, sliceAxis, stack, type Value } from '../../tensor/index.ts'
import {
  applyMass,
  checkMass,
  complexApply,
  complexLu,
  type ImplicitConfig,
  identityNumbers,
  implicitCorrection,
  isTracedState,
  type JacobianOption,
  jacobianAt,
  luApply,
  realLu,
} from './_newton.ts'
import { atBound, attempt, type Counter, snapshot, statusAfter, tally, towards } from './_step.ts'
import {
  allFinite,
  combine,
  errorScale,
  fromNumbers,
  minStep,
  numbersOf,
  rms,
  scaledRms,
  type Term,
  tolAt,
} from './_vec.ts'
import type { RadauState } from './implicit.ts'
import { type DenseSegment, hermiteSegment, type StepAttempt, type VectorField } from './problem.ts'

const S6 = Math.sqrt(6)
const RADAU_C = [(4 - S6) / 10, (4 + S6) / 10, 1]
const RADAU_E = [(-13 - 7 * S6) / 3, (-13 + 7 * S6) / 3, -1 / 3]
const MU_REAL = 3 + 3 ** (2 / 3) - 3 ** (1 / 3)
const MU_COMPLEX_RE = 3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3))
const MU_COMPLEX_IM = -0.5 * (3 ** (5 / 6) + 3 ** (7 / 6))
const RADAU_T = [
  [0.09443876248897524, -0.14125529502095421, 0.03002919410514742],
  [0.25021312296533332, 0.20412935229379994, -0.38294211275726192],
  [1, 1, 0],
]
const RADAU_TI = [
  [4.178718591551904, 0.32768282076106237, 0.52337644549944951],
  [-4.178718591551904, -0.32768282076106237, 0.47662355450055044],
  [0.50287263494578682, -2.571926949855605, 0.59603920482822492],
]
const RADAU_P = [
  [13 / 3 + (7 * S6) / 3, -23 / 3 - (22 * S6) / 3, 10 / 3 + 5 * S6],
  [13 / 3 - (7 * S6) / 3, -23 / 3 + (22 * S6) / 3, 10 / 3 - 5 * S6],
  [1 / 3, -8 / 3, 10 / 3],
]
/** The Radau IIA (order 5) coefficient matrix. */
const RADAU_A = [
  [(88 - 7 * S6) / 360, (296 - 169 * S6) / 1800, (-2 + 3 * S6) / 225],
  [(296 + 169 * S6) / 1800, (88 + 7 * S6) / 360, (-2 - 3 * S6) / 225],
  [(16 - S6) / 36, (16 + S6) / 36, 1 / 9],
]
const RADAU_NEWTON_MAXITER = 6
const MIN_FACTOR = 0.2
const MAX_FACTOR = 10

function predictFactor(hAbs: number, hAbsOld: number | undefined, err: number, errOld: number | undefined): number {
  const multiplier =
    errOld === undefined || hAbsOld === undefined || err === 0 ? 1 : (hAbs / hAbsOld) * (errOld / err) ** 0.25
  return Math.min(1, multiplier) * err ** -0.25
}

export function radauStep(s: RadauState, cfg: ImplicitConfig): RadauState {
  if (s.status !== 'running') throw new MdmlError('value', 'radau: the integration has ended')
  if (atBound(s)) return { ...s, tOld: s.t, t: s.tBound, status: 'finished', attempts: [] }
  const c: Counter = { ...s.counts }
  const f = tally(s.f, c)
  const shape = s.y.shape
  const y = numbersOf(s.y)
  const fyN = numbersOf(s.fy)
  const n = y.length
  const mass = checkMass(cfg.mass, n)
  const eyeM = mass ?? identityNumbers(n)
  const t = s.t
  const hmin = minStep(t, s.direction)
  let hAbs: number
  let hAbsOld = s.hAbsOld
  let errOld = s.errorNormOld
  if (s.hAbs > cfg.maxStep) {
    hAbs = cfg.maxStep
    hAbsOld = undefined
    errOld = undefined
  } else if (s.hAbs < hmin) {
    hAbs = hmin
    hAbsOld = undefined
    errOld = undefined
  } else hAbs = s.hAbs
  let J: Float64Array = s.jacobian
  let luR = s.luReal
  let luC = s.luComplex
  let currentJac = s.currentJac
  let rejected = false
  const attempts: StepAttempt[] = []
  const prevDense = s.z.length === 3 ? radauDense(s) : undefined
  for (let tries = 0; tries < 60; tries++) {
    if (hAbs < hmin) {
      return {
        ...s,
        status: 'failed',
        message: 'radau: required step size is less than spacing between numbers',
        attempts,
        counts: snapshot(c),
      }
    }
    const { tNew, h } = towards(t, hAbs, s.direction, s.tBound)
    hAbs = Math.abs(h)
    const z0: Float64Array[] =
      prevDense === undefined
        ? [0, 1, 2].map(() => new Float64Array(n))
        : RADAU_C.map((ci) => {
            const v = numbersOf(prevDense.at(t + h * ci))
            return Float64Array.from(v, (x, i) => x - (y[i] ?? 0))
          })
    let scale: Float64Array = Float64Array.from(y, (v, i) => tolAt(cfg.atol, i) + Math.abs(v) * tolAt(cfg.rtol, i))
    let converged = false
    let result: Collocation | undefined
    for (;;) {
      if (luR === undefined || luC === undefined) {
        const re = new Float64Array(n * n)
        const reC = new Float64Array(n * n)
        const imC = new Float64Array(n * n)
        for (let k = 0; k < n * n; k++) {
          re[k] = (MU_REAL / h) * (eyeM[k] ?? 0) - (J[k] ?? 0)
          reC[k] = (MU_COMPLEX_RE / h) * (eyeM[k] ?? 0) - (J[k] ?? 0)
          imC[k] = (MU_COMPLEX_IM / h) * (eyeM[k] ?? 0)
        }
        luR = realLu(re, n)
        luC = complexLu(reC, imC, n)
        c.nlu += 2
      }
      result = solveCollocation(f, t, y, shape, h, z0, scale, cfg.newtonTol, luR, luC, mass, c)
      converged = result.converged
      if (converged || currentJac) break
      J = jacobianAt(s.f, t, s.y, cfg.jac)
      c.njev += 1
      currentJac = true
      luR = undefined
      luC = undefined
    }
    if (!converged || result === undefined || luR === undefined) {
      attempts.push(attempt(h, Number.POSITIVE_INFINITY, false))
      hAbs *= 0.5
      luR = undefined
      luC = undefined
      continue
    }
    const Z = result.z
    const yNew = Float64Array.from(y, (v, i) => v + (Z[2]?.[i] ?? 0))
    const ze = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      ze[i] =
        ((RADAU_E[0] ?? 0) * (Z[0]?.[i] ?? 0) +
          (RADAU_E[1] ?? 0) * (Z[1]?.[i] ?? 0) +
          (RADAU_E[2] ?? 0) * (Z[2]?.[i] ?? 0)) /
        h
    }
    const mze = applyMass(mass, ze, n)
    let error = luApply(
      luR,
      Float64Array.from(fyN, (v, i) => v + (mze[i] ?? 0)),
    )
    scale = errorScale(y, yNew, cfg.rtol, cfg.atol)
    let err = scaledRms(error, scale)
    const safety = (0.9 * (2 * RADAU_NEWTON_MAXITER + 1)) / (2 * RADAU_NEWTON_MAXITER + result.iterations)
    if (rejected && err > 1) {
      const shifted = fromNumbers(
        Float64Array.from(y, (v, i) => v + (error[i] ?? 0)),
        shape,
      )
      const fe = numbersOf(f(t, shifted))
      error = luApply(
        luR,
        Float64Array.from(fe, (v, i) => v + (mze[i] ?? 0)),
      )
      err = scaledRms(error, scale)
    }
    if (!(err <= 1)) {
      attempts.push(attempt(h, err, false))
      const factor = Number.isFinite(err) ? predictFactor(hAbs, hAbsOld, err, errOld) : MIN_FACTOR
      hAbs *= Math.max(MIN_FACTOR, safety * factor)
      luR = undefined
      luC = undefined
      rejected = true
      continue
    }
    attempts.push(attempt(h, err, true))
    const recompute = result.iterations > 2 && (result.rate ?? 0) > 1e-3
    let factor = Math.min(MAX_FACTOR, safety * predictFactor(hAbs, hAbsOld, err, errOld))
    if (!recompute && factor < 1.2) factor = 1
    else {
      luR = undefined
      luC = undefined
    }
    // The stage increments as values: exact numbers, differentiable through the implicit function theorem.
    let zValues: Value[]
    let yNewValue: Value
    if (isTracedState(s.y, s.fy)) {
      const flatZ = new Float64Array(3 * n)
      for (let i = 0; i < 3; i++) flatZ.set(Z[i] ?? new Float64Array(n), i * n)
      const zt = implicitCorrection(
        flatZ,
        [3, n],
        (zv) => radauResidual(s.f, t, s.y, h, zv, n, mass),
        () => radauNewtonMatrix(s.f, t, y, shape, h, Z, n, mass, cfg.jac),
      )
      zValues = [0, 1, 2].map((i) => reshape(sliceRow(zt, i, n), shape))
      yNewValue = combine(s.y, [[1, zValues[2] as Value]])
    } else {
      zValues = Z.map((zi) => fromNumbers(zi, shape))
      yNewValue = fromNumbers(yNew, shape)
    }
    const fNew = f(tNew, yNewValue)
    if (recompute) {
      J = jacobianAt(s.f, tNew, yNewValue, cfg.jac)
      c.njev += 1
      currentJac = true
    } else currentJac = false
    return {
      ...s,
      tOld: t,
      yOld: s.y,
      t: tNew,
      y: yNewValue,
      fy: fNew,
      jacobian: J,
      luReal: luR,
      luComplex: luC,
      currentJac,
      hAbsOld: s.hAbs,
      errorNormOld: err,
      hAbs: hAbs * factor,
      z: zValues,
      h,
      attempts,
      status: statusAfter(tNew, s.tBound, s.direction),
      counts: snapshot(c),
    }
  }
  return { ...s, status: 'failed', message: 'radau: too many step rejections', attempts, counts: snapshot(c) }
}

function sliceRow(z: Value, i: number, n: number): Value {
  return reshape(sliceAxis(z, 0, i, i + 1), [n])
}

function radauResidual(
  f: VectorField,
  t: number,
  y: Value,
  h: number,
  z: Value,
  n: number,
  mass: Float64Array | undefined,
): Value {
  const rows = [0, 1, 2].map((i) => reshape(sliceRow(z, i, n), y.shape))
  const F = rows.map((zi, i) => f(t + (RADAU_C[i] ?? 0) * h, combine(y, [[1, zi]])))
  const out = rows.map((zi, i) => {
    const terms: Term[] = F.map((fj, j) => [-h * (RADAU_A[i]?.[j] ?? 0), fj] as const)
    const mz = mass === undefined ? zi : reshape(matmul(fromNumbers(mass, [n, n]), reshape(zi, [n, 1])), y.shape)
    return combine(mz, terms)
  })
  return stack(out.map((r) => reshape(r, [n])))
}

function radauNewtonMatrix(
  f: VectorField,
  t: number,
  y: Float64Array,
  shape: readonly number[],
  h: number,
  Z: readonly Float64Array[],
  n: number,
  mass: Float64Array | undefined,
  jac: JacobianOption,
): Float64Array {
  const N = 3 * n
  const m = new Float64Array(N * N)
  const eyeM = mass ?? identityNumbers(n)
  const Js = [0, 1, 2].map((j) =>
    jacobianAt(
      f,
      t + (RADAU_C[j] ?? 0) * h,
      fromNumbers(
        Float64Array.from(y, (v, k) => v + (Z[j]?.[k] ?? 0)),
        shape,
      ),
      jac,
    ),
  )
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      const a = RADAU_A[i]?.[j] ?? 0
      const Jj = Js[j] as Float64Array
      for (let r = 0; r < n; r++) {
        for (let q = 0; q < n; q++) {
          const v = (i === j ? (eyeM[r * n + q] ?? 0) : 0) - h * a * (Jj[r * n + q] ?? 0)
          m[(i * n + r) * N + j * n + q] = v
        }
      }
    }
  }
  return m
}

interface Collocation {
  readonly converged: boolean
  readonly iterations: number
  readonly z: Float64Array[]
  readonly rate: number | undefined
}

function solveCollocation(
  f: VectorField,
  t: number,
  y: Float64Array,
  shape: readonly number[],
  h: number,
  z0: readonly Float64Array[],
  scale: Float64Array,
  tol: number,
  luR: LuFactorisation,
  luC: LuFactorisation,
  mass: Float64Array | undefined,
  _c: Counter,
): Collocation {
  const n = y.length
  const mReal = MU_REAL / h
  const mcRe = MU_COMPLEX_RE / h
  const mcIm = MU_COMPLEX_IM / h
  const W = [0, 1, 2].map((i) => {
    const row = new Float64Array(n)
    for (let k = 0; k < n; k++) {
      let s = 0
      for (let j = 0; j < 3; j++) s += (RADAU_TI[i]?.[j] ?? 0) * (z0[j]?.[k] ?? 0)
      row[k] = s
    }
    return row
  })
  let Z = z0.map((r) => Float64Array.from(r))
  let normOld: number | undefined
  let rate: number | undefined
  let converged = false
  let k = 0
  for (; k < RADAU_NEWTON_MAXITER; k++) {
    const F = [0, 1, 2].map((i) =>
      numbersOf(
        f(
          t + (RADAU_C[i] ?? 0) * h,
          fromNumbers(
            Float64Array.from(y, (v, q) => v + (Z[i]?.[q] ?? 0)),
            shape,
          ),
        ),
      ),
    )
    if (!F.every((r) => allFinite(r))) break
    const w0 = applyMass(mass, W[0] as Float64Array, n)
    const w1 = applyMass(mass, W[1] as Float64Array, n)
    const w2 = applyMass(mass, W[2] as Float64Array, n)
    const fReal = new Float64Array(n)
    const fcRe = new Float64Array(n)
    const fcIm = new Float64Array(n)
    for (let q = 0; q < n; q++) {
      let a = 0
      let b = 0
      let d = 0
      for (let i = 0; i < 3; i++) {
        const v = F[i]?.[q] ?? 0
        a += v * (RADAU_TI[0]?.[i] ?? 0)
        b += v * (RADAU_TI[1]?.[i] ?? 0)
        d += v * (RADAU_TI[2]?.[i] ?? 0)
      }
      fReal[q] = a - mReal * (w0[q] ?? 0)
      // (mcRe + i mcIm) (w1 + i w2) = (mcRe w1 - mcIm w2) + i (mcRe w2 + mcIm w1)
      fcRe[q] = b - (mcRe * (w1[q] ?? 0) - mcIm * (w2[q] ?? 0))
      fcIm[q] = d - (mcRe * (w2[q] ?? 0) + mcIm * (w1[q] ?? 0))
    }
    const dR = luApply(luR, fReal)
    const dC = complexApply(luC, fcRe, fcIm)
    const dW = [dR, dC.re, dC.im]
    const all = new Float64Array(3 * n)
    for (let i = 0; i < 3; i++) for (let q = 0; q < n; q++) all[i * n + q] = (dW[i]?.[q] ?? 0) / (scale[q] ?? 1)
    const dwNorm = rms(all)
    if (normOld !== undefined) rate = dwNorm / normOld
    if (rate !== undefined && (rate >= 1 || (rate ** (RADAU_NEWTON_MAXITER - k) / (1 - rate)) * dwNorm > tol)) {
      break
    }
    for (let i = 0; i < 3; i++)
      for (let q = 0; q < n; q++) (W[i] as Float64Array)[q] = (W[i]?.[q] ?? 0) + (dW[i]?.[q] ?? 0)
    Z = [0, 1, 2].map((i) => {
      const row = new Float64Array(n)
      for (let q = 0; q < n; q++) {
        let s = 0
        for (let j = 0; j < 3; j++) s += (RADAU_T[i]?.[j] ?? 0) * (W[j]?.[q] ?? 0)
        row[q] = s
      }
      return row
    })
    if (dwNorm === 0 || (rate !== undefined && (rate / (1 - rate)) * dwNorm < tol)) {
      converged = true
      break
    }
    normOld = dwNorm
  }
  return { converged, iterations: k + 1, z: Z, rate }
}

export function radauDense(s: RadauState): DenseSegment {
  if (s.z.length !== 3) return hermiteSegment(s.tOld, s.yOld, s.fy, s.t, s.y, s.fy)
  const h = s.h
  return {
    t0: s.tOld,
    t1: s.t,
    at: (t) => {
      const x = h === 0 ? 0 : (t - s.tOld) / h
      const p = [x, x * x, x * x * x]
      const terms: Term[] = s.z.map((zi, i) => {
        const row = RADAU_P[i] ?? []
        return [(row[0] ?? 0) * (p[0] ?? 0) + (row[1] ?? 0) * (p[1] ?? 0) + (row[2] ?? 0) * (p[2] ?? 0), zi] as const
      })
      return combine(s.yOld, terms)
    },
  }
}
