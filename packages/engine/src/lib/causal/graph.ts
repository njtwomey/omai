/**
 * Effects from data given a causal graph: the graph (`pgm`) says which variables to adjust for, and the estimators of
 * this domain do the estimating. `backdoorEffect` adjusts for a minimal back-door set (regression adjustment or AIPW for a
 * binary treatment, ordinary least squares with robust errors for a numeric one), and `frontdoorEffect` applies the
 * front-door formula through mediators when an unobserved confounder rules the back-door out, for a binary treatment and
 * discrete mediators, with a bootstrap error. Identification itself (d-separation, adjustment sets, the ID algorithm's
 * cases, instruments) is `pgm/causal`.
 */

import { MdmlError } from '../base/index.ts'
import { frontdoorAdjustmentSets, minimalAdjustmentSet } from '../pgm/causal/identification.ts'
import type { DirectedModel } from '../pgm/graphs.ts'
import { bootstrapIndices, foldIn, type Key, key as makeKey } from '../random/index.ts'
import { fromStorage, toFlat } from '../tensor/index.ts'
import { covariance, mean, variance, wls } from './_dense.ts'
import { type EffectEstimate, effectFromSe } from './data.ts'
import { aipw, regressionAdjustment } from './estimation.ts'

/** Named columns of data. */
export type NamedData = Readonly<Record<string, ArrayLike<number>>>

/** An effect read through a graph. */
export interface GraphEffect {
  /** Which identification the graph allowed. */
  readonly method: 'backdoor' | 'frontdoor'
  /** The variables adjusted for (the back-door set) or passed through (the front-door mediators). */
  readonly adjustment: readonly string[]
  readonly effect: EffectEstimate
}

/** Options of {@link backdoorEffect}. */
export interface BackdoorOptions {
  /** Variables of the graph that are not observed. */
  readonly latent?: readonly string[]
  /** For a binary treatment: `'regression'` (Lin's interacted adjustment, default) or `'aipw'`. */
  readonly estimator?: 'regression' | 'aipw'
  readonly level?: number
}

const column = (data: NamedData, name: string): ArrayLike<number> => {
  const c = data[name]
  if (c === undefined) throw new MdmlError('value', `no column named ${name}`)
  return c
}

/**
 * The effect of a treatment on an outcome adjusted for a minimal back-door set found from the graph. For a 0/1 treatment
 * it is the average treatment effect by regression adjustment or AIPW; for a numeric treatment, the coefficient of a
 * least-squares regression on the treatment and the adjustment set (a linear effect), with heteroscedasticity-robust errors.
 *
 * @param data - Named columns.
 * @param graph - The causal DAG.
 * @param treatment - The treatment variable.
 * @param outcome - The outcome variable.
 * @param options - Latent variables, the estimator and the interval level.
 * @returns The adjustment set and the effect.
 * @throws MdmlError with code `value` when no back-door set exists among the observed variables.
 * @example
 * ```ts
 * import { backdoorEffect } from 'mdml-engine/causal'
 * import { dag } from 'mdml-engine/pgm'
 * const z = [0, 1, 2, 3, 4, 5, 6, 7]
 * const x = [0, 0, 1, 0, 1, 1, 1, 0]
 * const y = z.map((v, i) => v + 2 * (x[i] ?? 0) + 0.1 * Math.sin(i))
 * const r = backdoorEffect({ z, x, y }, dag([['z', 'x'], ['z', 'y'], ['x', 'y']]), 'x', 'y')
 * r.adjustment // => ['z']
 * ```
 */
export function backdoorEffect(
  data: NamedData,
  graph: DirectedModel,
  treatment: string,
  outcome: string,
  options: BackdoorOptions = {},
): GraphEffect {
  const set = minimalAdjustmentSet(
    graph,
    treatment,
    outcome,
    options.latent === undefined ? {} : { latent: options.latent },
  )
  if (set === null)
    throw new MdmlError(
      'value',
      `backdoorEffect: no back-door adjustment set for ${treatment} -> ${outcome} among the observed variables`,
    )
  const t = column(data, treatment)
  const y = column(data, outcome)
  const n = t.length
  const binary = Array.from(t).every((v) => v === 0 || v === 1)
  const level = options.level === undefined ? {} : { level: options.level }
  if (binary && set.length === 0) {
    const a = Array.from(y).filter((_, i) => t[i] === 1)
    const b = Array.from(y).filter((_, i) => t[i] === 0)
    const se = Math.sqrt(variance(a, 1) / a.length + variance(b, 1) / b.length)
    return {
      method: 'backdoor',
      adjustment: set,
      effect: effectFromSe('ate', mean(a) - mean(b), se, n, {
        ...level,
        method: 'difference of means (empty adjustment set)',
      }),
    }
  }
  const x = new Float64Array(n * set.length)
  set.forEach((name, j) => {
    const c = column(data, name)
    for (let i = 0; i < n; i++) x[i * set.length + j] = c[i] ?? 0
  })
  if (binary) {
    const causal = {
      x: fromStorage(x, [n, set.length], 'float64'),
      treatment: Float64Array.from(t),
      outcome: Float64Array.from(y),
    }
    const effect =
      options.estimator === 'aipw'
        ? aipw(causal, level)
        : regressionAdjustment(causal, { mode: 'interacted', ...level })
    return { method: 'backdoor', adjustment: set, effect }
  }
  const w = set.length + 2
  const design = new Float64Array(n * w)
  for (let i = 0; i < n; i++) {
    design[i * w] = 1
    design[i * w + 1] = t[i] ?? 0
    for (let j = 0; j < set.length; j++) design[i * w + 2 + j] = x[i * set.length + j] ?? 0
  }
  const fit = wls(design, n, w, y)
  const cov = covariance(fit, design, 'HC1')
  return {
    method: 'backdoor',
    adjustment: set,
    effect: effectFromSe('ate', fit.beta[1] ?? Number.NaN, Math.sqrt(cov[w + 1] ?? Number.NaN), n, {
      ...level,
      method: 'linear regression adjustment (HC1)',
    }),
  }
}

/** Options of {@link frontdoorEffect}. */
export interface FrontdoorOptions {
  readonly latent?: readonly string[]
  /** Bootstrap resamples for the standard error (default 200). */
  readonly resamples?: number
  readonly key?: Key
  readonly level?: number
}

function frontdoorValue(
  t: ArrayLike<number>,
  y: ArrayLike<number>,
  cells: ArrayLike<number>,
  rows: ArrayLike<number>,
): number {
  // E[Y | do(x)] = sum_m P(m | x) sum_x' E[Y | m, x'] P(x').
  const n = rows.length
  const counts = new Map<string, { n: number; s: number }>()
  const px = [0, 0]
  const mx = new Map<number, [number, number]>()
  for (let k = 0; k < n; k++) {
    const r = rows[k] ?? 0
    const a = t[r] === 1 ? 1 : 0
    px[a] = (px[a] ?? 0) + 1
    const m = cells[r] ?? 0
    const pair = mx.get(m) ?? [0, 0]
    pair[a] = (pair[a] ?? 0) + 1
    mx.set(m, pair)
    const key = `${m}|${a}`
    const c = counts.get(key) ?? { n: 0, s: 0 }
    c.n++
    c.s += y[r] ?? 0
    counts.set(key, c)
  }
  const doValue = (x: number): number => {
    let total = 0
    let weight = 0
    const nx = px[x] ?? 0
    if (nx === 0) return Number.NaN
    for (const [m, pair] of mx) {
      const pm = (pair[x] ?? 0) / nx
      let inner = 0
      let w = 0
      for (const xp of [0, 1]) {
        const c = counts.get(`${m}|${xp}`)
        if (c === undefined) continue
        inner += (c.s / c.n) * ((px[xp] ?? 0) / n)
        w += (px[xp] ?? 0) / n
      }
      if (w === 0) continue
      total += pm * (inner / w)
      weight += pm
    }
    return total / weight
  }
  return doValue(1) - doValue(0)
}

/**
 * The effect of a binary treatment through the front-door criterion: when an unobserved confounder opens every back-door
 * path but the treatment acts on the outcome only through observed mediators, `E[Y | do(x)] = sum_m P(m | x) sum_x' E[Y |
 * m, x'] P(x')`. The mediators (a front-door set the graph supplies) must be discrete; the standard error is the
 * bootstrap's over units.
 *
 * @param data - Named columns (the treatment 0/1, the mediators discrete, the outcome numeric).
 * @param graph - The causal DAG.
 * @param treatment - The treatment variable.
 * @param outcome - The outcome variable.
 * @param options - Latent variables, resamples and key.
 * @returns The front-door set and the average treatment effect.
 * @throws MdmlError with code `value` when the graph has no front-door set.
 * @example
 * ```ts
 * import { frontdoorEffect } from 'mdml-engine/causal'
 * import { dag } from 'mdml-engine/pgm'
 * const x = [0, 0, 0, 0, 1, 1, 1, 1]
 * const m = [0, 0, 0, 1, 0, 1, 1, 1]
 * const y = m.map((v) => 2 * v)
 * const g = dag([['u', 'x'], ['u', 'y'], ['x', 'm'], ['m', 'y']])
 * frontdoorEffect({ x, m, y }, g, 'x', 'y', { latent: ['u'], resamples: 5 }).effect.estimate.value // => 1
 * ```
 */
export function frontdoorEffect(
  data: NamedData,
  graph: DirectedModel,
  treatment: string,
  outcome: string,
  options: FrontdoorOptions = {},
): GraphEffect {
  const sets = frontdoorAdjustmentSets(
    graph,
    treatment,
    outcome,
    options.latent === undefined ? {} : { latent: options.latent },
  )
  const set = sets[0]
  if (set === undefined)
    throw new MdmlError('value', `frontdoorEffect: no front-door set for ${treatment} -> ${outcome}`)
  const t = column(data, treatment)
  const y = column(data, outcome)
  const n = t.length
  // One cell per distinct combination of the mediators' values.
  const levels = new Map<string, number>()
  const cells = Int32Array.from({ length: n }, (_, i) => {
    const key = set.map((name) => column(data, name)[i] ?? 0).join(',')
    if (!levels.has(key)) levels.set(key, levels.size)
    return levels.get(key) ?? 0
  })
  const all = Int32Array.from({ length: n }, (_, i) => i)
  const value = frontdoorValue(t, y, cells, all)
  const key = options.key ?? makeKey(0)
  const draws: number[] = []
  for (let b = 0; b < (options.resamples ?? 200); b++) {
    const rows = toFlat(bootstrapIndices(foldIn(key, b), n))
    const v = frontdoorValue(t, y, cells, rows)
    if (Number.isFinite(v)) draws.push(v)
  }
  return {
    method: 'frontdoor',
    adjustment: set,
    effect: effectFromSe('ate', value, Math.sqrt(variance(draws, 1)), n, {
      ...(options.level === undefined ? {} : { level: options.level }),
      method: 'front-door adjustment (bootstrap standard error)',
    }),
  }
}
