import { type Json, linspaceNumbers, WhiteboxError } from '../base/index.ts'
import { type TableData, table } from '../data/index.ts'
import { hypervolume } from '../optim/index.ts'
import type { SpaceValue } from '../space/index.ts'
import { type Leaf, leafCoordinate, leavesOf, nativeOf } from './_leaves.ts'
import {
  bestTrials,
  completeTrials,
  exportTrials,
  lossMatrix,
  lossOf,
  type Study,
  type StudyDirection,
  type Trial,
} from './study.ts'
import { fitGpSurrogate } from './surrogate.ts'

function objectiveIndex(study: Study, objective: number | undefined): number {
  const j = objective ?? 0
  if (!(Number.isInteger(j) && j >= 0 && j < study.directions.length))
    throw new WhiteboxError('value', `analysis: no objective ${j} in a study with ${study.directions.length}`)
  return j
}

/**
 * The trials of a study as a table (a `TableData` dataset): one row per trial with its number, state, values,
 * parameters (`params_<path>`; options as categories), budget and last intermediate value, ready for the data tools
 * and for plotting.
 *
 * @param study - The study.
 * @returns The table.
 * @concept hyperparameter-optimisation
 * @example
 * ```ts
 * import { createStudy, trialsTable } from 'whitebox-engine/tune'
 * import { dict, real } from 'whitebox-engine/space'
 * const study = createStudy(dict({ x: real(0, 1) }), {
 *   trials: [{ number: 0, params: { x: 0.5 }, state: 'complete', values: [0.25], intermediate: [], source: 'random' }],
 * })
 * Object.keys(trialsTable(study).columns) // => ['number', 'state', 'value', 'params_x', 'source']
 * ```
 */
export function trialsTable(study: Study): TableData {
  const rows = exportTrials(study)
  const names: string[] = []
  for (const r of rows) for (const k of Object.keys(r)) if (!names.includes(k)) names.push(k)
  const columns: Record<string, (number | null)[] | (string | null)[]> = {}
  for (const name of names) {
    const values = rows.map((r) => r[name] ?? null)
    const numeric = values.every((v) => v === null || typeof v === 'number')
    columns[name] = numeric
      ? values.map((v) => (v === null ? null : (v as number)))
      : values.map((v) => (v === null ? null : String(v as Json)))
  }
  return table(columns, { meta: { name: `${study.name} trials`, description: 'the trials of a study' } })
}

/** The optimisation history of a study: each trial's value and the best value so far. */
export interface History {
  readonly number: readonly number[]
  /** The value of each trial (NaN unless complete). */
  readonly value: readonly number[]
  /** The best value up to and including each trial (NaN before the first complete one). */
  readonly best: readonly number[]
}

/**
 * The optimisation history (Optuna's `plot_optimization_history` data): trial numbers, values and the running best of
 * one objective.
 *
 * @param study - The study.
 * @param objective - Which objective (default 0).
 * @returns The history.
 * @example
 * ```ts
 * import { createStudy, optimisationHistory } from 'whitebox-engine/tune'
 * import { real } from 'whitebox-engine/space'
 * const trial = (number: number, value: number) =>
 *   ({ number, params: value, state: 'complete', values: [value], intermediate: [], source: 'x' }) as const
 * optimisationHistory(createStudy(real(0, 9), { trials: [trial(0, 3), trial(1, 5), trial(2, 1)] })).best // => [3, 3, 1]
 * ```
 */
export function optimisationHistory(study: Study, objective?: number): History {
  const j = objectiveIndex(study, objective)
  const d = study.directions[j] as StudyDirection
  let best = Number.NaN
  const value: number[] = []
  const bests: number[] = []
  for (const t of study.trials) {
    const v = t.state === 'complete' ? (t.values[j] ?? Number.NaN) : Number.NaN
    if (!Number.isNaN(v) && (Number.isNaN(best) || lossOf(d, v) < lossOf(d, best))) best = v
    value.push(v)
    bests.push(best)
  }
  return { number: study.trials.map((t) => t.number), value, best: bests }
}

/**
 * The intermediate values of every trial (Optuna's `plot_intermediate_values` data), with how each trial ended.
 *
 * @param study - The study.
 * @returns One series per trial that reported.
 * @example
 * ```ts
 * import { createStudy, intermediateValues } from 'whitebox-engine/tune'
 * import { real } from 'whitebox-engine/space'
 * const t = { number: 0, params: 0, state: 'pruned', values: [], intermediate: [{ step: 0, value: 2 }], source: 'x' } as const
 * intermediateValues(createStudy(real(0, 1), { trials: [t] })) // => [{ number: 0, state: 'pruned', steps: [0], values: [2] }]
 * ```
 */
export function intermediateValues(
  study: Study,
): { readonly number: number; readonly state: string; readonly steps: number[]; readonly values: number[] }[] {
  return study.trials
    .filter((t) => t.intermediate.length > 0)
    .map((t) => ({
      number: t.number,
      state: t.state,
      steps: t.intermediate.map((iv) => iv.step),
      values: t.intermediate.map((iv) => iv.value),
    }))
}

/** One axis of a parallel-coordinate plot: a parameter's values per trial, with its range or option labels. */
export interface ParameterAxis {
  readonly path: string
  readonly kind: Leaf['kind']
  readonly log: boolean
  /** Values per trial (natural units; option indices for options; NaN where inactive). */
  readonly values: readonly number[]
  readonly range: readonly [number, number]
  readonly labels?: readonly string[]
}

function axes(study: Study, trials: readonly Trial[]): ParameterAxis[] {
  const leaves = leavesOf(study.space)
  const natives = trials.map((t) => nativeOf(study.space, t.params))
  return leaves.map((l, j) => ({
    path: l.path,
    kind: l.kind,
    log: l.log,
    values: natives.map((r) => r[j] ?? Number.NaN),
    range: [l.low, l.high] as const,
    ...(l.kind === 'categorical' ? { labels: Array.from({ length: l.n }, (_, i) => String(l.value(i) as Json)) } : {}),
  }))
}

/**
 * Parallel-coordinate data (Optuna's `plot_parallel_coordinate`): for the complete trials, one axis per parameter and
 * the objective values.
 *
 * @param study - The study.
 * @param objective - Which objective (default 0).
 * @returns Trial numbers, parameter axes and objective values.
 * @example
 * ```ts
 * import { createStudy, parallelCoordinates } from 'whitebox-engine/tune'
 * import { dict, real, choice } from 'whitebox-engine/space'
 * const t = { number: 0, params: { a: 0.5, b: 'y' }, state: 'complete', values: [1], intermediate: [], source: 'x' } as const
 * parallelCoordinates(createStudy(dict({ a: real(0, 1), b: choice(['x', 'y']) }), { trials: [t] })).axes.map((a) => a.values)
 * // => [[0.5], [1]]
 * ```
 */
export function parallelCoordinates(
  study: Study,
  objective?: number,
): {
  readonly numbers: readonly number[]
  readonly axes: readonly ParameterAxis[]
  readonly objective: readonly number[]
} {
  const j = objectiveIndex(study, objective)
  const complete = completeTrials(study)
  return {
    numbers: complete.map((t) => t.number),
    axes: axes(study, complete),
    objective: complete.map((t) => t.values[j] ?? Number.NaN),
  }
}

/**
 * Slice data (Optuna's `plot_slice`): for each parameter, the complete trials' parameter values against their objective
 * values (trials where the parameter is inactive are left out of that slice).
 *
 * @param study - The study.
 * @param objective - Which objective (default 0).
 * @returns One slice per parameter.
 * @example
 * ```ts
 * import { createStudy, sliceData } from 'whitebox-engine/tune'
 * import { dict, real } from 'whitebox-engine/space'
 * const t = { number: 0, params: { a: 0.5 }, state: 'complete', values: [2], intermediate: [], source: 'x' } as const
 * sliceData(createStudy(dict({ a: real(0, 1) }), { trials: [t] })) // => [{ path: 'a', numbers: [0], x: [0.5], y: [2] }]
 * ```
 */
export function sliceData(
  study: Study,
  objective?: number,
): { readonly path: string; readonly numbers: number[]; readonly x: number[]; readonly y: number[] }[] {
  const { numbers, axes: all, objective: y } = parallelCoordinates(study, objective)
  return all.map((a) => {
    const keep = a.values.map((v) => !Number.isNaN(v))
    return {
      path: a.path,
      numbers: numbers.filter((_, i) => keep[i]),
      x: a.values.filter((_, i) => keep[i]),
      y: y.filter((_, i) => keep[i]),
    }
  })
}

/** Contour data of two parameters: observed points and a surrogate's predictions on a grid. */
export interface ContourData {
  readonly x: { readonly path: string; readonly ticks: readonly number[]; readonly log: boolean }
  readonly y: { readonly path: string; readonly ticks: readonly number[]; readonly log: boolean }
  /** Predicted objective at (ticks.x[i], ticks.y[k]) as z[k][i] (rows along y). */
  readonly z: readonly (readonly number[])[]
  readonly points: readonly {
    readonly number: number
    readonly x: number
    readonly y: number
    readonly value: number
  }[]
}

/**
 * Contour data of two parameters (Optuna's `plot_contour`): the complete trials' points, and the objective over a
 * grid of the two parameters predicted by a Gaussian-process surrogate fitted to those two coordinates (the other
 * parameters marginalised by the fit's noise), in the units of the objective.
 *
 * @param study - The study.
 * @param xPath - The first parameter's path.
 * @param yPath - The second parameter's path.
 * @param options - Grid size per axis (default 20) and objective index.
 * @returns The contour data.
 * @throws WhiteboxError with code `value` for unknown paths or too few complete trials.
 * @example
 * ```ts
 * import { createStudy, contourData } from 'whitebox-engine/tune'
 * import { dict, real } from 'whitebox-engine/space'
 * const trials = [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]].map(([a = 0, b = 0], number) =>
 *   ({ number, params: { a, b }, state: 'complete', values: [a + b], intermediate: [], source: 'x' }) as const)
 * const c = contourData(createStudy(dict({ a: real(0, 1), b: real(0, 1) }), { trials }), 'a', 'b', { gridSize: 3 })
 * c.z.length // => 3
 * ```
 */
export function contourData(
  study: Study,
  xPath: string,
  yPath: string,
  options: { readonly gridSize?: number; readonly objective?: number } = {},
): ContourData {
  const j = objectiveIndex(study, options.objective)
  const d = study.directions[j] as StudyDirection
  const leaves = leavesOf(study.space)
  const ix = leaves.findIndex((l) => l.path === xPath)
  const iy = leaves.findIndex((l) => l.path === yPath)
  if (ix < 0 || iy < 0) throw new WhiteboxError('value', `contourData: unknown parameter ${ix < 0 ? xPath : yPath}`)
  const lx = leaves[ix] as Leaf
  const ly = leaves[iy] as Leaf
  const pts = completeTrials(study).flatMap((t) => {
    const n = nativeOf(study.space, t.params)
    const x = n[ix] ?? Number.NaN
    const y = n[iy] ?? Number.NaN
    return Number.isNaN(x) || Number.isNaN(y) ? [] : [{ number: t.number, x, y, value: t.values[j] ?? Number.NaN }]
  })
  if (pts.length < 2) throw new WhiteboxError('value', 'contourData: need at least two complete trials with both')
  const size = options.gridSize ?? 20
  const ticks = (l: Leaf): number[] =>
    l.kind === 'categorical'
      ? Array.from({ length: l.n }, (_, i) => i)
      : l.log
        ? linspaceNumbers(Math.log(l.low), Math.log(l.high), size).map(Math.exp)
        : linspaceNumbers(l.low, l.high, size)
  const tx = ticks(lx)
  const ty = ticks(ly)
  const column = (l: Leaf, v: number): number => (l.kind === 'categorical' ? v : leafCoordinate(l, v))
  const gp = fitGpSurrogate(
    pts.map((p) => [column(lx, p.x), column(ly, p.y)]),
    pts.map((p) => lossOf(d, p.value)),
    {
      categorical: [lx.kind === 'categorical', ly.kind === 'categorical'],
      sizes: [lx.n || 1, ly.n || 1],
    },
  )
  const z = ty.map((y) => gp.predict(tx.map((x) => [column(lx, x), column(ly, y)])).mean.map((m) => lossOf(d, m)))
  return {
    x: { path: xPath, ticks: tx, log: lx.log },
    y: { path: yPath, ticks: ty, log: ly.log },
    z,
    points: pts,
  }
}

/**
 * The Pareto front of a multi-objective study (Optuna's `plot_pareto_front` data): the non-dominated complete trials
 * and the dominated ones, with their values.
 *
 * @param study - The study.
 * @returns Front and dominated trials as numbers and value rows.
 * @concept multi-objective-optimisation
 * @example
 * ```ts
 * import { createStudy, paretoFrontData } from 'whitebox-engine/tune'
 * import { real } from 'whitebox-engine/space'
 * const trial = (number: number, values: number[]) =>
 *   ({ number, params: 0, state: 'complete', values, intermediate: [], source: 'x' }) as const
 * const study = createStudy(real(0, 1), { directions: ['minimize', 'minimize'], trials: [trial(0, [1, 2]), trial(1, [2, 3])] })
 * paretoFrontData(study).front.numbers // => [0]
 * ```
 */
export function paretoFrontData(study: Study): {
  readonly front: { readonly numbers: number[]; readonly values: number[][] }
  readonly dominated: { readonly numbers: number[]; readonly values: number[][] }
} {
  const best = new Set(bestTrials(study).map((t) => t.number))
  const complete = completeTrials(study)
  const part = (inFront: boolean): { numbers: number[]; values: number[][] } => {
    const ts = complete.filter((t) => best.has(t.number) === inFront)
    return { numbers: ts.map((t) => t.number), values: ts.map((t) => [...t.values]) }
  }
  return { front: part(true), dominated: part(false) }
}

/**
 * The hypervolume of the complete trials so far, after each trial (Optuna's `plot_hypervolume_history`).
 *
 * @param study - The study.
 * @param reference - The reference point, in the objectives' units.
 * @returns One hypervolume per trial.
 * @concept hypervolume
 * @example
 * ```ts
 * import { createStudy, hypervolumeHistory } from 'whitebox-engine/tune'
 * import { real } from 'whitebox-engine/space'
 * const trial = (number: number, values: number[]) =>
 *   ({ number, params: 0, state: 'complete', values, intermediate: [], source: 'x' }) as const
 * const study = createStudy(real(0, 1), { directions: ['minimize', 'minimize'], trials: [trial(0, [1, 2]), trial(1, [2, 1])] })
 * hypervolumeHistory(study, [3, 3]) // => [2, 3]
 * ```
 */
export function hypervolumeHistory(study: Study, reference: readonly number[]): number[] {
  if (reference.length !== study.directions.length)
    throw new WhiteboxError('shape', 'hypervolumeHistory: one reference coordinate per objective')
  const ref = study.directions.map((d, j) => lossOf(d, reference[j] ?? 0))
  const losses = lossMatrix(study, study.trials)
  const kept: number[][] = []
  return study.trials.map((t, i) => {
    if (t.state === 'complete') kept.push(losses[i] as number[])
    return kept.length === 0 ? 0 : hypervolume(kept, ref)
  })
}

/**
 * The empirical distribution of a study's objective values (Optuna's `plot_edf`): sorted values and the fraction of
 * complete trials at or below each.
 *
 * @param study - The study.
 * @param objective - Which objective (default 0).
 * @returns Sorted values and cumulative fractions.
 * @example
 * ```ts
 * import { createStudy, empiricalDistribution } from 'whitebox-engine/tune'
 * import { real } from 'whitebox-engine/space'
 * const trial = (number: number, value: number) =>
 *   ({ number, params: 0, state: 'complete', values: [value], intermediate: [], source: 'x' }) as const
 * empiricalDistribution(createStudy(real(0, 1), { trials: [trial(0, 3), trial(1, 1)] })) // => { values: [1, 3], fraction: [0.5, 1] }
 * ```
 */
export function empiricalDistribution(
  study: Study,
  objective?: number,
): { readonly values: number[]; readonly fraction: number[] } {
  const j = objectiveIndex(study, objective)
  const values = completeTrials(study)
    .map((t) => t.values[j] ?? Number.NaN)
    .filter((v) => !Number.isNaN(v))
    .sort((a, b) => a - b)
  return { values, fraction: values.map((_, i) => (i + 1) / values.length) }
}

/**
 * The parameters of the best trial of each objective (a summary for reports).
 *
 * @param study - The study.
 * @returns One configuration per objective (`undefined` when nothing is complete).
 * @example
 * ```ts
 * import { createStudy, bestParamsByObjective } from 'whitebox-engine/tune'
 * import { real } from 'whitebox-engine/space'
 * const trial = (number: number, values: number[]) =>
 *   ({ number, params: number, state: 'complete', values, intermediate: [], source: 'x' }) as const
 * const study = createStudy(real(0, 9), { directions: ['minimize', 'maximize'], trials: [trial(0, [1, 1]), trial(1, [2, 5])] })
 * bestParamsByObjective(study) // => [0, 1]
 * ```
 */
export function bestParamsByObjective(study: Study): (SpaceValue | undefined)[] {
  return study.directions.map((_, j) => {
    const h = optimisationHistory(study, j)
    const best = h.best.at(-1)
    const i = h.value.indexOf(best as number)
    return i < 0 ? undefined : study.trials[i]?.params
  })
}
