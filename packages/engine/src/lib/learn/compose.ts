/**
 * Composing estimators, scikit-learn style: `pipeline` chains transformers and a final estimator (with
 * imbalanced-learn's resampling steps and an optional cache of fitted steps), `featureUnion` concatenates transformers'
 * outputs, `columnTransformer` applies different transformers to different columns, and `transformedTargetRegressor`
 * fits on transformed targets and maps predictions back.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { type Exposition, type PipelineStage, pipelineStages } from '../exposition/index.ts'
import { r2Score } from '../metrics/index.ts'
import { foldIn } from '../random/index.ts'
import { dict, type Space } from '../space/index.ts'
import { asTensor, fromStorage, reshape, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../trace/index.ts'
import { innerFitOptions, inputNames, keyFor, runOptionsFor } from './_shared.ts'
import { parametersOf, rowTargetsOf, STAGE_ROWS, stageDataOf, surfaceAndLossOf } from './_stages.ts'
import {
  type Estimator,
  type EstimatorKind,
  type FitOptions,
  type Fitted,
  isEstimator,
  type LearnData,
  requireTargets,
  requireVerb,
  type Verbs,
} from './estimator.ts'
import { classNamesOf, featureNamesOf } from './explained.ts'
import { isResampler, type Resampler } from './preprocessing/resampling.ts'
import { fromView, hstackViews, type MatrixView, matrixView, takeColumns, targetVector } from './validation.ts'

/** A pipeline step: an estimator, a resampler (applied only in fit), or `'passthrough'`. */
export type Step = Estimator | Resampler | 'passthrough'

/** A named step. */
export type NamedStep = readonly [string, Step]

/**
 * A cache of fitted transformers, keyed by the estimator, its hyperparameters and a fingerprint of the data
 * (sklearn's pipeline `memory`): refitting a pipeline whose early steps and data are unchanged reuses them, as in a
 * grid search over the final step.
 */
export interface FitCache {
  get(key: string): Fitted | undefined
  set(key: string, fitted: Fitted): void
  readonly hits: () => number
  readonly size: () => number
}

/**
 * An in-memory {@link FitCache}.
 *
 * @param capacity - The most entries kept (oldest dropped first; default 256).
 * @returns The cache.
 * @example
 * ```ts
 * import { fitCache } from 'mdml-engine/learn'
 * const cache = fitCache()
 * cache.size() // => 0
 * ```
 */
export function fitCache(capacity = 256): FitCache {
  const store = new Map<string, Fitted>()
  let hits = 0
  return {
    get: (key) => {
      const f = store.get(key)
      if (f !== undefined) hits++
      return f
    },
    set: (key, fitted) => {
      store.set(key, fitted)
      if (store.size > capacity) {
        const first = store.keys().next()
        if (first.done !== true) store.delete(first.value)
      }
    },
    hits: () => hits,
    size: () => store.size,
  }
}

/** FNV-1a over the bytes of numbers: a fingerprint of data for cache keys. */
function fingerprint(values: ArrayLike<number>): string {
  let h = 0x811c9dc5
  const buf = new Float64Array(1)
  const bytes = new Uint8Array(buf.buffer)
  for (let i = 0; i < values.length; i++) {
    buf[0] = values[i] ?? 0
    for (const b of bytes) {
      h ^= b
      h = Math.imul(h, 0x01000193) >>> 0
    }
  }
  return `${values.length}:${h.toString(16)}`
}

function describeHyper(value: unknown): Json {
  if (isEstimator(value)) return { estimator: value.name, hyper: describeHyper(value.hyper) }
  if (typeof value === 'function') return `fn:${value.name}`
  if (Array.isArray(value)) return value.map(describeHyper)
  if (typeof value === 'object' && value !== null) {
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, describeHyper(v)]))
  }
  if (typeof value === 'number' && !Number.isFinite(value)) return String(value)
  return (value ?? null) as Json
}

function cacheKey(est: Estimator, data: LearnData): string {
  const x = matrixView(data.x, 'cache', { allowNaN: true, allowInfinite: true })
  const y = data.y === undefined ? '' : fingerprint(toFlat(asTensor(data.y)))
  return JSON.stringify([est.name, describeHyper(est.hyper), fingerprint(x.data), x.d, y])
}

/** Route `step.param` and `{ step: { param } }` changes to the named estimators. */
function routeChanges(steps: readonly NamedStep[], changes: Readonly<Record<string, unknown>>): NamedStep[] {
  const out = steps.map(([n, s]) => [n, s] as [string, Step])
  for (const [k, v] of Object.entries(changes)) {
    const dot = k.indexOf('.')
    const name = dot < 0 ? k : k.slice(0, dot)
    const entry = out.find(([n]) => n === name)
    if (entry === undefined) throw new MdmlError('value', `no step named ${name}`)
    if (dot < 0 && (isEstimator(v) || isResampler(v) || v === 'passthrough')) entry[1] = v as Step
    else {
      const step = entry[1]
      if (!isEstimator(step)) throw new MdmlError('value', `step ${name} has no hyperparameters`)
      entry[1] = step.with(dot < 0 ? (v as Readonly<Record<string, unknown>>) : { [k.slice(dot + 1)]: v })
    }
  }
  return out
}

function stepSpace(steps: readonly NamedStep[]): Space | undefined {
  const spaces = Object.fromEntries(
    steps.flatMap(([n, s]) => (isEstimator(s) && s.hyperSpace !== undefined ? [[n, s.hyperSpace]] : [])),
  )
  return Object.keys(spaces).length === 0 ? undefined : dict(spaces)
}

/** Hyperparameters of {@link pipeline}. */
export interface PipelineHyper {
  readonly steps: readonly NamedStep[]
  readonly cache?: FitCache
}

/** Options of {@link pipeline}. */
export interface PipelineOptions {
  /** A cache of fitted transformers. */
  readonly cache?: FitCache
  /** The most training rows each stage's recorded data carries (evenly spaced beyond; default 500, 0 for none). */
  readonly explainRows?: number
}

/** A fitted step of a pipeline as data. */
export type FittedStepParams = { readonly name: string; readonly estimator: string | null; readonly params: Json }

/** A fitted pipeline: the last step's verbs, plus the fitted steps and their intermediate outputs. */
export interface FittedPipeline extends Fitted<{ readonly steps: FittedStepParams[] }> {
  /**
   * What each step did on the training data (a `pipelineStages` record: shapes, samples, each stage's output rows
   * with their classes and column statistics, what it learned and, for a final estimator on one or two raw
   * features, the whole pipeline's surface over them with its training loss), then the steps' own records; empty
   * after `revive`.
   */
  readonly exposition: readonly Exposition[]
  /** The fitted steps by name (resamplers and passthroughs are absent). */
  readonly namedSteps: Readonly<Record<string, Fitted>>
  /** The output of every transforming step on new data (step inspection), by step name. */
  readonly intermediate: (x: TensorLike) => Readonly<Record<string, Tensor>>
}

function fitStep(
  step: Estimator,
  data: LearnData,
  options: FitOptions,
  last: boolean,
  cache: FitCache | undefined,
): { fitted: Fitted; out?: Tensor } {
  if (last) return { fitted: step.fit(data, options) }
  const key = cache === undefined ? undefined : cacheKey(step, data)
  const hit = key === undefined ? undefined : cache?.get(key)
  if (hit !== undefined) return { fitted: hit, out: requireVerb(hit, 'transform', 'pipeline')(data.x) }
  const both = step.fitTransform?.(data, options)
  const fitted = both?.fitted ?? step.fit(data, options)
  if (key !== undefined) cache?.set(key, fitted)
  return { fitted, out: both?.transformed ?? requireVerb(fitted, 'transform', 'pipeline')(data.x) }
}

interface PipelineState {
  readonly index: number
  readonly data: LearnData
  readonly fitted: readonly (Fitted | undefined)[]
  /** What each step did, for the exposition (unless the fit asked for none). */
  readonly stages: readonly PipelineStage[]
}

const SAMPLE_ROWS = 4

/** The first rows of a matrix (all its columns), as plain numbers. */
function sampleOf(x: TensorLike): { readonly rows: number[][]; readonly shape: [number, number] } {
  const m = matrixView(x, 'pipeline', { allowNaN: true, allowInfinite: true })
  const rows = Array.from({ length: Math.min(SAMPLE_ROWS, m.n) }, (_, i) =>
    Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)),
  )
  return { rows, shape: [m.n, m.d] }
}

/** A fitted step's parameters worth showing: up to four numbers or short number lists. */
function learnedOf(f: Fitted): PipelineStage['learned'] {
  const out: { name: string; value: number | string | readonly number[] }[] = []
  const params = (f.params ?? {}) as Readonly<Record<string, unknown>>
  if (typeof params !== 'object' || params === null) return out
  for (const [name, v] of Object.entries(params)) {
    if (out.length >= 4) break
    if (typeof v === 'number' && Number.isFinite(v)) out.push({ name, value: v })
    else if (typeof v === 'string' && v.length <= 24) out.push({ name, value: v })
    else if (Array.isArray(v) && v.length > 0 && v.every((u) => typeof u === 'number')) out.push({ name, value: v })
    else if (Array.isArray(v) && Array.isArray(v[0]) && (v[0] as unknown[]).every((u) => typeof u === 'number'))
      out.push({ name: `${name}[0]`, value: v[0] as number[] })
  }
  return out
}

/** The column names after a step: its own names for its output when it has them, else x0, x1, ... */
function namesAfter(f: Fitted | undefined, input: readonly string[], d: number): string[] {
  if (f?.featureNamesOut !== undefined) {
    try {
      const names = f.featureNamesOut(input)
      if (names.length === d) return names
    } catch {
      // A step that cannot name its outputs keeps the generic names.
    }
  }
  return Array.from({ length: d }, (_, j) => `x${j}`)
}

/**
 * A chain of steps fitted in order (sklearn.pipeline.Pipeline, with imbalanced-learn's resampler steps): each
 * intermediate step is fitted on the output of the previous ones and transforms for the next (resamplers change the
 * training rows and are skipped in prediction); the last step gives the pipeline its verbs. Fitting is a traced
 * algorithm with one step per pipeline step (intermediate outputs recorded as `activation` tensors when the policy
 * asks, each estimator's own fit nested as a child run), and an optional {@link FitCache} reuses fitted transformers.
 * Hyperparameters route by step: `with({ 'model.alpha': 2 })` or `with({ model: { alpha: 2 } })`.
 *
 * @param steps - Named steps.
 * @param options - A fit cache, and how many training rows each stage's recorded data carries.
 * @returns The estimator.
 * @throws MdmlError with code `value` for duplicate names or a non-transformer before the last step.
 * @concept pipelines
 * @index estimator Chain of steps fitted in order, like sklearn Pipeline.
 * @example
 * ```ts
 * import { pipeline } from 'mdml-engine/learn'
 * import { polynomialFeatures, standardScaler } from 'mdml-engine/learn/preprocessing'
 * import { toArray } from 'mdml-engine/tensor'
 * const p = pipeline([['poly', polynomialFeatures({ includeBias: false })], ['scale', standardScaler()]])
 * const f = p.fit({ x: [[0], [1], [2]] })
 * toArray(f.transform([[1]])) // => [[0, -0.3922322702763681]]
 * f.featureNamesOut() // => ['x0', 'x0^2']
 * p.with({ 'scale.withMean': false }).hyper.steps[1][1].hyper.withMean // => false
 * ```
 */
export function pipeline(
  steps: readonly NamedStep[],
  options: PipelineOptions = {},
): Estimator<PipelineHyper, FittedPipeline> {
  if (steps.length === 0) throw new MdmlError('value', 'pipeline: needs at least one step')
  if (new Set(steps.map(([n]) => n)).size !== steps.length)
    throw new MdmlError('value', 'pipeline: step names must be unique')
  const last = steps[steps.length - 1]?.[1]
  const kind: EstimatorKind = isEstimator(last) ? last.kind : 'transformer'
  const space = stepSpace(steps)
  const hyper: PipelineHyper = { steps, ...(options.cache === undefined ? {} : { cache: options.cache }) }

  const build = (fitted: readonly (Fitted | undefined)[], exposition: readonly Exposition[] = []): FittedPipeline => {
    const finalFitted = fitted[fitted.length - 1]
    const forward = (x: TensorLike, upTo: number): Tensor => {
      let cur: Tensor = fromView(matrixView(x, 'pipeline', { allowNaN: true, allowInfinite: true }))
      for (let i = 0; i < upTo; i++) {
        const f = fitted[i]
        if (f !== undefined) cur = requireVerb(f, 'transform', 'pipeline')(cur)
      }
      return cur
    }
    const n = steps.length
    const verbs: Record<string, unknown> = {}
    for (const verb of ['predict', 'predictProba', 'decisionFunction', 'predictDistribution'] as const) {
      const fn = finalFitted?.[verb]
      if (typeof fn === 'function')
        verbs[verb] = (x: TensorLike) => (fn as (x: TensorLike) => unknown)(forward(x, n - 1))
    }
    if (finalFitted === undefined || typeof finalFitted.transform === 'function') {
      verbs['transform'] = (x: TensorLike) => forward(x, n)
    }
    const invertible = fitted.every((f) => f === undefined || typeof f.inverseTransform === 'function')
    if (invertible && (finalFitted === undefined || typeof finalFitted.transform === 'function')) {
      verbs['inverseTransform'] = (x: TensorLike) => {
        let cur: Tensor = fromView(matrixView(x, 'pipeline', { allowNaN: true, allowInfinite: true }))
        for (let i = n - 1; i >= 0; i--) {
          const f = fitted[i]
          if (f !== undefined) cur = requireVerb(f, 'inverseTransform', 'pipeline')(cur)
        }
        return cur
      }
    }
    if (typeof finalFitted?.score === 'function') {
      const s = finalFitted.score
      verbs['score'] = (data: LearnData) => s({ ...data, x: forward(data.x, n - 1) })
    }
    const namedSteps = Object.fromEntries(
      steps.flatMap(([name], i) => (fitted[i] === undefined ? [] : [[name, fitted[i] as Fitted]])),
    )
    const first = fitted.find((f) => f !== undefined)
    return {
      ...(verbs as Verbs),
      estimator: 'pipeline',
      kind,
      params: {
        steps: steps.map(([name, s], i) => ({
          name,
          estimator: isEstimator(s) ? s.name : null,
          params: fitted[i]?.params ?? null,
        })),
      },
      ...(first?.nFeaturesIn === undefined ? {} : { nFeaturesIn: first.nFeaturesIn }),
      ...(finalFitted?.classes === undefined ? {} : { classes: finalFitted.classes }),
      ...(finalFitted?.featureImportances === undefined ? {} : { featureImportances: finalFitted.featureImportances }),
      featureNamesOut: (input?: readonly string[]) => {
        let names = input === undefined ? undefined : [...input]
        for (const f of fitted) {
          if (f === undefined) continue
          if (f.featureNamesOut === undefined) {
            throw new MdmlError('value', `pipeline: ${f.estimator} does not name its output features`)
          }
          names = f.featureNamesOut(names)
        }
        return names ?? inputNames(first?.nFeaturesIn ?? 0, undefined, 'pipeline')
      },
      namedSteps,
      exposition,
      intermediate: (x) => {
        const out: Record<string, Tensor> = {}
        let cur: Tensor = fromView(matrixView(x, 'pipeline', { allowNaN: true, allowInfinite: true }))
        steps.forEach(([name], i) => {
          const f = fitted[i]
          if (f !== undefined && typeof f.transform === 'function') {
            cur = f.transform(cur)
            out[name] = cur
          }
        })
        return out
      },
    } as FittedPipeline
  }

  return {
    name: 'pipeline',
    kind,
    hyper,
    ...(space === undefined ? {} : { hyperSpace: space }),
    fit: (data, fitOptions = {}) => {
      const explain = fitOptions.explain !== false
      const rowsLimit = Math.max(0, Math.floor(options.explainRows ?? STAGE_ROWS))
      const targets = explain ? rowTargetsOf(data, kind) : undefined
      const algorithm: Algorithm<PipelineState> = {
        name: 'pipeline',
        config: { steps: steps.map(([n, s]) => [n, isEstimator(s) ? s.name : isResampler(s) ? s.name : s]) },
        schema: channels({
          metrics: { step: { description: 'the index of the step just fitted' } },
          tensors: {
            output: { role: 'activation', description: 'the output of the step just fitted (training data)' },
          },
          messages: { step: { description: 'a step was fitted', payload: { name: 'string', estimator: 'string' } } },
        }),
        init: () => ({ index: 0, data, fitted: [], stages: [] }),
        step: (s, ctx) => {
          const [name, step] = steps[s.index] as NamedStep
          const isLast = s.index === steps.length - 1
          const inner = innerFitOptions(fitOptions, foldIn(ctx.key, 0), ctx.run, true)
          // The step's record: its input and output (shapes, first rows, column names) and what it learned.
          const stageOf = (
            kind: PipelineStage['kind'],
            output: TensorLike,
            outputColumns: (input: readonly string[], d: number) => string[],
            learned: PipelineStage['learned'],
            full?: { readonly fitted?: Fitted; readonly y?: TensorLike },
          ): PipelineStage[] => {
            if (!explain) return []
            const before = sampleOf(s.data.x)
            const after = sampleOf(output)
            const inputColumns = s.stages.at(-1)?.outputColumns ?? featureNamesOf(data, before.shape[1])
            const columns = outputColumns(inputColumns, after.shape[1])
            const fittedStep = full?.fitted
            const detail = fittedStep === undefined ? undefined : parametersOf(fittedStep, inputColumns, columns)
            return [
              ...s.stages,
              {
                name,
                kind,
                inputColumns,
                outputColumns: columns,
                inputSample: before.rows,
                outputSample: after.rows,
                inputShape: before.shape,
                outputShape: after.shape,
                learned,
                ...(rowsLimit === 0
                  ? {}
                  : { output: stageDataOf(output, columns, full?.y ?? s.data.y, targets, rowsLimit) }),
                ...(detail === undefined || detail.parameters.length === 0 ? {} : { parameters: detail.parameters }),
                ...(detail?.explainedVarianceRatio === undefined
                  ? {}
                  : { explainedVarianceRatio: detail.explainedVarianceRatio.slice(0, after.shape[1]) }),
              },
            ]
          }
          if (step === 'passthrough')
            return {
              index: s.index + 1,
              data: s.data,
              fitted: [...s.fitted, undefined],
              stages: stageOf('transformer', s.data.x, (i) => [...i], []),
            }
          if (isResampler(step)) {
            if (isLast) throw new MdmlError('value', 'pipeline: a resampler cannot be the last step')
            const r = step.fitResample(s.data, inner)
            return {
              index: s.index + 1,
              data: { x: r.x, y: r.y },
              fitted: [...s.fitted, undefined],
              stages: stageOf('resampler', r.x, (i) => [...i], [], r.y === undefined ? {} : { y: r.y }),
            }
          }
          const { fitted, out } = fitStep(step, s.data, inner, isLast, hyper.cache)
          let stages = s.stages
          if (explain && out !== undefined)
            stages = stageOf('transformer', out, (i, d) => namesAfter(fitted, i, d), learnedOf(fitted), { fitted })
          else if (explain) {
            // The final estimator: its probabilities (or predictions) on the training rows, and its training score.
            const proba = fitted.predictProba
            const output = proba !== undefined ? proba(s.data.x) : fitted.predict?.(s.data.x)
            if (output !== undefined) {
              const columnsOut = (_: readonly string[], d: number): string[] =>
                proba !== undefined && fitted.classes !== undefined
                  ? classNamesOf(data, fitted.classes).map((c) => `p(${c})`)
                  : Array.from({ length: d }, (_, j) => (d === 1 ? 'prediction' : `y${j}`))
              const shaped = output.shape.length === 1 ? reshape(output, [output.shape[0] ?? 0, 1]) : output
              const score = s.data.y === undefined ? undefined : fitted.score?.(s.data)
              stages = stageOf(
                'estimator',
                shaped as Tensor,
                columnsOut,
                [
                  ...(score === undefined ? [] : [{ name: 'training score', value: score }]),
                  ...learnedOf(fitted).slice(0, 3),
                ],
                { fitted },
              )
            }
          }
          return {
            index: s.index + 1,
            data: out === undefined ? s.data : { ...s.data, x: out },
            fitted: [...s.fitted, fitted],
            stages,
          }
        },
        done: (s) => (s.index >= steps.length ? 'completed' : undefined),
        observe: (s) => ({
          metrics: { step: s.index },
          ...(s.index > 0 ? { tensors: { output: { role: 'activation' as const, value: s.data.x as Tensor } } } : {}),
          messages:
            s.index > 0
              ? [
                  {
                    kind: 'step',
                    payload: {
                      name: steps[s.index - 1]?.[0] ?? '',
                      estimator: s.fitted[s.index - 1]?.estimator ?? 'none',
                    },
                  },
                ]
              : [],
        }),
      }
      const result = runAlgorithm(algorithm, { ...runOptionsFor(fitOptions, 'pipeline'), maxSteps: steps.length })
      const own = result.state.fitted.flatMap(
        (f) => (f as { readonly exposition?: readonly Exposition[] } | undefined)?.exposition ?? [],
      )
      if (!explain) return build(result.state.fitted)
      let stages = result.state.stages
      const lastStage = stages.at(-1)
      if (lastStage?.kind === 'estimator' && rowsLimit > 0) {
        // The whole pipeline over its raw inputs: the surface the final estimator draws, and its loss.
        const extra = surfaceAndLossOf(build(result.state.fitted), data, targets, rowsLimit)
        stages = [...stages.slice(0, -1), { ...lastStage, ...extra }]
      }
      const firstShape = stages[0]?.inputShape
      const record = pipelineStages({
        stages,
        ...(rowsLimit === 0 || firstShape === undefined
          ? {}
          : {
              input: stageDataOf(data.x, stages[0]?.inputColumns ?? [], data.y, targets, rowsLimit),
            }),
        ...(targets?.kind === 'classes' ? { classes: targets.names } : {}),
      })
      return build(result.state.fitted, [record, ...own])
    },
    revive: (params) => {
      const p = params as { readonly steps?: readonly FittedStepParams[] }
      return build(
        steps.map(([, s], i) => {
          const sp = p.steps?.[i]
          return isEstimator(s) && sp !== undefined && sp.params !== null ? s.revive(sp.params) : undefined
        }),
      )
    },
    with: (changes) => pipeline(routeChanges(steps, changes), options),
  }
}

/**
 * A pipeline from unnamed steps, named after their estimators (sklearn.pipeline.make_pipeline).
 *
 * @param steps - The steps.
 * @returns The pipeline.
 * @example
 * ```ts
 * import { makePipeline } from 'mdml-engine/learn'
 * import { standardScaler, minMaxScaler } from 'mdml-engine/learn/preprocessing'
 * makePipeline(standardScaler(), minMaxScaler()).hyper.steps.map(([n]) => n) // => ['standard-scaler', 'min-max-scaler']
 * ```
 */
export function makePipeline(...steps: readonly Step[]): Estimator<PipelineHyper, FittedPipeline> {
  const seen = new Map<string, number>()
  return pipeline(
    steps.map((s) => {
      const base = typeof s === 'string' ? s : s.name
      const k = (seen.get(base) ?? 0) + 1
      seen.set(base, k)
      return [k === 1 ? base : `${base}-${k}`, s] as const
    }),
  )
}

/** Hyperparameters of {@link featureUnion}. */
export interface FeatureUnionHyper {
  readonly transformers: readonly (readonly [string, Estimator | 'drop'])[]
  /** Multiply each transformer's output by a weight. */
  readonly weights?: Readonly<Record<string, number>>
}

/** A fitted union or column transformer. */
export interface FittedComposite extends Fitted<{ readonly parts: FittedStepParams[]; readonly nFeatures: number }> {
  readonly transform: (x: TensorLike) => Tensor
  readonly namedTransformers: Readonly<Record<string, Fitted>>
  readonly featureNamesOut: (input?: readonly string[]) => string[]
  readonly nFeaturesIn: number
}

function scaled(m: MatrixView, w: number | undefined): MatrixView {
  return w === undefined || w === 1 ? m : { ...m, data: Float64Array.from(m.data, (v) => v * w) }
}

/**
 * Concatenate the outputs of several transformers fitted on the same input (sklearn.pipeline.FeatureUnion), with
 * optional weights; output feature names are prefixed with the transformer's name.
 *
 * @param transformers - Named transformers (`'drop'` to disable one).
 * @param options - Weights by name.
 * @returns The estimator.
 * @example
 * ```ts
 * import { featureUnion } from 'mdml-engine/learn'
 * import { minMaxScaler, standardScaler } from 'mdml-engine/learn/preprocessing'
 * import { toArray } from 'mdml-engine/tensor'
 * const u = featureUnion([['z', standardScaler()], ['m', minMaxScaler()]]).fit({ x: [[0], [2]] })
 * toArray(u.transform([[2]])) // => [[1, 1]]
 * u.featureNamesOut() // => ['z__x0', 'm__x0']
 * ```
 */
export function featureUnion(
  transformers: FeatureUnionHyper['transformers'],
  options: { readonly weights?: Readonly<Record<string, number>> } = {},
): Estimator<FeatureUnionHyper, FittedComposite> {
  const hyper: FeatureUnionHyper = {
    transformers,
    ...(options.weights === undefined ? {} : { weights: options.weights }),
  }
  const build = (fitted: readonly (Fitted | undefined)[], d: number): FittedComposite => {
    const parts = transformers.map(([name], i) => ({ name, fitted: fitted[i] }))
    return {
      estimator: 'feature-union',
      kind: 'transformer',
      params: {
        parts: parts.map((p, i) => ({
          name: p.name,
          estimator: fitted[i]?.estimator ?? null,
          params: fitted[i]?.params ?? null,
        })),
        nFeatures: d,
      },
      nFeaturesIn: d,
      namedTransformers: Object.fromEntries(parts.flatMap((p) => (p.fitted === undefined ? [] : [[p.name, p.fitted]]))),
      transform: (x) => {
        const m = matrixView(x, 'featureUnion', { allowNaN: true, nFeatures: d })
        const outs = parts.flatMap((p) =>
          p.fitted === undefined
            ? []
            : [
                scaled(
                  matrixView(requireVerb(p.fitted, 'transform', 'featureUnion')(fromView(m)), 'featureUnion', {
                    allowNaN: true,
                  }),
                  options.weights?.[p.name],
                ),
              ],
        )
        return fromView(outs.length === 0 ? { n: m.n, d: 0, data: new Float64Array(0) } : hstackViews(outs))
      },
      featureNamesOut: (input) =>
        parts.flatMap((p) => {
          if (p.fitted === undefined) return []
          const names = p.fitted.featureNamesOut?.(input)
          if (names === undefined)
            throw new MdmlError('value', `featureUnion: ${p.fitted.estimator} does not name its outputs`)
          return names.map((nm) => `${p.name}__${nm}`)
        }),
    }
  }
  return {
    name: 'feature-union',
    kind: 'transformer',
    hyper,
    fit: (data, fitOptions = {}) => {
      const d = matrixView(data.x, 'featureUnion', { allowNaN: true }).d
      const key = keyFor(fitOptions, 'featureUnion')
      return build(
        transformers.map(([, t], i) =>
          t === 'drop' ? undefined : t.fit(data, innerFitOptions(fitOptions, foldIn(key, i), fitOptions.parent)),
        ),
        d,
      )
    },
    revive: (params) => {
      const p = params as FittedComposite['params']
      const fitted = transformers.map(([, t], i) => {
        const sp = p.parts[i]
        return t === 'drop' || sp === undefined || sp.params === null ? undefined : t.revive(sp.params)
      })
      return build(fitted, p.nFeatures)
    },
    with: (changes) => {
      const routed = routeChanges(transformers as readonly NamedStep[], changes)
      return featureUnion(
        routed.map(([n, s]) => [n, s === 'passthrough' || isResampler(s) ? 'drop' : s] as const),
        options,
      )
    },
  }
}

/** A column transformer's entry: a name, a transformer (or `'drop'`/`'passthrough'`) and its column indices. */
export type ColumnSpec = readonly [string, Estimator | 'drop' | 'passthrough', readonly number[]]

/** Hyperparameters of {@link columnTransformer}. */
export interface ColumnTransformerHyper {
  readonly transformers: readonly ColumnSpec[]
  /** What happens to the columns no transformer selects (default `'drop'`). */
  readonly remainder: Estimator | 'drop' | 'passthrough'
  /** Prefix output names with the transformer's name (default true). */
  readonly verboseFeatureNamesOut: boolean
}

/**
 * Apply different transformers to different columns and concatenate the results (sklearn.compose.ColumnTransformer):
 * outputs in the order of the transformers, then the remainder (the unselected columns, in order) dropped, passed
 * through or transformed; names prefixed `name__` (`remainder__` for the rest).
 *
 * @param transformers - Entries `[name, transformer, columns]`.
 * @param options - remainder and verboseFeatureNamesOut.
 * @returns The estimator.
 * @index estimator Apply different transformers to different columns.
 * @example
 * ```ts
 * import { columnTransformer } from 'mdml-engine/learn'
 * import { oneHotEncoder, standardScaler } from 'mdml-engine/learn/preprocessing'
 * import { toArray } from 'mdml-engine/tensor'
 * const ct = columnTransformer([['num', standardScaler(), [0]], ['cat', oneHotEncoder(), [1]]], { remainder: 'passthrough' })
 * const f = ct.fit({ x: [[0, 1, 7], [2, 2, 8]] })
 * toArray(f.transform([[2, 1, 9]])) // => [[1, 1, 0, 9]]
 * f.featureNamesOut() // => ['num__x0', 'cat__x1_1', 'cat__x1_2', 'remainder__x2']
 * ```
 */
export function columnTransformer(
  transformers: readonly ColumnSpec[],
  options: Partial<Omit<ColumnTransformerHyper, 'transformers'>> = {},
): Estimator<ColumnTransformerHyper, FittedComposite> {
  const hyper: ColumnTransformerHyper = {
    transformers,
    remainder: options.remainder ?? 'drop',
    verboseFeatureNamesOut: options.verboseFeatureNamesOut ?? true,
  }
  if (new Set(transformers.map(([n]) => n)).size !== transformers.length) {
    throw new MdmlError('value', 'columnTransformer: names must be unique')
  }
  const plan = (d: number): ColumnSpec[] => {
    const used = new Set(transformers.flatMap(([, , cols]) => cols))
    for (const c of used)
      if (!(Number.isInteger(c) && c >= 0 && c < d))
        throw new MdmlError('value', `columnTransformer: column ${c} out of range`)
    const rest = Array.from({ length: d }, (_, j) => j).filter((j) => !used.has(j))
    return [...transformers, ...(rest.length > 0 ? [['remainder', hyper.remainder, rest] as ColumnSpec] : [])]
  }
  const build = (fitted: readonly (Fitted | undefined)[], d: number): FittedComposite => {
    const entries = plan(d)
    return {
      estimator: 'column-transformer',
      kind: 'transformer',
      params: {
        parts: entries.map(([name], i) => ({
          name,
          estimator: fitted[i]?.estimator ?? null,
          params: fitted[i]?.params ?? null,
        })),
        nFeatures: d,
      },
      nFeaturesIn: d,
      namedTransformers: Object.fromEntries(
        entries.flatMap(([name], i) => (fitted[i] === undefined ? [] : [[name, fitted[i] as Fitted]])),
      ),
      transform: (x) => {
        const m = matrixView(x, 'columnTransformer', { allowNaN: true, nFeatures: d })
        const outs = entries.flatMap(([, t, cols], i) => {
          if (t === 'drop') return []
          const sub = takeColumns(m, cols)
          const f = fitted[i]
          if (t === 'passthrough' || f === undefined) return [sub]
          return [
            matrixView(requireVerb(f, 'transform', 'columnTransformer')(fromView(sub)), 'columnTransformer', {
              allowNaN: true,
            }),
          ]
        })
        return fromView(outs.length === 0 ? { n: m.n, d: 0, data: new Float64Array(0) } : hstackViews(outs))
      },
      featureNamesOut: (input) => {
        const names = inputNames(d, input, 'columnTransformer')
        return entries.flatMap(([name, t, cols], i) => {
          if (t === 'drop') return []
          const sub = cols.map((c) => names[c] ?? '')
          const f = fitted[i]
          const out = t === 'passthrough' || f === undefined ? sub : f.featureNamesOut?.(sub)
          if (out === undefined) throw new MdmlError('value', `columnTransformer: ${name} does not name its outputs`)
          return hyper.verboseFeatureNamesOut ? out.map((nm) => `${name}__${nm}`) : out
        })
      },
    }
  }
  return {
    name: 'column-transformer',
    kind: 'transformer',
    hyper,
    fit: (data, fitOptions = {}) => {
      const m = matrixView(data.x, 'columnTransformer', { allowNaN: true })
      const key = keyFor(fitOptions, 'columnTransformer')
      return build(
        plan(m.d).map(([, t, cols], i) =>
          t === 'drop' || t === 'passthrough'
            ? undefined
            : t.fit(
                { ...data, x: fromView(takeColumns(m, cols)) },
                innerFitOptions(fitOptions, foldIn(key, i), fitOptions.parent),
              ),
        ),
        m.d,
      )
    },
    revive: (params) => {
      const p = params as FittedComposite['params']
      return build(
        plan(p.nFeatures).map(([, t], i) => {
          const sp = p.parts[i]
          return t === 'drop' || t === 'passthrough' || sp === undefined || sp.params === null
            ? undefined
            : t.revive(sp.params)
        }),
        p.nFeatures,
      )
    },
    with: (changes) => {
      const routed = routeChanges(
        transformers.map(([n, t]) => [n, t] as NamedStep),
        changes,
      )
      return columnTransformer(
        transformers.map(([n, , cols], i) => [n, routed[i]?.[1] as Estimator | 'drop' | 'passthrough', cols] as const),
        options,
      )
    },
  }
}

/** Hyperparameters of {@link transformedTargetRegressor}. */
export interface TransformedTargetHyper {
  readonly regressor: Estimator
  /** A transformer of the targets (as a one-column matrix). */
  readonly transformer?: Estimator
  /** Or a function and its inverse, applied elementwise. */
  readonly func?: (y: number) => number
  readonly inverseFunc?: (y: number) => number
}

/**
 * Fit a regressor on transformed targets and map its predictions back (sklearn.compose.TransformedTargetRegressor):
 * with a transformer (fitted on y as a column) or a function pair (checked to invert on the training targets).
 *
 * @param hyper - The regressor and the transformer or functions.
 * @returns The estimator.
 * @throws MdmlError with code `value` when the inverse does not invert.
 * @example
 * ```ts
 * import { transformedTargetRegressor, defineEstimator } from 'mdml-engine/learn'
 * import { tensor, toFlat } from 'mdml-engine/tensor'
 * const meanOfTargets = defineEstimator(
 *   {
 *     name: 'mean',
 *     kind: 'regressor',
 *     fit: (data) => ({ mean: Array.from(data.y).reduce((a, b) => a + b, 0) / data.y.length }),
 *     methods: (p) => ({ predict: (x) => tensor(x.map(() => p.mean)) }),
 *   },
 *   {},
 * )
 * const t = transformedTargetRegressor({ regressor: meanOfTargets, func: Math.log, inverseFunc: Math.exp })
 * toFlat(t.fit({ x: [[0], [0]], y: [1, 100] }).predict([[0]])) // => [10.000000000000002]
 * ```
 */
export function transformedTargetRegressor(
  hyper: TransformedTargetHyper,
): Estimator<TransformedTargetHyper, Fitted<{ readonly regressor: Json; readonly transformer: Json }>> {
  const forwardOf =
    (fittedT: Fitted | undefined) =>
    (y: Float64Array): Float64Array => {
      if (fittedT !== undefined) {
        return Float64Array.from(toFlat(requireVerb(fittedT, 'transform', 'transformedTargetRegressor')(columnOf(y))))
      }
      return hyper.func === undefined ? y : Float64Array.from(y, hyper.func)
    }
  const inverseOf =
    (fittedT: Fitted | undefined) =>
    (y: Float64Array): Float64Array => {
      if (fittedT !== undefined) {
        return Float64Array.from(
          toFlat(requireVerb(fittedT, 'inverseTransform', 'transformedTargetRegressor')(columnOf(y))),
        )
      }
      return hyper.inverseFunc === undefined ? y : Float64Array.from(y, hyper.inverseFunc)
    }
  if ((hyper.func === undefined) !== (hyper.inverseFunc === undefined)) {
    throw new MdmlError('value', 'transformedTargetRegressor: give both func and inverseFunc')
  }
  const build = (
    regressor: Fitted,
    transformer: Fitted | undefined,
  ): Fitted<{ readonly regressor: Json; readonly transformer: Json }> => {
    const inverse = inverseOf(transformer)
    const predict = (x: TensorLike): Tensor => {
      const p = Float64Array.from(toFlat(requireVerb(regressor, 'predict', 'transformedTargetRegressor')(x)))
      const back = inverse(p)
      return fromStorage(back, [back.length], 'float64')
    }
    return {
      estimator: 'transformed-target-regressor',
      kind: 'regressor',
      params: { regressor: regressor.params, transformer: transformer?.params ?? null },
      predict,
      score: (data) =>
        r2Score(
          requireTargets(data, 'score'),
          predict(data.x),
          data.weights === undefined ? {} : { sampleWeight: data.weights },
        ).estimate.value,
      ...(regressor.nFeaturesIn === undefined ? {} : { nFeaturesIn: regressor.nFeaturesIn }),
      ...(regressor.featureImportances === undefined ? {} : { featureImportances: regressor.featureImportances }),
    }
  }
  return {
    name: 'transformed-target-regressor',
    kind: 'regressor',
    hyper,
    ...(hyper.regressor.hyperSpace === undefined
      ? {}
      : { hyperSpace: dict({ regressor: hyper.regressor.hyperSpace }) }),
    fit: (data, fitOptions = {}) => {
      const y = targetVector(requireTargets(data, 'transformedTargetRegressor'), 'transformedTargetRegressor')
      const key = keyFor(fitOptions, 'transformedTargetRegressor')
      const transformer = hyper.transformer?.fit(
        { x: columnOf(y) },
        innerFitOptions(fitOptions, foldIn(key, 0), fitOptions.parent),
      )
      const yt = forwardOf(transformer)(y)
      const back = inverseOf(transformer)(yt)
      for (let i = 0; i < y.length; i++) {
        if (Math.abs((back[i] ?? 0) - (y[i] ?? 0)) > 1e-7 * (1 + Math.abs(y[i] ?? 0))) {
          throw new MdmlError('value', 'transformedTargetRegressor: the inverse does not invert the transform on y')
        }
      }
      const regressor = hyper.regressor.fit(
        { ...data, y: yt },
        innerFitOptions(fitOptions, foldIn(key, 1), fitOptions.parent),
      )
      return build(regressor, transformer)
    },
    revive: (params) => {
      const p = params as { readonly regressor: Json; readonly transformer: Json }
      return build(
        hyper.regressor.revive(p.regressor),
        p.transformer === null ? undefined : hyper.transformer?.revive(p.transformer),
      )
    },
    with: (changes) => {
      const routed = routeChanges(
        [
          ['regressor', hyper.regressor],
          ...(hyper.transformer === undefined ? [] : [['transformer', hyper.transformer] as NamedStep]),
        ],
        changes,
      )
      const regressor = routed[0]?.[1]
      const transformer = routed[1]?.[1]
      return transformedTargetRegressor({
        ...hyper,
        regressor: isEstimator(regressor) ? regressor : hyper.regressor,
        ...(isEstimator(transformer) ? { transformer } : {}),
      })
    },
  }
}

function columnOf(y: Float64Array): Tensor {
  return fromView({ n: y.length, d: 1, data: y })
}
