/**
 * Perturbation explanations of fitted models over images, texts and series, as feature-attribution records with a
 * layout the views draw (an attribution map, a token strip, a series): occlusion, feature ablation over superpixels,
 * words or windows, RISE, and Shapley values or LIME over those units; and Morris screening as an importance record.
 */
import { argmaxOf, WhiteboxError } from '../../lib/base/index.ts'
import * as explain from '../../lib/explain/index.ts'
import { type Explained, featureAttributions, importances } from '../../lib/exposition/index.ts'
import { type Fitted, featureNamesOf, type LearnData, matrixView } from '../../lib/learn/index.ts'
import { type Key, key as makeKey } from '../../lib/random/index.ts'
import { fromStorage, toFlat } from '../../lib/tensor/index.ts'
import { modelOutput } from './_models.ts'

/** The image explainers {@link explainImage} offers. */
export type ImageMethod = 'occlusion' | 'ablation' | 'rise' | 'shap' | 'lime'

/** Options of {@link explainImage}. */
export interface ExplainImageOptions {
  /** The explainer (default occlusion). */
  readonly method?: ImageMethod
  /** The image shape `[h, w]` (default square). */
  readonly shape?: readonly [number, number]
  /** Occlusion's window side and stride (default 2 and 1). */
  readonly window?: number
  readonly stride?: number
  /** Superpixels for ablation, SHAP and LIME: a grid of `cell`-sided squares (default) or SLIC segments. */
  readonly segments?: 'grid' | 'slic'
  /** The grid cell side (default 2) or the SLIC segments wanted (default 16). */
  readonly cell?: number
  readonly nSegments?: number
  /** RISE's masks (default 500) and its coarse grid (default 4). */
  readonly nMasks?: number
  readonly cells?: number
  /** Samples of Kernel SHAP's coalitions or LIME's perturbations (default 400). */
  readonly nSamples?: number
  /** What removed pixels become (default 0). */
  readonly baseline?: number
  /** The class explained (default the class predicted for the row). */
  readonly classIndex?: number
  readonly key?: Key
}

/** An image's attribution map. */
export interface ImageExplanation {
  readonly method: ImageMethod
  /** One value per pixel, row-major. */
  readonly values: readonly number[]
  readonly prediction: number
  /** The output with every pixel at the baseline. */
  readonly baseValue: number
}

function squareShape(d: number, given: readonly [number, number] | undefined, who: string): [number, number] {
  if (given !== undefined) {
    if (given[0] * given[1] !== d) throw new WhiteboxError('shape', `${who}: shape ${given.join('x')} for ${d} pixels`)
    return [given[0], given[1]]
  }
  const s = Math.round(Math.sqrt(d))
  if (s * s !== d) throw new WhiteboxError('shape', `${who}: ${d} pixels are not a square image; give shape`)
  return [s, s]
}

function predictedClass(fitted: Fitted, row: readonly number[]): number | undefined {
  if (fitted.kind !== 'classifier' || fitted.predictProba === undefined) return undefined
  return argmaxOf(toFlat(fitted.predictProba([row as number[]])))
}

/**
 * Explain a fitted model's prediction for one image (a row of flattened pixels) by perturbation: occlusion by a
 * sliding window, ablation of superpixels, RISE's random masks, Kernel SHAP over superpixels, or LIME over them. The
 * record carries the image layout, so its standard view is the input beside its attribution map.
 *
 * @param fitted - A fitted model on flattened images.
 * @param data - The images (rows of pixels).
 * @param row - The image explained.
 * @param options - method, shape, window and stride, segments, cell or nSegments, nMasks and cells, nSamples,
 *   baseline, classIndex and key.
 * @returns The attribution map with a `featureAttributions` record.
 */
export function explainImage(
  fitted: Fitted,
  data: LearnData,
  row: number,
  options: ExplainImageOptions = {},
): Explained<ImageExplanation> {
  const x = matrixView(data.x, 'explainImage')
  if (!(Number.isInteger(row) && row >= 0 && row < x.n)) throw new WhiteboxError('value', `explainImage: no row ${row}`)
  const shape = squareShape(x.d, options.shape, 'explainImage')
  const pixels = Array.from(x.data.subarray(row * x.d, (row + 1) * x.d))
  const output = modelOutput(fitted, data, 'auto', options.classIndex ?? predictedClass(fitted, pixels))
  const predict = output.predict
  const method = options.method ?? 'occlusion'
  const baseline = options.baseline ?? 0
  const key = options.key ?? makeKey(0)
  const segments =
    options.segments === 'slic'
      ? explain.slicSuperpixels(pixels, shape, { nSegments: options.nSegments ?? 16 })
      : explain.gridSuperpixels(shape, options.cell ?? 2)
  const masked = (): ReturnType<typeof explain.maskedPredictor> =>
    explain.maskedPredictor(predict, pixels, segments, baseline)
  let values: number[]
  if (method === 'occlusion') {
    const w = options.window ?? 2
    const st = options.stride ?? 1
    values = explain.occlusion(predict, [pixels], { shape, window: [w, w], strides: [st, st], baseline })[0] ?? []
  } else if (method === 'ablation') {
    values = explain.featureAblation(predict, [pixels], { featureMask: segments, baseline })[0] ?? []
  } else if (method === 'rise') {
    values =
      explain.rise(predict, [pixels], {
        shape,
        nMasks: options.nMasks ?? 500,
        cells: options.cells ?? 4,
        baseline,
        key,
      })[0] ?? []
  } else if (method === 'shap') {
    const { predict: f, groups } = masked()
    const G = groups.length
    const r = explain.kernelShap(f, [new Array<number>(G).fill(1)], [new Array<number>(G).fill(0)], {
      nSamples: options.nSamples ?? 400,
      key,
    })
    values = explain.expandGroups(r.values[0] ?? [], groups, x.d, { share: true })
  } else {
    const { predict: f, groups } = masked()
    const r = explain.limeMasked(f, groups.length, {
      nSamples: options.nSamples ?? 400,
      kernelWidth: 0.25,
      distanceScale: 1,
      nFeatures: groups.length,
      featureSelection: 'none',
      key,
    })
    const w = new Array<number>(groups.length).fill(0)
    for (const [j, v] of r.weights) w[j] = v
    values = explain.expandGroups(w, groups, x.d)
  }
  const at = (rows: number[][]): number[] =>
    toFlat(predict(fromStorage(Float64Array.from(rows.flat()), [rows.length, x.d], 'float64')))
  const [prediction = 0, baseValue = 0] = at([pixels, new Array<number>(x.d).fill(baseline)])
  const features = featureNamesOf(data, x.d)
  return {
    method,
    values,
    prediction,
    baseValue,
    exposition: [
      featureAttributions({
        method,
        features,
        output: output.name,
        baseValue,
        values: [values],
        data: [pixels],
        predictions: [prediction],
        indices: [row],
        layout: { kind: 'image', shape },
        ...(method === 'occlusion' || method === 'rise' ? {} : { segments }),
      }),
    ],
  }
}

/** The text explainers {@link explainText} offers. */
export type TextMethod = 'ablation' | 'shap' | 'lime'

/** Options of {@link explainText}. */
export interface ExplainTextOptions {
  readonly method?: TextMethod
  /** What the scores are (default `score`). */
  readonly output?: string
  /** A token absent words become (default: removed). */
  readonly mask?: string
  /** Kernel SHAP's coalitions or LIME's perturbations (default 500). */
  readonly nSamples?: number
  readonly key?: Key
}

/** A text's attributions, token by token. */
export interface TextExplanation {
  readonly tokens: readonly string[]
  readonly values: readonly number[]
  readonly prediction: number
  /** The score of the empty (or fully masked) text. */
  readonly baseValue: number
}

/**
 * Explain a text classifier's score for one text, word by word: ablation of each word, Kernel SHAP over the words
 * (exact for short texts), or LIME's text explainer. The model is any function scoring texts; the record lays the
 * words out as a token strip.
 *
 * @param classify - The model: one score per text (a probability, a logit).
 * @param text - The text explained.
 * @param options - method, output, mask, nSamples and key.
 * @returns The tokens and their attributions with a `featureAttributions` record.
 */
export function explainText(
  classify: (texts: readonly string[]) => readonly number[],
  text: string,
  options: ExplainTextOptions = {},
): Explained<TextExplanation> {
  const t = explain.tokenise(text)
  const T = t.tokens.length
  if (T === 0) throw new WhiteboxError('value', 'explainText: the text has no words')
  const f = explain.tokenPredictor(t.tokens, classify, options.mask === undefined ? {} : { mask: options.mask })
  const method = options.method ?? 'shap'
  const key = options.key ?? makeKey(0)
  const ones = new Array<number>(T).fill(1)
  const zeros = new Array<number>(T).fill(0)
  let values: number[]
  if (method === 'ablation') values = explain.featureAblation(f, [ones], { baseline: 0 })[0] ?? []
  else if (method === 'shap')
    values = [...(explain.kernelShap(f, [ones], [zeros], { nSamples: options.nSamples ?? 500, key }).values[0] ?? [])]
  else {
    const r = explain.limeMasked(f, T, {
      nSamples: options.nSamples ?? 500,
      nFeatures: T,
      featureSelection: 'none',
      key,
    })
    values = zeros.map((_, j) => r.weights.find(([k]) => k === j)?.[1] ?? 0)
  }
  const [prediction = 0, baseValue = 0] = toFlat(
    f(fromStorage(Float64Array.from([...ones, ...zeros]), [2, T], 'float64')),
  )
  return {
    tokens: t.tokens,
    values,
    prediction,
    baseValue,
    exposition: [
      featureAttributions({
        method,
        features: [...t.tokens],
        output: options.output ?? 'score',
        baseValue,
        values: [values],
        data: [ones],
        predictions: [prediction],
        layout: { kind: 'tokens', text, spans: t.starts.map((s, i) => [s, t.ends[i] ?? s] as const) },
      }),
    ],
  }
}

/** The series explainers {@link explainSeries} offers. */
export type SeriesMethod = 'occlusion' | 'ablation' | 'shap'

/** Options of {@link explainSeries}. */
export interface ExplainSeriesOptions {
  readonly method?: SeriesMethod
  /** Steps per occluding window or per explained window (default 4). */
  readonly window?: number
  /** What removed steps become (default the series' mean). */
  readonly baseline?: number
  readonly nSamples?: number
  readonly classIndex?: number
  readonly key?: Key
}

/**
 * Explain a fitted model's prediction for one series (a row of time steps) by occluding windows of it, ablating
 * consecutive windows, or Kernel SHAP over the windows; the record lays the features out along time.
 *
 * @param fitted - A fitted model on series of equal length.
 * @param data - The series (rows of steps).
 * @param row - The series explained.
 * @param options - method, window, baseline, nSamples, classIndex and key.
 * @returns One attribution per step with a `featureAttributions` record.
 */
export function explainSeries(
  fitted: Fitted,
  data: LearnData,
  row: number,
  options: ExplainSeriesOptions = {},
): Explained<{ readonly values: readonly number[]; readonly prediction: number }> {
  const x = matrixView(data.x, 'explainSeries')
  if (!(Number.isInteger(row) && row >= 0 && row < x.n))
    throw new WhiteboxError('value', `explainSeries: no row ${row}`)
  const series = Array.from(x.data.subarray(row * x.d, (row + 1) * x.d))
  const output = modelOutput(fitted, data, 'auto', options.classIndex ?? predictedClass(fitted, series))
  const method = options.method ?? 'occlusion'
  const w = Math.min(options.window ?? 4, x.d)
  const baseline = options.baseline ?? series.reduce((a, b) => a + b, 0) / Math.max(1, x.d)
  const windows = explain.timeWindows(x.d, w)
  let values: number[]
  if (method === 'occlusion')
    values = explain.occlusion(output.predict, [series], { shape: [x.d], window: [w], baseline })[0] ?? []
  else if (method === 'ablation')
    values = explain.featureAblation(output.predict, [series], { featureMask: windows, baseline })[0] ?? []
  else {
    const { predict, groups } = explain.maskedPredictor(output.predict, series, windows, baseline)
    const G = groups.length
    const r = explain.kernelShap(predict, [new Array<number>(G).fill(1)], [new Array<number>(G).fill(0)], {
      nSamples: options.nSamples ?? 400,
      key: options.key ?? makeKey(0),
    })
    values = explain.expandGroups(r.values[0] ?? [], groups, x.d, { share: true })
  }
  const prediction = toFlat(output.predict(fromStorage(Float64Array.from(series), [1, x.d], 'float64')))[0] ?? 0
  const baseValue =
    toFlat(output.predict(fromStorage(new Float64Array(x.d).fill(baseline), [1, x.d], 'float64')))[0] ?? 0
  return {
    values,
    prediction,
    exposition: [
      featureAttributions({
        method,
        features: featureNamesOf(data, x.d),
        output: output.name,
        baseValue,
        values: [values],
        data: [series],
        predictions: [prediction],
        indices: [row],
        layout: { kind: 'series' },
        ...(method === 'occlusion' ? {} : { segments: windows }),
      }),
    ],
  }
}

/** Options of {@link morrisScreening}. */
export interface MorrisScreeningOptions {
  /** Trajectories (default 30). */
  readonly nTrajectories?: number
  /** Grid levels (default 4). */
  readonly levels?: number
  readonly classIndex?: number
  readonly key?: Key
}

/**
 * Morris screening of a fitted model over the box its data span: each feature's mean absolute elementary effect (its
 * importance) with the effects' standard deviation (non-linearity and interactions) as the spread, and every
 * trajectory's |effect| as a repeat, in an `importances` record.
 *
 * @param fitted - A fitted model.
 * @param data - The data whose per-feature range is the box.
 * @param options - nTrajectories, levels, classIndex and key.
 * @returns mu*, sigma and mu with an `importances` record.
 */
export function morrisScreening(
  fitted: Fitted,
  data: LearnData,
  options: MorrisScreeningOptions = {},
): Explained<{
  readonly muStar: readonly number[]
  readonly sigma: readonly number[]
  readonly mu: readonly number[]
}> {
  const x = matrixView(data.x, 'morrisScreening')
  const output = modelOutput(fitted, data, 'auto', options.classIndex)
  const bounds = Array.from({ length: x.d }, (_, j) => {
    let lo = Number.POSITIVE_INFINITY
    let hi = Number.NEGATIVE_INFINITY
    for (let i = 0; i < x.n; i++) {
      const v = x.data[i * x.d + j] ?? 0
      lo = Math.min(lo, v)
      hi = Math.max(hi, v)
    }
    return [lo, hi > lo ? hi : lo + 1] as [number, number]
  })
  const r = explain.morrisEffects(output.predict, bounds, {
    nTrajectories: options.nTrajectories ?? 30,
    levels: options.levels ?? 4,
    key: options.key ?? makeKey(0),
  })
  return {
    muStar: r.muStar,
    sigma: r.sigma,
    mu: r.mu,
    exposition: [
      importances({
        method: 'Morris',
        features: featureNamesOf(data, x.d),
        mean: r.muStar,
        spread: r.sigma,
        repeats: r.effects.map((e) => e.map(Math.abs)),
        metric: `mean |elementary effect| on ${output.name}`,
        direction: 'higher',
      }),
    ],
  }
}
