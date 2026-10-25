/**
 * Probability calibration for binary classifiers: reliability curves (observed frequency against mean forecast per bin)
 * with expected and maximum calibration errors and the Brier score, and `calibrateClassifier`, which recalibrates a
 * fitted model by Platt scaling, isotonic regression or temperature scaling on held-out rows (as scikit-learn's
 * `CalibratedClassifierCV` and netcal) and compares them.
 */

import { WhiteboxError } from '../../lib/base/index.ts'
import { calibrationRecord, type Explained, type ReliabilityCurve } from '../../lib/exposition/index.ts'
import {
  type Calibrator,
  type Fitted,
  isotonicCalibration,
  type LearnData,
  plattScaling,
  temperatureScaling,
} from '../../lib/learn/index.ts'
import {
  brierScoreLoss,
  calibrationCurve,
  expectedCalibrationError,
  maximumCalibrationError,
} from '../../lib/metrics/index.ts'
import { type Key, key as makeKey } from '../../lib/random/index.ts'
import { toFlat } from '../../lib/tensor/index.ts'
import {
  binaryLabels,
  type FairnessData,
  positiveScores,
  type SensitiveSpec,
  sensitiveGroups,
} from '../fairness/index.ts'
import { splitData } from './_split.ts'

/** The calibrators {@link calibrateClassifier} fits. */
export type CalibratorName = 'platt' | 'isotonic' | 'temperature'

/** Options of the reliability measures. */
export interface ReliabilityOptions {
  /** Bins of the curves and errors (default 10). */
  readonly nBins?: number
  /** Equal-width (default) or equal-mass bins. */
  readonly strategy?: 'uniform' | 'quantile'
}

/**
 * One reliability curve of binary forecasts: the bins' mean forecast and observed frequency with the expected and
 * maximum calibration errors and the Brier score (`whitebox-engine/metrics`).
 *
 * @param name - The curve's name.
 * @param labels - Binary labels.
 * @param probabilities - Positive-class probabilities.
 * @param options - Bins and strategy.
 * @param group - A group the curve belongs to (curves of one group share a colour).
 * @returns The curve.
 */
export function reliabilityCurve(
  name: string,
  labels: readonly number[],
  probabilities: readonly number[],
  options: ReliabilityOptions = {},
  group?: string,
): ReliabilityCurve {
  const nBins = options.nBins ?? 10
  const c = calibrationCurve(labels, probabilities, { nBins, strategy: options.strategy ?? 'uniform' })
  // Two columns, so the errors read the positive class's probability against the labels.
  const two = probabilities.map((p) => [1 - p, p])
  return {
    name,
    ...(group === undefined ? {} : { group }),
    meanPredicted: c.probPred,
    observed: c.probTrue,
    counts: c.counts,
    ece: expectedCalibrationError(labels, two, { nBins }).estimate.value,
    mce: maximumCalibrationError(labels, two, { nBins }).estimate.value,
    brier: brierScoreLoss(labels, probabilities).estimate.value,
  }
}

/** Options of {@link calibrateClassifier}. */
export interface CalibrateOptions extends ReliabilityOptions {
  /** The calibrators to fit (default Platt, isotonic and temperature). */
  readonly methods?: readonly CalibratorName[]
  /** Rows to calibrate on (default: half the data, the other half to measure on). */
  readonly calibration?: LearnData
  /** Also draw each group's curve before and after the first calibrator (a protected attribute or labels). */
  readonly sensitive?: SensitiveSpec
  readonly key?: Key
}

const logit = (p: number): number => {
  const q = Math.min(1 - 1e-12, Math.max(1e-12, p))
  return Math.log(q / (1 - q))
}

/** A calibrator fitted to scores and labels, as a map from probabilities to probabilities. */
function fitCalibrator(
  name: CalibratorName,
  p: readonly number[],
  y: readonly number[],
): (q: readonly number[]) => number[] {
  const fitted: Calibrator =
    name === 'platt'
      ? plattScaling(p.map(logit), y)
      : name === 'isotonic'
        ? isotonicCalibration(p, y)
        : temperatureScaling(p.map(logit), y)
  return (q) => {
    const out = toFlat(fitted.predict(name === 'isotonic' ? [...q] : q.map(logit)))
    // Binary maps give the positive class; two-column outputs give both.
    return out.length === q.length ? Array.from(out) : q.map((_, i) => out[2 * i + 1] ?? 0)
  }
}

const NAMES: Readonly<Record<CalibratorName, string>> = {
  platt: 'Platt scaling',
  isotonic: 'isotonic',
  temperature: 'temperature',
}

/**
 * Recalibrate a fitted binary classifier's probabilities (sklearn's `CalibratedClassifierCV` with a prefit model;
 * netcal): fit Platt scaling (a logistic map of the logit), isotonic regression (a monotone step map) and temperature
 * scaling (one divisor of the logit) on calibration rows, and measure each on held-out rows. Exposition: a
 * `calibration` record with the reliability curves before and after, each with its expected and maximum calibration
 * error and its map from raw to calibrated probability; with `sensitive`, each group's curve before and after the
 * first calibrator.
 *
 * @param model - A fitted binary classifier with probabilities.
 * @param data - Labelled rows (split into calibration and test rows unless `calibration` is given).
 * @param options - Calibrators, bins, calibration rows, groups and key.
 * @returns Each calibrator's map and its errors, with the record.
 * @throws WhiteboxError `value` when the model gives no probabilities.
 */
export function calibrateClassifier(
  model: Fitted,
  data: FairnessData,
  options: CalibrateOptions = {},
): Explained<{
  readonly maps: Readonly<Partial<Record<CalibratorName, (p: readonly number[]) => number[]>>>
  readonly ece: Readonly<Record<string, number>>
}> {
  const [cal, test] =
    options.calibration === undefined
      ? splitData(data, [0.5, 0.5], options.key ?? makeKey(0))
      : [options.calibration as FairnessData, data]
  if (cal === undefined || test === undefined) throw new WhiteboxError('value', 'calibrateClassifier: no rows')
  const raw = (d: LearnData): number[] => {
    const s = positiveScores(model, d.x)
    if (!s.probability) throw new WhiteboxError('value', 'calibrateClassifier: the model gives no probabilities')
    return s.scores
  }
  const pCal = raw(cal)
  const yCal = binaryLabels(cal, 'calibrateClassifier')
  const pTest = raw(test)
  const yTest = binaryLabels(test, 'calibrateClassifier')
  const methods = options.methods ?? ['platt', 'isotonic', 'temperature']
  const grid = Array.from({ length: 101 }, (_, i) => i / 100)
  const maps: Partial<Record<CalibratorName, (p: readonly number[]) => number[]>> = {}
  const curves: ReliabilityCurve[] = [reliabilityCurve('uncalibrated', yTest, pTest, options)]
  for (const m of methods) {
    const map = fitCalibrator(m, pCal, yCal)
    maps[m] = map
    curves.push({ ...reliabilityCurve(NAMES[m], yTest, map(pTest), options), map: { x: grid, y: map(grid) } })
  }
  if (options.sensitive !== undefined) {
    const groups = sensitiveGroups(test, options.sensitive)
    const first = methods[0]
    const after = first === undefined ? undefined : maps[first]?.(pTest)
    for (const [g, name] of groups.names.entries()) {
      const rows = groups.index.flatMap((v, i) => (v === g ? [i] : []))
      const pick = (v: readonly number[]): number[] => rows.map((i) => v[i] ?? 0)
      curves.push(reliabilityCurve(`${name}, uncalibrated`, pick(yTest), pick(pTest), options, name))
      if (after !== undefined && first !== undefined)
        curves.push(reliabilityCurve(`${name}, ${NAMES[first]}`, pick(yTest), pick(after), options, name))
    }
  }
  return {
    maps,
    ece: Object.fromEntries(curves.map((c) => [c.name, c.ece])),
    exposition: [
      calibrationRecord({ nBins: options.nBins ?? 10, strategy: options.strategy ?? 'uniform', curves, split: 'test' }),
    ],
  }
}
