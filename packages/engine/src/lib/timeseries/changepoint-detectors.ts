/**
 * Changepoint detectors behind one interface, so a page can run several on a series and lay their answers over the true
 * changepoints: Bayesian online changepoint detection, the offline PELT and binary segmentation, and the sequential
 * CUSUM and Page-Hinkley tests. A detector is a function from a series to the indices of the changepoints it reports
 * (the index of the first observation of a new segment, as the generator's truth gives them; for the two sequential
 * tests, the index at which the alarm was raised, which lags the change by their detection delay). The table gives each
 * its own knobs as a `union` space, like the other slots of the library.
 */

import {
  binarySegmentation,
  bocpd,
  hazards,
  mapChangepoints,
  pelt,
  type SegmentCost,
  segmentModels,
} from '../dynamics/filtering/index.ts'
import {
  choice,
  type DictSpace,
  defaults,
  dict,
  int,
  type Option,
  type OptionTable,
  optionTable,
  real,
  type SpaceValue,
} from '../space/index.ts'
import { cusum, pageHinkley } from './online-detection.ts'

/** A detector: the series in, the indices of the changepoints it reports out (sorted, without repeats). */
export type ChangepointDetect = (y: readonly number[]) => number[]

/** What a detector is, for selectors and validity rules. */
export interface DetectorProperties {
  /** Whether it reads the series once, in order (it could run live); the others see the whole series. */
  readonly online: boolean
  /** Whether the index it reports is where the segment begins, rather than the time of the alarm. */
  readonly locatesChange: boolean
}

const unique = (xs: readonly number[]): number[] => [...new Set(xs)].sort((a, b) => a - b)

function detector(
  description: string,
  properties: DetectorProperties,
  space: DictSpace,
  build: (knobs: Readonly<Record<string, SpaceValue>>) => ChangepointDetect,
): Option<ChangepointDetect, DetectorProperties> {
  // Knobs not given take the option's defaults.
  const full = defaults(space) as Readonly<Record<string, SpaceValue>>
  return { description, properties, space, make: (config) => build({ ...full, ...config }) }
}

const costs = choice(['l2', 'l1', 'normal', 'poisson'] as const, { default: 'l2', label: 'segment cost' })
const penalty = real(0.01, 10_000, { log: true, default: 10, label: 'penalty per segment' })

/**
 * The detectors as one slot: `bocpd` (Bayesian online changepoint detection with the segment model and hazard of the
 * option tables, reporting the starts its most probable run length commits to), `pelt` and `binseg` (offline, over a
 * segment cost), `cusum` and `page-hinkley` (sequential tests of a shift of the mean, reporting alarms). Each
 * option's `make` takes its knobs and returns a {@link ChangepointDetect}.
 *
 * @concept changepoint-detection
 * @example
 * ```ts
 * import { changepointDetectors } from 'mdml-engine/timeseries'
 * const detect = changepointDetectors.make({ kind: 'pelt', cost: 'l2', penalty: 5, minSize: 2 })
 * detect([0, 0, 0, 0, 5, 5, 5, 5]) // => [4]
 * ```
 */
export const changepointDetectors: OptionTable<ChangepointDetect, DetectorProperties> = optionTable(
  {
    bocpd: detector(
      'Bayesian online changepoint detection: the run-length posterior under a segment model and a hazard',
      { online: true, locatesChange: true },
      dict({
        model: segmentModels.space,
        hazard: hazards.space,
        changeThreshold: real(0.05, 0.99, { default: 0.5, label: 'detection threshold' }),
      }),
      (k) => (y) =>
        mapChangepoints(
          bocpd(y, segmentModels.make(k['model']), hazards.make(k['hazard']), { rows: false }),
          k['changeThreshold'] as number,
        ).map((c) => c.index),
    ),
    pelt: detector(
      'PELT: the exact segmentation minimising the segment costs plus a penalty per segment',
      { online: false, locatesChange: true },
      dict({ cost: costs, penalty, minSize: int(1, 1000, { log: true, default: 2, label: 'shortest segment' }) }),
      (k) => (y) =>
        pelt(y, k['penalty'] as number, { cost: k['cost'] as SegmentCost, minSize: k['minSize'] as number }).slice(
          0,
          -1,
        ),
    ),
    binseg: detector(
      'binary segmentation: repeatedly split where one split lowers the cost most, for a number of changepoints',
      { online: false, locatesChange: true },
      dict({
        cost: costs,
        changepoints: int(1, 100, { default: 3, label: 'number of changepoints' }),
        minSize: int(1, 1000, { log: true, default: 2, label: 'shortest segment' }),
      }),
      (k) => (y) =>
        binarySegmentation(
          y,
          { changepoints: k['changepoints'] as number },
          { cost: k['cost'] as SegmentCost, minSize: k['minSize'] as number },
        ).slice(0, -1),
    ),
    cusum: detector(
      'two-sided CUSUM of standardised deviations from a mean and spread estimated on a warm-up',
      { online: true, locatesChange: false },
      dict({
        allowance: real(0, 5, { default: 0.5, label: 'allowance k (standard deviations)' }),
        threshold: real(0.5, 50, { log: true, default: 5, label: 'decision limit h (standard deviations)' }),
        warmup: int(2, 10_000, { log: true, default: 20, label: 'warm-up observations' }),
      }),
      (k) => (y) =>
        unique(
          cusum(y, {
            allowance: k['allowance'] as number,
            threshold: k['threshold'] as number,
            warmup: k['warmup'] as number,
          }).alarms.map((a) => a.index),
        ),
    ),
    'page-hinkley': detector(
      'the Page-Hinkley test of a cumulative deviation from the running mean above its minimum',
      { online: true, locatesChange: false },
      dict({
        delta: real(0, 10, { default: 0.005, label: 'tolerated change δ' }),
        threshold: real(0.1, 10_000, { log: true, default: 50, label: 'alarm threshold λ' }),
        minInstances: int(1, 10_000, { log: true, default: 30, label: 'observations before an alarm' }),
      }),
      (k) => (y) =>
        unique(
          pageHinkley(y, {
            delta: k['delta'] as number,
            threshold: k['threshold'] as number,
            minInstances: k['minInstances'] as number,
          }).alarms.map((a) => a.index),
        ),
    ),
  },
  {
    default: 'bocpd',
    label: 'detector',
    description:
      'How changepoints are found: online Bayesian, offline optimal or greedy segmentation, or a sequential test.',
  },
)
