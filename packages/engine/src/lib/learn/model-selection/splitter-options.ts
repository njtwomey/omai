/**
 * The splitter of a cross-validation as an option table: k-fold, stratified k-fold, their repeated forms, shuffle
 * splits and leave-one-out, each with the knobs of its own (folds, shuffling, repeats, test fraction). The table is one
 * `union` space for controls, grids and tuners, and builds the `Splitter` from a configuration.
 */

import { bool, dict, int, knobOption, type Option, type OptionTable, optionTable, real } from '../../space/index.ts'
import {
  kFold,
  leaveOneOut,
  repeatedKFold,
  repeatedStratifiedKFold,
  type Splitter,
  shuffleSplit,
  stratifiedKFold,
} from './splitters.ts'

/** What a splitter scheme guarantees, for validity rules and selectors. */
export interface SplitterProperties {
  /** Whether every sample is tested exactly once (what `crossValPredict` needs). */
  readonly partition: boolean
  /** Whether the scheme keeps class proportions (it needs labels). */
  readonly stratified: boolean
  /** Whether the folds depend on the key. */
  readonly randomised: boolean
}

type Knobs = (name: string) => number | boolean

function scheme(
  description: string,
  properties: SplitterProperties,
  space: ReturnType<typeof dict>,
  build: (k: Knobs) => Splitter,
): Option<Splitter, SplitterProperties> {
  return knobOption(description, properties, space, build)
}

const folds = (d: number): ReturnType<typeof int> => int(2, 20, { default: d, label: 'folds' })

/**
 * The splitters of a cross-validation as one slot: `kfold` (contiguous or shuffled folds), `stratified` (k-fold keeping
 * class proportions), `repeated` (shuffled k-fold repeated with fresh shuffles), `repeatedStratified`, `shuffle`
 * (independent random train and test splits) and `leaveOneOut`.
 *
 * @concept cross-validation
 * @example
 * ```ts
 * import { splitterSchemes } from 'mdml-engine/learn/model-selection'
 * splitterSchemes.make({ kind: 'repeated', folds: 3, repeats: 2 }).nSplits({ n: 12 }) // => 6
 * ```
 */
export const splitterSchemes: OptionTable<Splitter, SplitterProperties> = optionTable(
  {
    kfold: scheme(
      'k contiguous folds, shuffled first when asked',
      { partition: true, stratified: false, randomised: false },
      dict({ folds: folds(5), shuffle: bool({ default: false, label: 'shuffle' }) }),
      (k) => kFold(k('folds') as number, { shuffle: k('shuffle') as boolean }),
    ),
    stratified: scheme(
      'k folds that keep the class proportions',
      { partition: true, stratified: true, randomised: false },
      dict({ folds: folds(5), shuffle: bool({ default: false, label: 'shuffle' }) }),
      (k) => stratifiedKFold(k('folds') as number, { shuffle: k('shuffle') as boolean }),
    ),
    repeated: scheme(
      'shuffled k-fold repeated with a fresh shuffle each time',
      { partition: false, stratified: false, randomised: true },
      dict({ folds: folds(5), repeats: int(1, 50, { default: 3, label: 'repeats' }) }),
      (k) => repeatedKFold(k('folds') as number, k('repeats') as number),
    ),
    repeatedStratified: scheme(
      'shuffled stratified k-fold repeated with a fresh shuffle each time',
      { partition: false, stratified: true, randomised: true },
      dict({ folds: folds(5), repeats: int(1, 50, { default: 3, label: 'repeats' }) }),
      (k) => repeatedStratifiedKFold(k('folds') as number, k('repeats') as number),
    ),
    shuffle: scheme(
      'independent random train and test splits of a given test fraction',
      { partition: false, stratified: false, randomised: true },
      dict({
        splits: int(1, 100, { default: 10, label: 'splits' }),
        testSize: real(0.05, 0.95, { default: 0.25, label: 'test fraction' }),
      }),
      (k) => shuffleSplit(k('splits') as number, { testSize: k('testSize') as number }),
    ),
    leaveOneOut: scheme(
      'each sample is the test set once',
      { partition: true, stratified: false, randomised: false },
      dict({}),
      () => leaveOneOut(),
    ),
  },
  { default: 'kfold', label: 'splitter', description: 'How the rows are divided into training and test sets.' },
)
