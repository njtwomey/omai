import { learnGolden, num } from '../../../testing/learn.ts'
import { type Tensor, toFlat } from '../../tensor/index.ts'
import { timeSeriesSplit } from './blocked-splitters.ts'
import {
  type Fold,
  groupKFold,
  holdout,
  kFold,
  leaveOneGroupOut,
  leaveOneOut,
  leavePGroupsOut,
  leavePOut,
  predefinedSplit,
  type Splitter,
  splitIndices,
  stratifiedGroupKFold,
  stratifiedKFold,
} from './splitters.ts'

function output(folds: readonly Fold[]): Record<string, number[]> {
  return {
    train: folds.flatMap((f) => Array.from(f.train)),
    test: folds.flatMap((f) => Array.from(f.test)),
    trainSizes: folds.map((f) => f.train.length),
    testSizes: folds.map((f) => f.test.length),
  }
}

const run = (s: Splitter, [y, groups]: readonly Tensor[]): Record<string, number[]> => {
  const ys = toFlat(y as Tensor)
  return output(splitIndices(s, { n: ys.length, y: ys, ...(groups === undefined ? {} : { groups: toFlat(groups) }) }))
}

learnGolden('splitters against sklearn.model_selection', 'learn/model_selection/splitters', {
  kFold: (i, p) => run(kFold(num(p, 'nSplits')), i),
  stratifiedKFold: (i, p) => run(stratifiedKFold(num(p, 'nSplits')), i),
  groupKFold: (i, p) => run(groupKFold(num(p, 'nSplits')), i),
  stratifiedGroupKFold: (i, p) => run(stratifiedGroupKFold(num(p, 'nSplits')), i),
  leaveOneOut: (i) => run(leaveOneOut(), i),
  leavePOut: (i, p) => run(leavePOut(num(p, 'p')), i),
  leaveOneGroupOut: (i) => run(leaveOneGroupOut(), i),
  leavePGroupsOut: (i, p) => run(leavePGroupsOut(num(p, 'p')), i),
  predefinedSplit: ([f]) => run(predefinedSplit(toFlat(f as Tensor)), [f as Tensor]),
  timeSeriesSplit: (i, p) => {
    const max = num(p, 'maxTrainSize')
    const test = num(p, 'testSize')
    return run(
      timeSeriesSplit(num(p, 'nSplits'), {
        gap: num(p, 'gap'),
        ...(max < 0 ? {} : { maxTrainSize: max }),
        ...(test < 0 ? {} : { testSize: test }),
      }),
      i,
    )
  },
  holdout: (i, p) =>
    run(
      holdout({
        testSize: num(p, 'testSize'),
        shuffle: false,
        ...(typeof p['trainSize'] === 'number' ? { trainSize: p['trainSize'] } : {}),
      }),
      i,
    ),
})
