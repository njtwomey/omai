import { learnGolden, num, str } from '../../../testing/learn.ts'
import { key } from '../../random/index.ts'
import { svc } from '../../svm/index.ts'
import { type Tensor, toFlat } from '../../tensor/index.ts'
import { stratifiedKFold } from '../model-selection/splitters.ts'
import { naiveBayes } from '../naive-bayes.ts'
import { smoten, smotenc, svmSmote, valueDifferenceMetric } from './oversampling.ts'
import {
  adasyn,
  borderlineSamples,
  borderlineSmote,
  nearMiss,
  type Resampler,
  randomOverSampler,
  randomUnderSampler,
  smote,
  tomekLinks,
} from './resampling.ts'
import {
  allKnn,
  clusterCentroids,
  condensedNearestNeighbour,
  editedNearestNeighbours,
  instanceHardnessThreshold,
  neighbourhoodCleaningRule,
  oneSidedSelection,
  repeatedEditedNearestNeighbours,
} from './undersampling.ts'

const samplers: Readonly<Record<string, Resampler>> = {
  smote: smote({ kNeighbors: 3 }),
  adasyn: adasyn({ kNeighbors: 3 }),
  borderline: borderlineSmote({ kNeighbors: 3, mNeighbors: 8 }),
  over: randomOverSampler(),
  under: randomUnderSampler(),
  smoten: smoten({ kNeighbors: 3 }),
  smotenc: smotenc({ categorical: [1], kNeighbors: 3 }),
  centroids: clusterCentroids(),
  svmsmote: svmSmote({ svm: svc.with({ kernel: 'rbf' }), kNeighbors: 3, mNeighbors: 8 }),
}

/** The kept indices of a resampler on a case's data. */
const kept = (sampler: Resampler, [x, y]: readonly Tensor[]): { indices: number[] } => ({
  indices: Array.from(sampler.fitResample({ x: x as Tensor, y: y as Tensor }, { key: key(0) }).indices),
})

/** The starting samples of the condensing rules: a position in each of the two edited classes. */
const seeds = (p: Readonly<Record<string, unknown>>): Record<number, number[]> => ({
  0: [num(p, 'seed0')],
  1: [num(p, 'seed1')],
})

learnGolden('resampling against imbalanced-learn', 'learn/preprocessing/resampling', {
  tomek: ([x, y]) => ({ indices: Array.from(tomekLinks().fitResample({ x: x as Tensor, y: y as Tensor }).indices) }),
  nearMiss: ([x, y], p) => ({
    indices: Array.from(
      nearMiss({ version: num(p, 'version') as 1 | 2 | 3 }).fitResample({ x: x as Tensor, y: y as Tensor }).indices,
    ),
  }),
  danger: ([x, y], p) => ({ indices: borderlineSamples(x as Tensor, y as Tensor, num(p, 'label'), num(p, 'm')) }),
  enn: (d, p) => kept(editedNearestNeighbours({ kindSel: str(p, 'kind') as 'all' | 'mode' }), d),
  renn: (d, p) => kept(repeatedEditedNearestNeighbours({ kindSel: str(p, 'kind', 'all') as 'all' | 'mode' }), d),
  allknn: (d, p) => kept(allKnn({ nNeighbors: num(p, 'n', 3), allowMinority: num(p, 'allow', 0) === 1 }), d),
  ncr: (d, p) => kept(neighbourhoodCleaningRule({ thresholdCleaning: num(p, 'threshold', 0.5) }), d),
  cnn: (d, p) => kept(condensedNearestNeighbour({ seeds: seeds(p), nNeighbors: num(p, 'k', 1) }), d),
  oss: (d, p) => kept(oneSidedSelection({ seeds: seeds(p), nNeighbors: num(p, 'k', 1) }), d),
  iht: (d) => kept(instanceHardnessThreshold({ estimator: naiveBayes(), cv: stratifiedKFold(3) }), d),
  vdm: ([x, y, q]) => ({ distance: valueDifferenceMetric(x as Tensor, y as Tensor)(q as Tensor, x as Tensor) }),
  counts: ([x, y], p) => {
    const sampler = samplers[str(p, 'sampler')]
    if (sampler === undefined) throw new Error('unknown sampler')
    const ys = toFlat(sampler.fitResample({ x: x as Tensor, y: y as Tensor }, { key: key(0) }).y)
    return { counts: [0, 1, 2].map((c) => ys.filter((v) => v === c).length) }
  },
})
