import { learnGolden, num, str } from '../../../testing/learn.ts'
import { key } from '../../random/index.ts'
import { type Tensor, toFlat } from '../../tensor/index.ts'
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

const samplers: Readonly<Record<string, Resampler>> = {
  smote: smote({ kNeighbors: 3 }),
  adasyn: adasyn({ kNeighbors: 3 }),
  borderline: borderlineSmote({ kNeighbors: 3, mNeighbors: 8 }),
  over: randomOverSampler(),
  under: randomUnderSampler(),
}

learnGolden('resampling against imbalanced-learn', 'learn/preprocessing/resampling', {
  tomek: ([x, y]) => ({ indices: Array.from(tomekLinks().fitResample({ x: x as Tensor, y: y as Tensor }).indices) }),
  nearMiss: ([x, y], p) => ({
    indices: Array.from(
      nearMiss({ version: num(p, 'version') as 1 | 2 | 3 }).fitResample({ x: x as Tensor, y: y as Tensor }).indices,
    ),
  }),
  danger: ([x, y], p) => ({ indices: borderlineSamples(x as Tensor, y as Tensor, num(p, 'label'), num(p, 'm')) }),
  counts: ([x, y], p) => {
    const sampler = samplers[str(p, 'sampler')]
    if (sampler === undefined) throw new Error('unknown sampler')
    const ys = toFlat(sampler.fitResample({ x: x as Tensor, y: y as Tensor }, { key: key(0) }).y)
    return { counts: [0, 1, 2].map((c) => ys.filter((v) => v === c).length) }
  },
})
