import { learnGolden, num, str } from '../../../testing/learn.ts'
import { type Tensor, toFlat } from '../../tensor/index.ts'
import { kFold, stratifiedKFold } from '../model-selection/splitters.ts'
import {
  type HandleUnknown,
  labelBinarizer,
  labelEncoder,
  multiLabelBinarizer,
  oneHotEncoder,
  ordinalEncoder,
  targetEncoder,
} from './encoders.ts'

const opt = (p: Readonly<Record<string, unknown>>, name: string): Record<string, number> =>
  typeof p[name] === 'number' ? { [name]: p[name] as number } : {}

learnGolden('encoders against sklearn.preprocessing', 'learn/preprocessing/encoders', {
  oneHot: ([x, z], p) => {
    const f = oneHotEncoder({
      drop: str(p, 'drop', 'none') as 'none' | 'first' | 'if_binary',
      handleUnknown: str(p, 'handleUnknown', 'error') as HandleUnknown,
      ...opt(p, 'minFrequency'),
      ...opt(p, 'maxCategories'),
    }).fit({ x: x as Tensor })
    const t = f.transform(z as Tensor)
    return { transform: t, inverse: f.inverseTransform(t), names: f.featureNamesOut().join(',') }
  },
  ordinal: ([x, z], p) => {
    const f = ordinalEncoder({
      handleUnknown: str(p, 'handleUnknown', 'error') as 'error' | 'use_encoded_value',
      ...opt(p, 'unknownValue'),
      ...opt(p, 'encodedMissingValue'),
      ...opt(p, 'minFrequency'),
    }).fit({ x: x as Tensor })
    const t = f.transform(z as Tensor)
    return { transform: t, inverse: f.inverseTransform(t) }
  },
  label: ([y, q]) => {
    const f = labelEncoder().fit({ x: [[0]], y: y as Tensor })
    return { transform: f.transform(q as Tensor), classes: [...f.classes] }
  },
  labelBinarizer: ([y]) => {
    const f = labelBinarizer().fit({ x: [[0]], y: y as Tensor })
    const t = f.transform(y as Tensor)
    return { transform: t, inverse: f.inverseTransform(t) }
  },
  multiLabel: (_, p) => {
    const b = multiLabelBinarizer(p['sets'] as number[][])
    return { transform: b.transform(p['query'] as number[][]), classes: [...b.classes] }
  },
  target: ([x, y, z], p) => {
    const smooth = p['smooth'] === 'auto' ? 'auto' : num(p, 'smooth')
    const est = targetEncoder({
      smooth,
      targetType: str(p, 'targetType') as 'continuous' | 'binary' | 'multiclass',
      cv: p['stratified'] === true ? stratifiedKFold(3) : kFold(3),
    })
    const fitTransform = est.fitTransform
    if (fitTransform === undefined) throw new Error('targetEncoder has no fitTransform')
    const { fitted, transformed } = fitTransform({ x: x as Tensor, y: toFlat(y as Tensor) })
    return { fitTransform: transformed, transform: fitted.transform(z as Tensor) }
  },
})
