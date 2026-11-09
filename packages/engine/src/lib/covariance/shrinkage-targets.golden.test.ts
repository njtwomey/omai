import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import { type ShrinkageTarget, shrinkageTargetMatrix, targetShrinkage } from './shrinkage-targets.ts'

const rule = (o: Readonly<Record<string, unknown>>): 'analytic' | 'oas' | 'rblw' =>
  (o['rule'] as 'analytic' | 'oas' | 'rblw' | undefined) ?? 'analytic'

goldenSuite('shrinkage targets against the references', 'covariance/shrinkage_targets', {
  target: ([x], o) => tensor(shrinkageTargetMatrix(x, o['target'] as ShrinkageTarget)),
  intensity: ([x], o) => targetShrinkage(x, { target: o['target'] as ShrinkageTarget, intensity: rule(o) }).shrinkage,
  covariance: ([x], o) =>
    tensor(targetShrinkage(x, { target: o['target'] as ShrinkageTarget, intensity: rule(o) }).covariance),
})
