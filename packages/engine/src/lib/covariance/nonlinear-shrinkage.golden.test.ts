import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import { quadraticInverseShrinkage } from './nonlinear-shrinkage.ts'

const fit = (x: Parameters<typeof quadraticInverseShrinkage>[0], o: Readonly<Record<string, unknown>>) =>
  quadraticInverseShrinkage(x, { assumeCentered: o['assumeCentered'] === true })

goldenSuite('quadratic-inverse shrinkage against a transcription of QIS.m', 'covariance/nonlinear_shrinkage', {
  covariance: ([x], o) => tensor(fit(x, o).covariance),
  sampleEigenvalues: ([x], o) => tensor(fit(x, o).sampleEigenvalues),
  shrunkEigenvalues: ([x], o) => tensor(fit(x, o).shrunkEigenvalues),
})
