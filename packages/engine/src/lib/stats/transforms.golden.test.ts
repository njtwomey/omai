import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import * as S from './index.ts'

goldenSuite('power transforms against scipy and sklearn', 'stats/transforms', {
  boxcoxLlf: ([x], p) => S.boxcoxLlf(Number(p['lam']), x),
  boxcoxFixed: ([x], p) => S.boxcox(x, { lambda: Number(p['lam']) }).transformed,
  yeojohnsonLlf: ([x], p) => S.yeojohnsonLlf(Number(p['lam']), x),
  yeojohnsonFixed: ([x], p) => S.yeojohnson(x, { lambda: Number(p['lam']) }).transformed,
  invBoxcox: ([y], p) => S.invBoxcox(y, Number(p['lam'])),
  boxcoxLambda: ([x]) => S.boxcox(x).lambda,
  boxcoxInterval: ([x]) => tensor([...(S.boxcox(x, { level: 0.9 }).interval ?? [])]),
  yeojohnsonLambda: ([x]) => S.yeojohnsonNormmax(x),
  invYeojohnson: ([y], p) => S.invYeojohnson(y, Number(p['lam'])),
  standardise: ([x]) => S.standardise(x).transformed,
})
