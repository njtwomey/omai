import { learnGolden, num } from '../../testing/learn.ts'
import { type Tensor, toArray, toFlat } from '../tensor/index.ts'
import { type DiscriminantHyper, discriminantAnalysis } from './discriminant.ts'

const priorsOf = (p: Readonly<Record<string, unknown>>): Partial<DiscriminantHyper> =>
  p['priors'] === undefined ? {} : { priors: toFlat(p['priors'] as Tensor) }

function outputs(f: ReturnType<ReturnType<typeof discriminantAnalysis>['fit']>, q: Tensor, withAxes: boolean) {
  const dec = toArray(f.decisionFunction(q)) as number[] | number[][]
  const diff = Array.isArray(dec[0])
    ? (dec as number[][]).map((row) => row.slice(1).map((v, c) => v - (row[0] ?? 0) + 0 * c))
    : (dec as number[]).map((v) => [v])
  return {
    predict: f.predict(q),
    proba: f.predictProba(q),
    decisionDiff: diff,
    means: f.params.means,
    priors: f.params.priors,
    ...(withAxes ? { explained: f.params.explainedVarianceRatio, scalings: f.params.scalings } : {}),
  }
}

learnGolden('discriminant analysis against sklearn.discriminant_analysis', 'learn/discriminant', {
  lda: ([x, y, q], p) => {
    const shrinkage = p['shrinkage']
    const f = discriminantAnalysis({
      covariance: 'pooled',
      ...(shrinkage === undefined ? {} : { shrinkage: shrinkage === 'auto' ? 'auto' : (shrinkage as number) }),
      ...priorsOf(p),
    }).fit({ x: x as Tensor, y: y as Tensor })
    const o = outputs(f, q as Tensor, true)
    // The lsqr solver has no axes; the golden's keys decide what is compared.
    return o as never
  },
  qda: ([x, y, q], p) => {
    const f = discriminantAnalysis({
      covariance: 'class',
      regParam: num(p, 'regParam', 0),
      ...(p['shrinkage'] === undefined ? {} : { shrinkage: p['shrinkage'] as number | 'auto' }),
      ...priorsOf(p),
    }).fit({ x: x as Tensor, y: y as Tensor })
    return outputs(f, q as Tensor, false) as never
  },
})
