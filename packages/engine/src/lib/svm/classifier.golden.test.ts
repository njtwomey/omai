import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { tensor, toFlat } from '../tensor/index.ts'
import { type SvcHyper, type SvcParams, svc } from './classifier.ts'

type Out = Readonly<Record<string, unknown>>

/** libsvm's layout of the dual coefficients: row j - 1 holds the class i < j vectors of pair (i, j), row i those of class j. */
function libsvmDualCoef(p: SvcParams): number[][] {
  const k = p.classes.length
  const nv = p.vectors.length
  const classOf = p.nSupport.flatMap((n, c) => new Array<number>(n).fill(c))
  if (k === 2) {
    const row = new Array<number>(nv).fill(0)
    const pair = p.pairs[0]
    for (const [t, position] of (pair?.support ?? []).entries()) row[position] = pair?.coef[t] ?? 0
    return [row]
  }
  const rows = Array.from({ length: k - 1 }, () => new Array<number>(nv).fill(0))
  for (const pair of p.pairs) {
    pair.support.forEach((position, t) => {
      const c = classOf[position] ?? 0
      const row = c === pair.positive ? pair.negative - 1 : pair.positive
      ;(rows[row] as number[])[position] = pair.coef[t] ?? 0
    })
  }
  return rows
}

describe('svc against scikit-learn (libsvm)', () => {
  for (const c of loadFixture('svm/classifier').cases) {
    it(c.name, () => {
      const out = c.output as Out
      const x = decode(c.inputs[0]) as never
      const y = decode(c.inputs[1]) as never
      const xt = decode(c.inputs[2]) as never
      const weights = c.inputs[3] === undefined ? undefined : (decode(c.inputs[3]) as never)
      const p = c.params as Record<string, unknown>
      const hyper: Partial<SvcHyper> = {
        form: p['form'] as 'c' | 'nu',
        tol: 1e-8,
        ...(p['C'] === undefined ? {} : { C: p['C'] as number }),
        ...(p['nu'] === undefined ? {} : { nu: p['nu'] as number }),
        kernel: p['kernel'] as 'rbf',
        ...(p['gamma'] === undefined ? {} : { gamma: p['gamma'] as number | 'scale' | 'auto' }),
        ...(p['degree'] === undefined ? {} : { degree: p['degree'] as number }),
        ...(p['coef0'] === undefined ? {} : { coef0: p['coef0'] as number }),
        ...(p['classWeight'] === undefined
          ? {}
          : {
              classWeight:
                p['classWeight'] === 'balanced'
                  ? ('balanced' as const)
                  : Object.fromEntries(
                      Object.entries(p['classWeight'] as Record<string, number>).map(([k, v]) => [Number(k), v]),
                    ),
            }),
      }
      const fitted = svc.with(hyper).fit({ x, y, ...(weights === undefined ? {} : { weights }) }, { explain: false })
      const params = fitted.params
      expectMatches(tensor(params.supportIndex), out['support'], 0, 0, `${c.name} support_`)
      expectMatches(tensor(params.nSupport), out['nSupport'], 0, 0, `${c.name} n_support_`)
      const dual = libsvmDualCoef(params)
      expectMatches(tensor(dual), out['dualCoef'], c.rtol, c.atol, `${c.name} dual_coef_`)
      expectMatches(
        tensor(params.pairs.map((pair) => pair.intercept)),
        out['intercept'],
        c.rtol,
        c.atol,
        `${c.name} intercept_`,
      )
      expectMatches(fitted.predict(xt), out['predict'], 0, 0, `${c.name} predict`)
      if (params.classes.length === 2) {
        expectMatches(fitted.decisionFunction(xt), out['ovo'], c.rtol, c.atol, `${c.name} decision_function`)
      } else {
        const ovr = fitted.decisionFunction(xt)
        expectMatches(ovr, out['ovr'], c.rtol, c.atol, `${c.name} decision_function (ovr)`)
        const ovo = svc
          .with({ ...hyper, decisionShape: 'ovo' })
          .revive(params)
          .decisionFunction(xt)
        expectMatches(ovo, out['ovo'], c.rtol, c.atol, `${c.name} decision_function (ovo)`)
      }
      expect(toFlat(fitted.predict(xt)).length).toBeGreaterThan(0)
    })
  }
})
