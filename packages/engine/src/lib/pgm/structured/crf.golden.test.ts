import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { jacobianFd } from '../../numerics/differentiate/index.ts'
import { fsolve } from '../../numerics/roots/index.ts'
import { sgd } from '../../optim/index.ts'
import { type Tensor, tensor, toArray, toFlat } from '../../tensor/index.ts'
import { expandAttributeTemplates, featureIndex, parseCrfppTemplate, type TokenTable } from '../../text/index.ts'
import { metricSeries, recorder } from '../../trace/index.ts'
import * as P from '../index.ts'

const arr = (x: unknown): unknown => toArray(decode(x) as Tensor)

/** Train to the exact optimum: Newton's method on the gradient, with a finite-difference Jacobian of it. */
function newtonFit(start: P.LinearChainCrf, data: readonly P.LabelledSequence[], c2: number): P.LinearChainCrf {
  const grad = (w: readonly number[]): number[] =>
    Array.from(P.crfObjective(P.withCrfParameters(start, w), data, { c2 }).gradient)
  const r = fsolve((v) => tensor(grad(Array.from(toFlat(v)))), Array.from(P.crfParameters(start)), {
    jacobian: (v) => jacobianFd(grad, Array.from(toFlat(v))),
    ftol: 1e-12,
  })
  return P.withCrfParameters(start, toFlat(r.x))
}

describe('linear-chain CRFs against pytorch-crf and sklearn-crfsuite', () => {
  for (const c of loadFixture('pgm/structured/crf').cases) {
    it(c.name, () => {
      if (c.fn === 'exact') {
        const emissions = arr(c.inputs[0]) as number[][]
        const transitions = arr(c.inputs[1]) as number[][]
        const tags = arr(c.inputs[2]) as number[]
        const K = transitions.length
        const labels = transitions.map((_, k) => `y${k}`)
        // One attribute per position carries that position's emission scores.
        const attributes = emissions.map((_, t) => `pos=${t}`)
        const crf = P.linearChainCrf(labels, attributes, { state: emissions.flat(), transition: transitions.flat() })
        const x = emissions.map((_, t) => ({ [`pos=${t}`]: 1 }))
        const ll = P.crfLogLikelihood(crf, x, tags)
        const marginals = P.crfMarginals(crf, x)
        expectMatches(tensor([ll, ...marginals.flat()]), c.output, c.rtol, c.atol, c.name)
        expect(P.crfDecode(crf, x).labels).toEqual((c.params['viterbi'] as number[]).map((k) => labels[k]))
        expect(marginals.every((row) => row.length === K)).toBe(true)
        return
      }
      if (c.fn === 'crfpp') {
        const template = parseCrfppTemplate(c.params['template'] as string)
        const tables = c.params['tables'] as TokenTable[]
        const Y = c.params['y'] as string[][]
        const labels = c.params['labels'] as string[]
        const observations = tables.map((t) => P.crfppObservation(template, t))
        // The expansion follows CRF++'s apply_rule exactly.
        expect(observations.map((o) => o.unigram.map((item) => Object.keys(item)))).toEqual(
          (c.params['expanded'] as Record<string, number>[][]).map((s) => s.map((item) => Object.keys(item))),
        )
        const start = P.crfppCrf(template, labels, tables, { minFrequency: c.params['minFrequency'] as number })
        expect(start.bigramAttributes).toEqual(['B'])
        const data = observations.map((x, i) => ({ x, y: Y[i] ?? [] }))
        const fitted = newtonFit(start, data, c.params['c2'] as number)
        const state = c.params['state'] as Record<string, number>
        const K = labels.length
        for (const [i, a] of fitted.attributes.entries())
          for (const [k, label] of labels.entries())
            expect(Math.abs((fitted.stateWeights[i * K + k] ?? 0) - (state[`${a}|${label}`] ?? 0))).toBeLessThan(2e-6)
        // CRF++'s bare B line is CRFsuite's label-pair transition.
        const transition = c.params['transition'] as Record<string, number>
        for (const [i, a] of labels.entries())
          for (const [k, b] of labels.entries())
            expect(Math.abs((fitted.bigramWeights[i * K + k] ?? 0) - (transition[`${a}|${b}`] ?? 0))).toBeLessThan(2e-6)
        const test = P.crfppObservation(template, c.params['test'] as TokenTable)
        expect(P.crfDecode(fitted, test).labels).toEqual(c.params['prediction'])
        return
      }
      const X = c.params['x'] as P.ItemFeatures[][]
      const Y = c.params['y'] as string[][]
      const labels = c.params['labels'] as string[]
      const data = X.map((x, i) => ({ x, y: Y[i] ?? [] }))
      const start = P.linearChainCrf(labels, featureIndex(X).features)
      const fitted = newtonFit(start, data, c.params['c2'] as number)
      // CRFsuite reports weights to 6 decimals and stops at its own tolerance.
      const state = c.params['state'] as Record<string, number>
      const K = labels.length
      for (const [i, a] of fitted.attributes.entries()) {
        for (const [k, label] of labels.entries()) {
          const theirs = state[`${a}|${label}`] ?? 0
          expect(Math.abs((fitted.stateWeights[i * K + k] ?? 0) - theirs)).toBeLessThan(2e-6)
        }
      }
      const transition = c.params['transition'] as Record<string, number>
      for (const [i, a] of labels.entries()) {
        for (const [k, b] of labels.entries()) {
          expect(Math.abs((fitted.transitionWeights[i * K + k] ?? 0) - (transition[`${a}|${b}`] ?? 0))).toBeLessThan(
            2e-6,
          )
        }
      }
      const marginals = c.params['marginals'] as number[][][]
      for (const [s, x] of X.entries()) {
        const m = P.crfMarginals(fitted, x)
        for (const [t, row] of (marginals[s] ?? []).entries())
          for (const [k, v] of row.entries()) expect(Math.abs((m[t]?.[k] ?? 0) - v)).toBeLessThan(1e-5)
      }
      const test = c.params['test'] as P.ItemFeatures[][]
      expect(P.crfDecode(fitted, test[0] ?? []).labels).toEqual(c.params['prediction'])
      // The trained cases fit to the exact optimum by Newton's method with a finite-difference Jacobian (about 2 s on
      // an idle machine), so they get room for a busy one.
    }, 30_000)
  }
})

describe('CRF features, training and general CRFs', () => {
  it('expands templates with boundary padding and indexes attributes by frequency', () => {
    const tokens = [
      { w: 'a', p: 'D' },
      { w: 'cat', p: 'N' },
    ]
    const padded = expandAttributeTemplates(tokens, [[['w', -1]], [['p', 1]]], { boundary: 'pad', bias: true })
    expect(padded[0]).toEqual({ bias: 1, 'w[-1]=_B-1': 1, 'p[1]=N': 1 })
    expect(padded[1]).toEqual({ bias: 1, 'w[-1]=a': 1, 'p[1]=_B+1': 1 })
    expect(featureIndex([padded]).counts).toEqual([2, 1, 1, 1, 1])
  })

  it('CRF++ bigram templates give observation-dependent transitions: exact normaliser and gradient', () => {
    const template = parseCrfppTemplate('U00:%x[0,0]\nB\nB01:%x[0,1]\nB02:%x[-1,0]/%x[0,0]')
    const tables: TokenTable[] = [
      [
        ['a', 'x'],
        ['b', 'y'],
        ['a', 'y'],
      ],
      [
        ['b', 'x'],
        ['a', 'x'],
      ],
    ]
    const labels = ['P', 'Q', 'R']
    const start = P.crfppCrf(template, labels, tables)
    expect(start.bigramAttributes).toEqual(['B', 'B01:y', 'B02:a/b', 'B02:b/a', 'B01:x'])
    const n = P.crfParameters(start).length
    const crf = P.withCrfParameters(
      start,
      Array.from({ length: n }, (_, i) => Math.sin(1.7 * i + 0.3)),
    )
    const x = P.crfppObservation(template, tables[0] ?? [])
    // Brute force: sum over all 27 labellings.
    let z = 0
    for (let a = 0; a < 3; a++)
      for (let b = 0; b < 3; b++) for (let c = 0; c < 3; c++) z += Math.exp(P.crfScore(crf, x, [a, b, c]))
    expect(P.crfLogLikelihood(crf, x, [0, 1, 2])).toBeCloseTo(P.crfScore(crf, x, [0, 1, 2]) - Math.log(z), 10)
    const data = tables.map((t, i) => ({
      x: P.crfppObservation(template, t),
      y: i === 0 ? ['P', 'Q', 'R'] : ['R', 'R'],
    }))
    const { gradient } = P.crfObjective(crf, data, { c2: 0.2 })
    const f = (w: readonly number[]): number[] => [P.crfObjective(P.withCrfParameters(crf, w), data, { c2: 0.2 }).value]
    const fd = toFlat(jacobianFd(f, Array.from(P.crfParameters(crf))))
    for (const [i, g] of gradient.entries()) expect(g).toBeCloseTo(fd[i] ?? 0, 6)
    const marginals = P.crfMarginals(crf, x)
    for (const row of marginals) expect(row.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 12)
  })

  it('the objective gradient matches finite differences, and first-order training is traced', () => {
    const data = [
      { x: [{ a: 1, b: 0.5 }, { b: 1 }], y: ['P', 'Q'] },
      { x: [{ a: 1 }, { a: 1 }, { b: 2 }], y: ['P', 'P', 'Q'] },
    ]
    const crf = P.withCrfParameters(
      P.linearChainCrf(['P', 'Q'], ['a', 'b']),
      [0.1, -0.2, 0.3, 0.05, -0.1, 0.2, 0.4, -0.3],
    )
    const { gradient } = P.crfObjective(crf, data, { c2: 0.3 })
    const f = (w: readonly number[]): number[] => [P.crfObjective(P.withCrfParameters(crf, w), data, { c2: 0.3 }).value]
    const fd = toFlat(jacobianFd(f, Array.from(P.crfParameters(crf))))
    for (const [i, g] of gradient.entries()) expect(g).toBeCloseTo(fd[i] ?? 0, 7)
    const rec = recorder()
    const fit = P.fitCrf(data, {
      c2: 0.3,
      method: sgd({ learningRate: 0.2 }),
      maxIter: 2000,
      gradTol: 1e-7,
      tracer: rec,
    })
    expect(fit.converged).toBe(true)
    const objective = metricSeries(rec.run(), 'objective').values
    expect(objective[objective.length - 1] ?? 0).toBeLessThan(objective[0] ?? 0)
    expect(P.crfDecode(fit.crf, data[1]?.x ?? []).labels).toEqual(['P', 'P', 'Q'])
  })

  it('general CRFs reduce to linear chains and train with exact or BP marginals', () => {
    // A two-node chain as a general CRF: unary features per variable and label, one pairwise agreement feature.
    const variables = ['y0', 'y1'].map((name) => ({ name, states: ['P', 'Q'] }))
    const instance = (xs: readonly number[]): P.CrfInstance => ({
      cliques: [
        { variables: ['y0'], features: [[[0, xs[0] ?? 0]], [[1, xs[0] ?? 0]]] },
        { variables: ['y1'], features: [[[0, xs[1] ?? 0]], [[1, xs[1] ?? 0]]] },
        { variables: ['y0', 'y1'], features: [[[2, 1]], [], [], [[2, 1]]] },
      ],
    })
    const crf = P.generalCrf(variables, ['x:P', 'x:Q', 'agree'], [0.4, -0.2, 0.7])
    const chain = P.linearChainCrf(['P', 'Q'], ['x'], { state: [0.4, -0.2], transition: [0.7, 0, 0, 0.7] })
    const fg = P.crfFactorGraph(crf, instance([1, 2]))
    const exact = P.junctionTreeMarginals(fg)
    const viaChain = P.crfMarginals(chain, [{ x: 1 }, { x: 2 }])
    expect(P.factorValues(exact['y1'] as P.Factor)[0]).toBeCloseTo(viaChain[1]?.[0] ?? 0, 12)
    const data = [
      { instance: instance([1, 2]), y: { y0: 'P', y1: 'P' } },
      { instance: instance([2, -1]), y: { y0: 'Q', y1: 'Q' } },
      { instance: instance([0.5, 0.5]), y: { y0: 'P', y1: 'Q' } },
    ]
    const general = P.generalCrfObjective(crf, data, { c2: 0.1 })
    const linear = P.crfObjective(
      chain,
      data.map((d, i) => ({
        x:
          [
            [1, 2],
            [2, -1],
            [0.5, 0.5],
          ][i]?.map((v) => ({ x: v })) ?? [],
        y: [d.y.y0, d.y.y1],
      })),
      { c2: 0 },
    )
    expect(general.value).toBeCloseTo(linear.value + 0.1 * (0.16 + 0.04 + 0.49), 10)
    const bp = P.generalCrfObjective(crf, data, { c2: 0.1, inference: 'bp' })
    expect(bp.value).toBeCloseTo(general.value, 10)
    const fit = P.fitGeneralCrf(crf, data, { c2: 0.1, maxIter: 3000, gradTol: 1e-7 })
    expect(fit.converged).toBe(true)
    expect(
      Array.from(P.generalCrfObjective(fit.crf, data, { c2: 0.1 }).gradient).every((g) => Math.abs(g) < 1e-6),
    ).toBe(true)
  })
})
