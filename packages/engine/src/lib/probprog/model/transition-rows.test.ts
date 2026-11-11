// A Markov chain whose transition rows are random (a Dirichlet latent per row of the matrix, `at(A, prev)` as the
// probabilities of the next state), with Gaussian readings at known means: the posterior of every row and of every state
// against the exact one by enumeration of all the state paths (the rows integrate out in closed form, a Dirichlet-multinomial
// per row, so the exact posterior of a row is a mixture over paths of Dirichlets of the path's transition counts).
import { describe, expect, it } from 'vitest'
import {
  compileGraph,
  epAlgorithm,
  defaultOperators as factorOperators,
  type MpState,
  marginalsOf,
} from '../../message-passing/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import * as d from './dists/index.ts'
import { defineModel, ops, toFactorGraph } from './index.ts'

const K = 3
const MEANS = [-3, 0, 3]
const SD = 1.5
const ALPHA = [
  [5, 1, 1],
  [1, 5, 1],
  [1, 1, 5],
]
const Y = [-2.5, -3.4, -1.2, 0.3, -0.4, 2.2, 3.9, 3.1, -2.8]
const Y2 = [0.2, -0.3, 0.9, -0.1, 3.2, 2.7, 3.5, 0.4, -0.6]

const lgam = (z: number): number => {
  const c = [
    676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ]
  const x = z - 1
  let a = 0.99999999999980993
  for (let i = 0; i < 8; i++) a += (c[i] as number) / (x + i + 1)
  const t = x + 7.5
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a)
}

/** The exact posterior by enumeration of the K^T paths: the state marginals and the mean and sd of every entry of the matrix. */
function exact(y: readonly number[]): { states: number[][]; mean: number[][]; sd: number[][] } {
  const T = y.length
  const paths = K ** T
  const logw: number[] = []
  const counts: number[][][] = []
  for (let p = 0; p < paths; p++) {
    const z: number[] = []
    for (let t = 0, r = p; t < T; t++, r = Math.floor(r / K)) z.push(r % K)
    const n = Array.from({ length: K }, () => new Array<number>(K).fill(0))
    for (let t = 1; t < T; t++) {
      const row = n[z[t - 1] as number] as number[]
      row[z[t] as number] = (row[z[t] as number] as number) + 1
    }
    let lw = -Math.log(K)
    for (let t = 0; t < T; t++) lw += -0.5 * (((y[t] as number) - (MEANS[z[t] as number] as number)) / SD) ** 2
    for (let j = 0; j < K; j++) {
      const a = ALPHA[j] as number[]
      const nj = n[j] as number[]
      const a0 = a.reduce((s, v) => s + v, 0)
      const n0 = nj.reduce((s, v) => s + v, 0)
      lw += lgam(a0) - lgam(a0 + n0) + a.reduce((s, v, k) => s + lgam(v + (nj[k] as number)) - lgam(v), 0)
    }
    logw.push(lw)
    counts.push(n)
    ;(counts[p] as number[][]).push(z)
  }
  const top = Math.max(...logw)
  const w = logw.map((v) => Math.exp(v - top))
  const total = w.reduce((s, v) => s + v, 0)
  const states = Array.from({ length: T }, () => new Array<number>(K).fill(0))
  const m1 = Array.from({ length: K }, () => new Array<number>(K).fill(0))
  const m2 = Array.from({ length: K }, () => new Array<number>(K).fill(0))
  for (let p = 0; p < paths; p++) {
    const wp = (w[p] as number) / total
    const z = (counts[p] as number[][])[K] as number[]
    z.forEach((s, t) => {
      ;(states[t] as number[])[s] = ((states[t] as number[])[s] as number) + wp
    })
    for (let j = 0; j < K; j++) {
      const a = ALPHA[j] as number[]
      const nj = (counts[p] as number[][])[j] as number[]
      const a0 = a.reduce((s, v) => s + v, 0) + nj.reduce((s, v) => s + v, 0)
      for (let k = 0; k < K; k++) {
        const ak = (a[k] as number) + (nj[k] as number)
        const mean = ak / a0
        ;(m1[j] as number[])[k] = ((m1[j] as number[])[k] as number) + wp * mean
        ;(m2[j] as number[])[k] = ((m2[j] as number[])[k] as number) + wp * ((ak * (ak + 1)) / (a0 * (a0 + 1)))
      }
    }
  }
  return {
    states,
    mean: m1,
    sd: m2.map((row, j) => row.map((v, k) => Math.sqrt(v - ((m1[j] as number[])[k] as number) ** 2))),
  }
}

function ep(y: readonly number[]) {
  const model = defineModel('rows', (b) => {
    const data = b.data('y', { dims: ['time'] })
    const regime = b.range('regime', { size: K })
    const means = b.const(MEANS, ['regime'])
    const alpha = b.const(ALPHA, ['regime'])
    const A = b.latent('A', d.dirichlet(alpha), { dims: [regime] })
    b.scan('z', {
      over: 'time',
      init: d.categorical({ probs: [1 / K, 1 / K, 1 / K] }),
      step: ({ prev }) => d.categorical({ probs: ops.at(A, prev) }),
      emit: ({ state, observe }) => observe('y', d.normal(ops.at(means, state), SD), data),
    })
  })
  const c = toFactorGraph(model.bind({ y: [...y] }))
  expect(c.issues).toEqual([])
  const g = compileGraph(c.graph as never, factorOperators())
  const r = runAlgorithm(epAlgorithm(g, { tol: 1e-10 }), { maxSteps: 400 })
  return { c, status: r.status, marginals: marginalsOf(g, r.state as MpState) }
}

describe('a Markov chain with Dirichlet transition rows, against enumeration of the paths', () => {
  it.each([
    ['a chain that visits all three states', Y],
    ['a chain that rarely leaves its state', Y2],
  ])('%s: the rows and the states are close to the exact posterior', (_, y) => {
    const e = exact(y)
    const r = ep(y)
    expect(r.status).toBe('converged')
    // One gate per transition, each choosing between the rows by the previous state.
    expect(r.c.graph?.factors.filter((f) => f.operator === 'gate').length).toBe(y.length - 1)
    for (let j = 0; j < K; j++) {
      const alpha = (r.marginals[`A[${j}]`] as { alpha: number[] }).alpha
      const a0 = alpha.reduce((s, v) => s + v, 0)
      alpha.forEach((v, k) => {
        const sd = Math.sqrt((v * (a0 - v)) / (a0 * a0 * (a0 + 1)))
        expect(v / a0, `row ${j}, entry ${k}: mean`).toBeCloseTo((e.mean[j] as number[])[k] as number, 1)
        expect(Math.abs(v / a0 - ((e.mean[j] as number[])[k] as number)), `row ${j}, entry ${k}`).toBeLessThan(0.02)
        // EP's spread is within a tenth either way of the exact one.
        expect(sd / ((e.sd[j] as number[])[k] as number), `row ${j}, entry ${k}: sd`).toBeGreaterThan(0.9)
        expect(sd / ((e.sd[j] as number[])[k] as number), `row ${j}, entry ${k}: sd`).toBeLessThan(1.15)
      })
    }
    for (let t = 0; t < y.length; t++) {
      const probs = (r.marginals[`z[${t}]`] as { probs: number[] }).probs
      probs.forEach((p, k) => {
        expect(Math.abs(p - ((e.states[t] as number[])[k] as number)), `state ${t}`).toBeLessThan(0.05)
      })
    }
  })

  it('refuses a matrix with fewer rows than the chain has states, with its reason', () => {
    const bad = defineModel('bad-rows', (b) => {
      const data = b.data('y', { dims: ['time'] })
      b.range('regime', { size: K })
      const few = b.range('few', { size: 2 })
      const A = b.latent('A', d.dirichlet(b.const(ALPHA.slice(0, 2), ['few'])), { dims: [few] })
      b.scan('z', {
        over: 'time',
        init: d.categorical({ probs: [1 / K, 1 / K, 1 / K] }),
        step: ({ prev }) => d.categorical({ probs: ops.at(A, prev) }),
        emit: ({ state, observe }) => observe('y', d.normal(ops.at(b.const(MEANS, ['regime']), state), SD), data),
      })
    })
    const c = toFactorGraph(bad.bind({ y: [...Y] }))
    expect(c.issues.length).toBeGreaterThan(0)
    expect(c.issues[0]?.reason).toMatch(/2 rows/)
  })
})
