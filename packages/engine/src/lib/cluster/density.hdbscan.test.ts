import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { blobs } from '../datasets/index.ts'
import { key } from '../random/index.ts'
import { policies, recorder, runAlgorithm } from '../trace/index.ts'
import {
  hdbscan,
  hdbscanAlgorithm,
  hdbscanClustering,
  hdbscanContract,
  hdbscanEstimator,
  hdbscanSelections,
} from './index.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T

const blobData = generate(blobs, { key: key(3), knobs: { n: 45, centers: 3, clusterStd: 0.5 } }).data
// Two tight groups, one of them with a repeated point (a zero distance, so an infinite lambda), and a far point.
const twoGroups = [[0], [0.1], [0.1], [0.2], [0.3], [5], [5.1], [5.2], [5.3], [9]]

describe('hdbscanAlgorithm', () => {
  it('run to the end gives the result hdbscan gives, in tree, condense, select and label steps', () => {
    const alg = hdbscanAlgorithm(twoGroups, { minClusterSize: 3 })
    const run = runAlgorithm(alg, { maxSteps: 100 })
    const r = hdbscan(twoGroups, { minClusterSize: 3 })
    expect(run.status).toBe('converged')
    expect(run.state.phase).toBe('done')
    expect([...run.state.labels]).toEqual(r.labels)
    expect([...run.state.labels]).toEqual([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    expect(run.state.merges).toHaveLength(9)
    expect(r.linkage).toHaveLength(9)
    // n - 1 merges, one step per tree node that is still an internal node of the condensed tree, one per cluster, one to label.
    expect(r.condensedTree.length).toBeGreaterThan(0)
    expect(run.steps).toBe(9 + 7 + r.stability.length + 1)
  })

  it('logs the tree growing, the condensing, the selection and the labels by path', () => {
    const rec = recorder()
    runAlgorithm(hdbscanAlgorithm(twoGroups, { minClusterSize: 3 }), {
      maxSteps: 100,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
    })
    const metric = (name: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
    expect(metric('tree/merge')).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9])
    const heights = metric('tree/height')
    expect(heights.every((h, i) => i === 0 || h >= (heights[i - 1] ?? 0))).toBe(true)
    expect(metric('tree/size').at(-1)).toBe(10)
    expect(metric('condense/split').length).toBe(7)
    expect(metric('condense/split').reduce((a, b) => a + b, 0)).toBe(1)
    const phases = metric('hdbscan/phase')
    expect(phases.every((p, i) => i === 0 || p >= (phases[i - 1] ?? 0))).toBe(true)
    expect(phases.at(-1)).toBe(4)
    expect(metric('select/selected').at(-1)).toBe(2)
    expect(metric('labels/clusters')).toEqual([2])
    expect(metric('labels/noise')).toEqual([0])
    const labels = rec.events.find((e) => e.type === 'tensor' && e.name === 'labels/labels')
    expect(labels?.type === 'tensor' ? labels.shape : []).toEqual([10])
    expect(Object.keys(hdbscanContract)).toContain('select/stability')
  })

  it('resumes from a JSON checkpoint in every phase, with an infinite lambda in the rows', () => {
    const alg = hdbscanAlgorithm(twoGroups, { minClusterSize: 2 })
    const whole = runAlgorithm(alg, { maxSteps: 100 })
    expect(whole.state.rows.some((r) => !Number.isFinite(r[2]))).toBe(true)
    for (const cut of [3, 9, 12, whole.steps - 1]) {
      const half = runAlgorithm(alg, { maxSteps: cut }).state
      const stored = viaJson(alg.serialise?.(half) ?? null)
      const rest = runAlgorithm(alg, { maxSteps: 100, start: { state: stored, step: cut, serialised: true } })
      expect([...rest.state.labels]).toEqual([...whole.state.labels])
      expect(rest.state.rows).toEqual(whole.state.rows)
      expect(rest.state.merges).toEqual(whole.state.merges)
      expect(rest.steps).toBe(whole.steps)
    }
  })

  it('never changes a state it was given', () => {
    const alg = hdbscanAlgorithm(twoGroups, { minClusterSize: 3 })
    const early = runAlgorithm(alg, { maxSteps: 4 }).state
    const before = JSON.stringify(alg.serialise?.(early))
    runAlgorithm(alg, { maxSteps: 100, start: { state: early, step: 4 } })
    expect(JSON.stringify(alg.serialise?.(early))).toBe(before)
  })

  it('selects the leaves of the condensed tree when asked, and many more small clusters on blobs', () => {
    const eom = hdbscan(blobData.x, { minClusterSize: 5 })
    const leaf = hdbscan(blobData.x, { minClusterSize: 5, selection: { kind: 'leaf' } })
    expect(Math.max(...leaf.labels) + 1).toBeGreaterThanOrEqual(Math.max(...eom.labels) + 1)
    // Every leaf cluster is a cluster of the condensed tree with no children.
    const parents = new Set(leaf.condensedTree.filter((r) => r[1] >= 45).map((r) => r[0]))
    expect(leaf.selected.every((c) => !parents.has(c))).toBe(true)
    expect(hdbscanSelections.names).toEqual(['eom', 'leaf'])
  })

  it('alpha divides the distances: no merge gets higher, and 1 is the default', () => {
    const plain = hdbscan(twoGroups, { minClusterSize: 3 })
    const same = hdbscan(twoGroups, { minClusterSize: 3, alpha: 1 })
    const wide = hdbscan(twoGroups, { minClusterSize: 3, alpha: 4 })
    expect(same.linkage).toEqual(plain.linkage)
    for (const [i, m] of wide.linkage.entries()) expect(m[2]).toBeLessThanOrEqual((plain.linkage[i]?.[2] ?? 0) + 1e-12)
  })

  it('has the invariants of the method: ordered heights, valid strengths, labels inside selected clusters', () => {
    const r = hdbscan(blobData.x, { minClusterSize: 6 })
    for (const [i, m] of r.linkage.entries())
      expect(m[2]).toBeGreaterThanOrEqual(i === 0 ? 0 : (r.linkage[i - 1]?.[2] ?? 0))
    expect(r.linkage.at(-1)?.[3]).toBe(45)
    for (const p of r.probabilities) expect(p >= 0 && p <= 1).toBe(true)
    expect(Math.max(...r.labels)).toBe(r.selected.length - 1)
    for (const [i, l] of r.labels.entries()) if (l < 0) expect(r.probabilities[i]).toBe(0)
  })

  it('rejects a smallest cluster below 2 and a non-positive alpha', () => {
    expect(() => hdbscanAlgorithm(twoGroups, { minClusterSize: 1 })).toThrow(/minClusterSize/)
    expect(() => hdbscanAlgorithm(twoGroups, { alpha: 0 })).toThrow(/alpha/)
  })
})

describe('hdbscanEstimator', () => {
  it('fits the blobs, shows no labels before the last step, and agrees with the function', () => {
    const f = hdbscanClustering({ minClusterSize: 5 }).fit(blobData)
    expect(f.params.labels).toEqual(hdbscan(blobData.x, { minClusterSize: 5 }).labels)
    const s = hdbscanEstimator.with({ minClusterSize: 5 }).session(blobData, { key: key(0), logs: 'all' })
    s.advance({ steps: 3 })
    expect(s.fittedAt(3).params.labels.every((l) => l === -1)).toBe(true)
    expect(s.fittedAt(3).params.linkage).toHaveLength(3)
  })

  it('exposes the selection rule as a hyperparameter', () => {
    const leaf = hdbscanEstimator.with({ minClusterSize: 5, selection: 'leaf' }).fit(blobData)
    const eom = hdbscanEstimator.with({ minClusterSize: 5 }).fit(blobData)
    expect(Math.max(...leaf.params.labels)).toBeGreaterThanOrEqual(Math.max(...eom.params.labels))
  })
})
