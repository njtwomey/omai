import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { swissRoll } from '../datasets/index.ts'
import { smoothSolvers } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { spearman } from '../stats/index.ts'
import { metricSeries, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { nestedCircles, sphereClusters, twoScaleClusters } from './_planted.ts'
import { pcaStart } from './_support.ts'
import { pacmap, pacmapAdam, pacmapAlgorithm, pacmapLoss, pacmapPairs, pacmapProblem, pacmapWeights } from './pacmap.ts'
import { trustworthiness } from './quality.ts'
import { umap } from './umap.ts'

const line = Array.from({ length: 30 }, (_, i) => [i, ((i * 7) % 5) * 0.01])

describe('PaCMAP pairs', () => {
  it('has n k near pairs, n round(k/2) mid-near pairs and n 2k further pairs, none to itself, none near in the further set', () => {
    const k = 6
    const pairs = pacmapPairs(line, key(0), { nNeighbors: k })
    const n = line.length
    expect(pairs.near.length).toBe(n * k)
    expect(pairs.mid.length).toBe(n * 3)
    expect(pairs.far.length).toBe(n * 12)
    for (const list of [pairs.near, pairs.mid, pairs.far]) for (const [i, j] of list) expect(i).not.toBe(j)
    for (let i = 0; i < n; i++) {
      const near = new Set(pairs.near.filter((p) => p[0] === i).map((p) => p[1]))
      expect(near.size).toBe(k)
      const far = pairs.far.filter((p) => p[0] === i).map((p) => p[1])
      expect(new Set(far).size).toBe(12)
      for (const j of far) expect(near.has(j)).toBe(false)
    }
  })

  it('near pairs are the true neighbours on a line, mid-near pairs are farther than near ones, and the key decides the random pairs', () => {
    const pairs = pacmapPairs(line, key(0), { nNeighbors: 4 })
    // the middle row's four nearest on an evenly spaced line are the two on each side
    const mid = pairs.near
      .filter((p) => p[0] === 15)
      .map((p) => p[1])
      .sort((a, b) => a - b)
    expect(mid).toEqual([13, 14, 16, 17])
    const meanGap = (list: readonly (readonly [number, number])[]): number =>
      list.reduce((a, [i, j]) => a + Math.abs(i - j), 0) / list.length
    expect(meanGap(pairs.mid)).toBeGreaterThan(2 * meanGap(pairs.near))
    expect(meanGap(pairs.far)).toBeGreaterThan(meanGap(pairs.mid) * 0.8)
    expect(pacmapPairs(line, key(0), { nNeighbors: 4 })).toEqual(pairs)
    const other = pacmapPairs(line, key(1), { nNeighbors: 4 })
    expect(other.near).toEqual(pairs.near)
    expect(other.mid).not.toEqual(pairs.mid)
    expect(other.far).not.toEqual(pairs.far)
  })

  it('a mid-near pair is the second closest of six random rows: on 0..7 the end rows pair with gap 2 or 3 only', () => {
    // Six of the seven other rows are drawn. For row 0 the second closest is row 2, or row 3 when row 1 or row 2 was
    // left out: never row 1 (the closest, when drawn), never beyond row 3. Row 7 is the mirror image.
    const x = Array.from({ length: 8 }, (_, i) => [i])
    const pairs = pacmapPairs(x, key(3), { nNeighbors: 1, mnRatio: 40, fpRatio: 1 })
    expect(pairs.mid.length).toBe(8 * 40)
    const partners = (i: number): Set<number> => new Set(pairs.mid.filter((p) => p[0] === i).map((p) => p[1]))
    expect([...partners(0)].sort()).toEqual([2, 3])
    expect([...partners(7)].sort()).toEqual([4, 5])
  })

  it('rejects too few rows', () => {
    expect(() => pacmapPairs(line.slice(0, 5), key(0))).toThrow()
    expect(() => pacmapPairs(line.slice(0, 12), key(0), { nNeighbors: 8 })).toThrow(/rows/)
  })
})

describe('PaCMAP loss and schedule', () => {
  it('the weights follow the three phases', () => {
    expect(pacmapWeights(0)).toEqual({ near: 2, mid: 1000, far: 1 })
    expect(pacmapWeights(50).mid).toBeCloseTo(501.5, 12)
    expect(pacmapWeights(99).mid).toBeCloseTo(0.01 * 1000 + 0.99 * 3, 12)
    expect(pacmapWeights(100)).toEqual({ near: 3, mid: 3, far: 1 })
    expect(pacmapWeights(199)).toEqual({ near: 3, mid: 3, far: 1 })
    expect(pacmapWeights(200)).toEqual({ near: 1, mid: 0, far: 1 })
    expect(pacmapWeights(449)).toEqual({ near: 1, mid: 0, far: 1 })
    expect(pacmapWeights(5, [10, 10, 10], 100)).toEqual({ near: 2, mid: 0.5 * 100 + 0.5 * 3, far: 1 })
  })

  it('equals the hand-computed pair losses on three points, and the gradient matches finite differences', () => {
    // y0 = (0,0), y1 = (1,1), y2 = (3,0): |y0-y1|^2 = 2, |y0-y2|^2 = 9, |y1-y2|^2 = 5
    const y = [0, 0, 1, 1, 3, 0]
    const pairs = {
      near: [[0, 1]] as [number, number][],
      mid: [[0, 2]] as [number, number][],
      far: [[1, 2]] as [number, number][],
    }
    const w = { near: 2, mid: 3, far: 0.5 }
    const handLoss = (2 * 3) / (10 + 3) + (3 * 10) / (10000 + 10) + 0.5 / (1 + 6)
    const { loss, grad } = pacmapLoss(y, 2, pairs, w)
    expect(loss).toBeCloseTo(handLoss, 14)
    // gradient by hand: near coefficient 2 * 20 / 13^2 on (y0 - y1) = (-1, -1) goes to y0 ...
    const cNear = (2 * 20) / 13 ** 2
    const cMid = (3 * 20000) / 10010 ** 2
    const cFar = (-0.5 * 2) / 7 ** 2
    expect(grad[0]).toBeCloseTo(cNear * -1 + cMid * -3, 14)
    expect(grad[1]).toBeCloseTo(cNear * -1 + 0, 14)
    expect(grad[2]).toBeCloseTo(cNear * 1 + cFar * -2, 14)
    expect(grad[3]).toBeCloseTo(cNear * 1 + cFar * 1, 14)
    expect(grad[4]).toBeCloseTo(cMid * 3 + cFar * 2, 14)
    expect(grad[5]).toBeCloseTo(cFar * -1, 14)
    const h = 1e-6
    for (let q = 0; q < y.length; q++) {
      const up = y.map((v, i) => (i === q ? v + h : v))
      const dn = y.map((v, i) => (i === q ? v - h : v))
      const fd = (pacmapLoss(up, 2, pairs, w).loss - pacmapLoss(dn, 2, pairs, w).loss) / (2 * h)
      expect(grad[q]).toBeCloseTo(fd, 8)
    }
  })

  it('a zero weight removes a kind of pair, and the near loss saturates below its weight', () => {
    const pairs = { near: [[0, 1]] as [number, number][], mid: [[0, 1]] as [number, number][], far: [] }
    expect(pacmapLoss([0, 100], 1, pairs, { near: 1, mid: 0, far: 0 }).loss).toBeLessThan(1)
    expect(pacmapLoss([0, 100], 1, pairs, { near: 1, mid: 0, far: 0 }).loss).toBeGreaterThan(0.99)
    expect(pacmapLoss([0, 100], 1, pairs, { near: 0, mid: 0, far: 0 }).loss).toBe(0)
  })
})

describe('PaCMAP algorithm', () => {
  const p = sphereClusters(key(2), 12)
  const pairs = pacmapPairs(p.x, key(0), { nNeighbors: 6 })
  const start = pcaStart(p.x, 2, 0.01)

  it('runs the three phases (the tape records the phase), lowers the final-phase loss and keeps the embedding on the tape', () => {
    const f = pacmap.with({ nNeighbors: 6, phases: [12, 12, 30] }).fit(p, { key: key(0), logs: 'all' })
    const phase = metricSeries(f.run, 'solver/phase').values
    expect(phase.length).toBe(55)
    expect([phase[0], phase[11], phase[12], phase[23], phase[24], phase[54]]).toEqual([1, 1, 2, 2, 3, 3])
    const loss = metricSeries(f.run, 'solver/loss').values
    // within the last phase the weights are fixed, so the loss falls
    expect(loss.at(-1) ?? 0).toBeLessThan(loss[24] ?? 0)
    expect(f.params.embedding.length).toBe(48)
    expect(f.params.objective).toBeCloseTo(loss.at(-1) ?? 0, 12)
  })

  it('is deterministic from a key and the key matters', () => {
    const run = (k: number) =>
      pacmap.with({ nNeighbors: 6, phases: [8, 8, 8] }).fit(p, { key: key(k) }).params.embedding
    expect(run(0)).toEqual(run(0))
    expect(run(1)).not.toEqual(run(0))
  })

  it('serialises its state, and a stored state resumes to the same run', () => {
    const alg = pacmapAlgorithm(pairs, start, { phases: [6, 6, 6] })
    const whole = runAlgorithm(alg, { maxSteps: 18, key: key(0) })
    expect(whole.steps).toBe(18)
    const half = runAlgorithm(alg, { maxSteps: 7, key: key(0) })
    // the optimiser's stateless links hold undefined, which JSON stores as null: the JSON is stable and the layout exact
    const json = stateToJson(half.state)
    expect(stateToJson(stateFromJson(json))).toEqual(json)
    expect((stateFromJson(json) as { y: number[][] }).y).toEqual(half.state.y)
    const resumed = runAlgorithm(alg, {
      maxSteps: 18,
      key: key(0),
      start: { state: stateToJson(half.state), step: 7, serialised: true },
    })
    expect(resumed.state.y).toEqual(whole.state.y)
  })

  it('the optimiser is a slot: Adam is the named default, other optimisers and the smooth solvers also lower the loss', () => {
    expect(pacmapAdam).toMatchObject({ kind: 'adam', eps: 1e-7 })
    const base = pacmapLoss(start.flat(), 2, pairs, pacmapWeights(200)).loss
    const sgdFit = pacmap
      .with({
        nNeighbors: 6,
        phases: [0, 0, 60],
        learningRate: 0.05,
        optimiser: { kind: 'sgd', momentum: 0.9, nesterov: false, weightDecay: 0 },
      })
      .fit(p, { key: key(0) })
    expect(sgdFit.params.objective).toBeLessThan(base)
    // the objective of the last phase as a smooth problem, minimised by L-BFGS from the same start
    const problem = pacmapProblem(pairs, pacmapWeights(200), start)
    const inner = smoothSolvers.make(problem, { kind: 'lbfgs' }, key(0))
    const out = runAlgorithm(inner, { maxSteps: 150, key: key(0) })
    const reached = (out.state as { value: number }).value
    expect(reached).toBeLessThan(base)
    expect(reached).toBeCloseTo(pacmapLoss((out.state as { x: Float64Array }).x, 2, pairs, pacmapWeights(200)).loss, 10)
  })
})

describe('PaCMAP on planted manifolds', () => {
  it('unrolls a swiss roll: trustworthiness above 0.97, beyond the PCA projection', () => {
    const roll = generate(swissRoll, { key: key(1), knobs: { n: 300 } }).data
    const f = pacmap.with({ nNeighbors: 10 }).fit({ x: roll.x }, { key: key(0) })
    const t = trustworthiness(roll.x, f.params.embedding, 10)
    // measured 0.995 (PCA to the plane 0.944); the bar leaves room for the roll's sampling
    expect(t).toBeGreaterThan(0.97)
    expect(t).toBeGreaterThan(trustworthiness(roll.x, pcaStart(roll.x, 2, 1), 10))
  })

  it('keeps four clusters on a sphere apart (trustworthiness and 10-NN label purity of the embedding)', () => {
    const p = sphereClusters(key(2), 30)
    const f = pacmap.with({ nNeighbors: 10 }).fit({ x: p.x }, { key: key(0) })
    expect(trustworthiness(p.x, f.params.embedding, 10)).toBeGreaterThan(0.95)
    const e = f.params.embedding
    let agree = 0
    for (let i = 0; i < e.length; i++) {
      const near = e
        .map((r, j) => ({ j, d: Math.hypot((r[0] ?? 0) - (e[i]?.[0] ?? 0), (r[1] ?? 0) - (e[i]?.[1] ?? 0)) }))
        .filter((v) => v.j !== i)
        .sort((a, b) => a.d - b.d)
        .slice(0, 10)
      agree += near.filter((v) => p.labels[v.j] === p.labels[i]).length / 10
    }
    expect(agree / e.length).toBeGreaterThan(0.95)
  })

  it('keeps two nested circles as two rings: neighbours stay on their own circle, the outer ring is the wider', () => {
    const p = nestedCircles(key(3), 80)
    const f = pacmap.with({ nNeighbors: 10 }).fit({ x: p.x }, { key: key(0) })
    expect(trustworthiness(p.x, f.params.embedding, 10)).toBeGreaterThan(0.97)
    const e = f.params.embedding
    const mean = [0, 1].map((q) => e.reduce((a, r) => a + (r[q] ?? 0), 0) / e.length)
    const radius = e.map((r) => Math.hypot((r[0] ?? 0) - (mean[0] ?? 0), (r[1] ?? 0) - (mean[1] ?? 0)))
    const ring = (c: number): number => {
      const rs = radius.filter((_, i) => p.labels[i] === c)
      return rs.reduce((a, v) => a + v, 0) / rs.length
    }
    expect(ring(1)).toBeGreaterThan(ring(0))
  })

  it('preserves the layout of cluster centres better than UMAP, which the mid-near pairs are for', () => {
    const p = twoScaleClusters(key(4), 20)
    const centreCorrelation = (emb: number[][]): number => {
      const centres = p.centres.map((_, c) => {
        const rows = emb.filter((_, i) => p.labels[i] === c)
        return [0, 1].map((q) => rows.reduce((a, r) => a + (r[q] ?? 0), 0) / rows.length)
      })
      const truth: number[] = []
      const got: number[] = []
      for (let i = 0; i < centres.length; i++)
        for (let j = i + 1; j < centres.length; j++) {
          const d = (a: number[], b: number[]): number => Math.hypot(...a.map((v, q) => v - (b[q] ?? 0)))
          truth.push(d(p.centres[i] as number[], p.centres[j] as number[]))
          got.push(d(centres[i] as number[], centres[j] as number[]))
        }
      return spearman(truth, got).statistic
    }
    const seeds = [0, 1, 2]
    const mean = (v: number[]): number => v.reduce((a, b) => a + b, 0) / v.length
    const pac = mean(
      seeds.map((s) =>
        centreCorrelation(pacmap.with({ nNeighbors: 10 }).fit({ x: p.x }, { key: key(s) }).params.embedding),
      ),
    )
    const uma = mean(
      seeds.map((s) =>
        centreCorrelation(
          umap.with({ nNeighbors: 10, nEpochs: 200 }).fit({ x: p.x }, { key: key(s) }).params.embedding,
        ),
      ),
    )
    // measured over these seeds: PaCMAP 0.70, UMAP -0.09 (its fuzzy graph has no edge between the groups, so their placement is arbitrary)
    expect(pac).toBeGreaterThan(0.5)
    expect(pac).toBeGreaterThan(uma + 0.4)
    const noMid = mean(
      seeds.map((s) =>
        centreCorrelation(
          pacmap.with({ nNeighbors: 10, mnRatio: 0 }).fit({ x: p.x }, { key: key(s) }).params.embedding,
        ),
      ),
    )
    expect(pac).toBeGreaterThan(noMid)
  })
})
