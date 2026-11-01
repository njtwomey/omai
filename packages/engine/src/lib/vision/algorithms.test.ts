import { describe, expect, it } from 'vitest'
import { fromStorage, toArray, toFlat } from '../tensor/index.ts'
import { policies, recorder, runAlgorithm } from '../trace/index.ts'
import {
  floodFillModes,
  meanShiftCriteria,
  seamMaskStrengths,
  simplifications,
  softNmsMethods,
} from './algorithm-options.ts'
import { softNms, softNmsAlgorithm } from './boxes.ts'
import { floodFill, floodFillAlgorithm, floodFillResult } from './components.ts'
import { approxPolyDP, approxPolyDPAlgorithm } from './contours.ts'
import { houghLineAlgorithm, houghLinePeaks, houghPeaksAlgorithm } from './edges.ts'
import { enclosingCircleOf, minEnclosingCircle, minEnclosingCircleAlgorithm } from './fitting.ts'
import { geomContract, greedyContract } from './greedy-contract.ts'
import { seamCarve, seamCarveAlgorithm, seamCarveImage, seamContract } from './seams.ts'
import { regionGrowing, regionGrowingAlgorithm, regionLabels } from './segmentation.ts'
import { camShift, camShiftAlgorithm, meanShift, meanShiftAlgorithm, trackContract } from './tracking.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T
const logPolicy = { ...policies.minimal, metrics: true, detail: 'full' } as const
const metric = (rec: ReturnType<typeof recorder>, name: string): number[] =>
  rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))

describe('seamCarveAlgorithm', () => {
  const img = Array.from({ length: 8 }, (_, i) =>
    Array.from({ length: 10 }, (_, j) => Math.sin(i * 0.7) + ((j * 3 + i) % 5) / 5),
  )

  it('removes one seam per step, narrowing then shortening, and equals seamCarve', () => {
    const run = runAlgorithm(seamCarveAlgorithm(img, [6, 7]), { maxSteps: 20 })
    expect([run.status, run.steps]).toEqual(['converged', 3 + 2])
    const out = seamCarveImage(run.state)
    expect(out.shape).toEqual([6, 7])
    expect(toArray(seamCarve(img, [6, 7]))).toEqual(toArray(out))
  })

  it('keeps a protected region and removes a marked one, resumes from JSON and logs the seams', () => {
    const protect = img.map((row) => row.map((_, j) => (j === 4 ? 1 : 0)))
    const kept = seamCarve(
      img.map((row, i) => row.map((v, j) => (j === 4 ? 100 + i : v))),
      [8, 8],
      { protect },
    )
    expect(toFlat(kept).filter((v) => v >= 100)).toHaveLength(8)
    const remove = img.map((row) => row.map((_, j) => (j === 2 ? 1 : 0)))
    const marked = img.map((row, i) => row.map((v, j) => (j === 2 ? -100 - i : v)))
    expect(toFlat(seamCarve(marked, [8, 9], { remove })).filter((v) => v <= -100)).toHaveLength(0)
    const alg = seamCarveAlgorithm(img, [6, 7])
    const whole = runAlgorithm(alg, { maxSteps: 20 })
    const part = runAlgorithm(alg, { maxSteps: 2 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 20,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 2, serialised: true },
    })
    expect(toArray(seamCarveImage(rest.state))).toEqual(toArray(seamCarveImage(whole.state)))
    const rec = recorder()
    runAlgorithm(alg, { maxSteps: 20, tracer: rec, policy: logPolicy })
    expect(metric(rec, 'seam/width')).toEqual([9, 8, 7, 7, 6])
    expect(metric(rec, 'seam/cost').every((c) => c >= 0)).toBe(true)
    expect(Object.keys(seamContract)).toHaveLength(4)
    expect(seamMaskStrengths.make({ kind: 'gentle' }).bias).toBe(1000)
    expect(() => seamCarveAlgorithm(img, [9, 7])).toThrow(/target/)
  })
})

describe('meanShiftAlgorithm and camShiftAlgorithm', () => {
  const prob = Array.from({ length: 30 }, (_, i) =>
    Array.from({ length: 30 }, (_, j) => (Math.abs(i - 15) < 3 && Math.abs(j - 12) < 7 ? 1 : 0)),
  )

  it('moves the window one update per step towards the mass, as meanShift does', () => {
    const run = runAlgorithm(meanShiftAlgorithm(prob, [2, 8, 8, 8]), { maxSteps: 100 })
    expect(run.status).toBe('converged')
    const direct = meanShift(prob, [2, 8, 8, 8])
    expect(run.state.window).toEqual(direct.window)
    expect(run.state.iterations).toBe(direct.iterations)
    const rec = recorder()
    runAlgorithm(meanShiftAlgorithm(prob, [2, 8, 8, 8]), { maxSteps: 100, tracer: rec, policy: logPolicy })
    // The mass under the window never falls as it climbs.
    const mass = metric(rec, 'track/moment')
    expect(mass.at(-1) ?? 0).toBeGreaterThanOrEqual(mass[0] ?? 1)
    expect(meanShiftCriteria.make({ kind: 'tight' })).toEqual({ maxIter: 500, eps: 0.5 })
    expect(Object.keys(trackContract)).toHaveLength(4)
    expect(() => meanShiftAlgorithm(prob, [0, 0, 0, 5])).toThrow(/non-positive/)
  })

  it('adds the CamShift box in a last step and resumes from JSON', () => {
    const alg = camShiftAlgorithm(prob, [8, 10, 8, 8])
    const whole = runAlgorithm(alg, { maxSteps: 100 })
    expect(whole.status).toBe('converged')
    expect(whole.state.box?.size.map((v) => Math.round(v))).toEqual([6, 15])
    expect(camShift(prob, [8, 10, 8, 8]).box).toEqual(whole.state.box)
    const part = runAlgorithm(alg, { maxSteps: 1 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 100,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 1, serialised: true },
    })
    expect(rest.state).toEqual(whole.state)
  })
})

describe('softNmsAlgorithm', () => {
  const boxes = [
    [0, 0, 10, 10],
    [1, 1, 11, 11],
    [0, 0, 10, 5],
    [20, 20, 30, 30],
  ]
  const scores = [0.9, 0.85, 0.8, 0.7]

  it('keeps one box per step, decaying the others, as softNms does', () => {
    for (const method of ['gaussian', 'linear'] as const) {
      const run = runAlgorithm(softNmsAlgorithm(boxes, scores, { method }), { maxSteps: 20 })
      const direct = softNms(boxes, scores, { method })
      expect(run.state.indices).toEqual(direct.indices)
      expect(run.state.kept).toEqual(direct.scores)
      expect(run.state.indices[0]).toBe(0)
      // Kept scores come out no larger than the originals.
      run.state.indices.forEach((i, k) => expect(run.state.kept[k] ?? 9).toBeLessThanOrEqual((scores[i] ?? 0) + 1e-6))
    }
  })

  it('stops at topK or the score threshold, resumes and logs', () => {
    expect(softNms(boxes, scores, { topK: 2 }).indices).toHaveLength(2)
    expect(softNms(boxes, scores, { method: 'linear', scoreThreshold: 0.6 }).indices).toEqual([0, 3])
    const alg = softNmsAlgorithm(boxes, scores)
    const whole = runAlgorithm(alg, { maxSteps: 20 })
    const part = runAlgorithm(alg, { maxSteps: 1 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 20,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 1, serialised: true },
    })
    expect(rest.state.indices).toEqual(whole.state.indices)
    const rec = recorder()
    runAlgorithm(alg, { maxSteps: 20, tracer: rec, policy: logPolicy })
    expect(metric(rec, 'greedy/pick')).toEqual(whole.state.indices)
    expect(Object.keys(greedyContract)).toHaveLength(7)
    expect(softNmsMethods.make({ kind: 'linear' }).method).toBe('linear')
    expect(() => softNmsAlgorithm(boxes, [1, 2])).toThrow(/one score/)
  })
})

describe('houghPeaksAlgorithm', () => {
  it('takes one peak per step, strongest first, and equals houghLinePeaks', () => {
    const edges = Array.from({ length: 24 }, (_, i) =>
      Array.from({ length: 24 }, (_, j) => (j === 5 || i === 15 ? 1 : 0)),
    )
    const { algorithm, geometry } = houghLineAlgorithm(edges)
    const acc = fromStorage(
      runAlgorithm(algorithm, { maxSteps: 200 }).state.accumulator,
      [...geometry.shape],
      'float64',
    )
    const run = runAlgorithm(houghPeaksAlgorithm(acc, geometry), { maxSteps: 20 })
    expect(run.status).toBe('converged')
    expect(run.state.peaks).toEqual(houghLinePeaks(acc, geometry))
    expect(run.state.peaks.length).toBeGreaterThanOrEqual(2)
    const votes = run.state.peaks.map((p) => p.votes)
    expect(votes).toEqual([...votes].sort((a, b) => b - a))
    const capped = runAlgorithm(houghPeaksAlgorithm(acc, geometry, { numPeaks: 1 }), { maxSteps: 20 })
    expect([capped.status, capped.state.peaks.length]).toEqual(['completed', 1])
  })
})

describe('regionGrowingAlgorithm', () => {
  const img = [
    [0.1, 0.12, 0.9, 0.88],
    [0.11, 0.1, 0.92, 0.9],
    [0.1, 0.5, 0.9, 0.91],
  ]

  it('absorbs one pixel per step into the nearest region and equals regionGrowing', () => {
    const seeds: [number, number][] = [
      [0, 0],
      [3, 0],
    ]
    const run = runAlgorithm(regionGrowingAlgorithm(img, seeds), { maxSteps: 100 })
    expect(run.status).toBe('converged')
    expect(toArray(regionLabels(run.state))).toEqual(toArray(regionGrowing(img, seeds)))
    // The unreached pixel (0.5, far from both means) stays unlabelled.
    expect(toFlat(regionLabels(run.state))[2 * 4 + 1]).toBe(0)
    const alg = regionGrowingAlgorithm(img, seeds)
    const part = runAlgorithm(alg, { maxSteps: 3 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 100,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 3, serialised: true },
    })
    expect(Array.from(rest.state.labels)).toEqual(Array.from(run.state.labels))
    expect(() => regionGrowingAlgorithm(img, [[9, 9]]).init(undefined as never)).toThrow(/outside/)
  })
})

describe('floodFillAlgorithm', () => {
  const img = [
    [0, 0, 1, 1],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
  ]

  it('scans one span per step, painting as it goes, and equals floodFill', () => {
    const run = runAlgorithm(floodFillAlgorithm(img, [0, 0], 0.5), { maxSteps: 20 })
    expect(run.status).toBe('converged')
    const r = floodFillResult(run.state)
    const direct = floodFill(img, [0, 0], 0.5)
    expect(toArray(r.image)).toEqual(toArray(direct.image))
    expect(r.area).toBe(direct.area)
    expect(r.rect).toEqual(direct.rect)
    expect(r.area).toBe(8)
    const rec = recorder()
    runAlgorithm(floodFillAlgorithm(img, [0, 0], 0.5), { maxSteps: 20, tracer: rec, policy: logPolicy })
    expect(metric(rec, 'fill/area').at(-1)).toBe(8)
    const eight = floodFill(img, [0, 0], 0.5, { connectivity: 8 })
    expect(eight.area).toBeGreaterThanOrEqual(direct.area)
    expect(floodFillModes.make({ kind: 'fixed' }).fixedRange).toBe(true)
  })

  it('serialises mid-fill and resumes', () => {
    const alg = floodFillAlgorithm(img, [0, 0], 0.5)
    const whole = runAlgorithm(alg, { maxSteps: 20 })
    const part = runAlgorithm(alg, { maxSteps: 1 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 20,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 1, serialised: true },
    })
    expect(Array.from(rest.state.filled)).toEqual(Array.from(whole.state.filled))
    expect(Array.from(rest.state.data)).toEqual(Array.from(whole.state.data))
    expect(() => floodFillAlgorithm(img, [9, 9], 1)).toThrow(/seed/)
  })
})

describe('minEnclosingCircleAlgorithm', () => {
  const pts = [
    [0, 0],
    [4, 0],
    [2, 1],
    [1, 3],
    [3, -2],
    [2, 0.5],
  ]

  it('adds one point per step and ends on a circle that holds them all', () => {
    const run = runAlgorithm(minEnclosingCircleAlgorithm(pts), { maxSteps: 20 })
    expect([run.status, run.steps]).toEqual(['converged', pts.length - 2])
    const { center, radius } = enclosingCircleOf(run.state)
    expect(minEnclosingCircle(pts)).toEqual({ center, radius })
    for (const p of pts) expect(Math.hypot((p[0] ?? 0) - center[0], (p[1] ?? 0) - center[1])).toBeLessThan(radius)
    // The radius never shrinks as points are added.
    const rec = recorder()
    runAlgorithm(minEnclosingCircleAlgorithm(pts), {
      maxSteps: 20,
      tracer: rec,
      policy: { ...policies.minimal, tensors: true, roles: ['output', 'state'], detail: 'full' },
    })
    expect(minEnclosingCircle([]).radius).toBe(0)
    expect(minEnclosingCircle([[2, 3]])).toEqual({ center: [2, 3], radius: 1e-4 })
  })

  it('serialises and resumes', () => {
    const alg = minEnclosingCircleAlgorithm(pts)
    const whole = runAlgorithm(alg, { maxSteps: 20 })
    const part = runAlgorithm(alg, { maxSteps: 1 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 20,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 1, serialised: true },
    })
    expect(rest.state).toEqual(whole.state)
  })
})

describe('approxPolyDPAlgorithm', () => {
  const curve = [
    [0, 0],
    [1, 0.1],
    [2, 0],
    [3, 0.1],
    [4, 0],
    [4, 2],
    [2, 2.1],
    [0, 2],
  ]

  it('splits one segment per step, within epsilon, and equals approxPolyDP', () => {
    for (const closed of [true, false]) {
      const run = runAlgorithm(approxPolyDPAlgorithm(curve, 0.5, closed), { maxSteps: 100 })
      expect(run.status).toBe('converged')
      expect(run.state.result).toEqual(approxPolyDP(curve, 0.5, closed))
      const kept = approxPolyDP(curve, 0.5, closed)
      expect(kept.length).toBeLessThan(curve.length)
    }
    expect(
      approxPolyDP(
        [
          [0, 0],
          [1, 0.1],
          [2, 0],
          [2, 2],
          [0, 2],
        ],
        0.5,
        true,
      ),
    ).toEqual([
      [0, 0],
      [2, 0],
      [2, 2],
      [0, 2],
    ])
    expect(approxPolyDP([], 1, true)).toEqual([])
    expect(runAlgorithm(approxPolyDPAlgorithm([], 1, true), { maxSteps: 5 }).steps).toBe(0)
  })

  it('serialises, resumes and logs', () => {
    const alg = approxPolyDPAlgorithm(curve, 0.5, true)
    const whole = runAlgorithm(alg, { maxSteps: 100 })
    const part = runAlgorithm(alg, { maxSteps: 1 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 100,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 1, serialised: true },
    })
    expect(rest.state.result).toEqual(whole.state.result)
    const rec = recorder()
    runAlgorithm(alg, { maxSteps: 100, tracer: rec, policy: logPolicy })
    expect(metric(rec, 'geom/points').length).toBe(whole.steps)
    expect(Object.keys(geomContract)).toHaveLength(3)
    expect(simplifications.make({ kind: 'polyline', epsilon: 2 })).toEqual({ epsilon: 2, closed: false })
  })
})
