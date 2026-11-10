/**
 * The lab's problem on the main thread: a two-dimensional classification generator of the library, its rows split into
 * training and held-out points, and the rectangle the boundary is drawn over. Light work (a few hundred rows); the
 * heavy computing is in the workers, which receive this as plain data.
 */

import { type ClassificationData, type Generated, generate, trainTestSplit } from 'omai-engine/data'
import { blobs, circles, moons, spirals, xor } from 'omai-engine/datasets'
import { foldIn, type Key, key } from 'omai-engine/random'
import { toFlat } from 'omai-engine/tensor'
import type { DatasetId, Points, Problem, ProblemSettings } from './types.ts'

/** The datasets on offer and what the noise knob means for each. */
export const DATASETS: readonly {
  readonly id: DatasetId
  readonly label: string
  readonly noise: readonly [number, number, number]
}[] = [
  { id: 'moons', label: 'Moons', noise: [0.02, 0.8, 0.2] },
  { id: 'circles', label: 'Circles', noise: [0.01, 0.5, 0.1] },
  { id: 'xor', label: 'XOR', noise: [0.05, 1, 0.3] },
  { id: 'spirals', label: 'Spirals', noise: [0.01, 0.3, 0.05] },
  { id: 'blobs', label: 'Two blobs', noise: [0.1, 3, 1] },
]

const generator = (id: DatasetId, rows: number, noise: number, seed: Key): Generated<ClassificationData> => {
  const options = (knobs: Record<string, number>): { knobs: Record<string, number>; key: Key } => ({ knobs, key: seed })
  switch (id) {
    case 'moons':
      return generate(moons, options({ n: rows, noise }))
    case 'circles':
      return generate(circles, options({ n: rows, noise }))
    case 'xor':
      return generate(xor, options({ n: rows, noise }))
    case 'spirals':
      return generate(spirals, options({ n: rows, noise }))
    case 'blobs':
      return generate(blobs, options({ n: rows, nFeatures: 2, centers: 2, clusterStd: noise, boxSize: 2 }))
  }
}

const pointsOf = (data: ClassificationData): Points => {
  const x = toFlat(data.x)
  const y = toFlat(data.y)
  return {
    x: Array.from({ length: y.length }, (_, i) => [x[2 * i] ?? 0, x[2 * i + 1] ?? 0]),
    y: Array.from(y, (v) => (v > 0.5 ? 1 : 0)),
  }
}

/**
 * Generates the dataset from the seed and splits off 30% of its rows to hold out.
 *
 * @param settings - The dataset, rows, noise and seed.
 * @returns The problem.
 */
export function makeProblem(settings: ProblemSettings): Problem {
  const seed = key(settings.seed)
  const g = generator(settings.dataset, settings.rows, settings.noise, foldIn(seed, 0))
  const split = trainTestSplit(g.data, { testSize: 0.3, key: foldIn(seed, 1) })
  const train = pointsOf(split.train)
  const test = pointsOf(split.test)
  const all = [...train.x, ...test.x]
  const range = (j: number): [number, number] => {
    const v = all.map((r) => r[j] ?? 0)
    const lo = Math.min(...v)
    const hi = Math.max(...v)
    const pad = 0.12 * (hi - lo || 1)
    return [lo - pad, hi + pad]
  }
  return { settings, train, test, extent: { x: range(0), y: range(1) } }
}
