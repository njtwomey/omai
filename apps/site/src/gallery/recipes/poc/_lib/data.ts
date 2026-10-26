// Shared by the proof-of-concept methods: two-dimensional two-class datasets chosen by name and generated from a key,
// a grid over their bounding box for decision surfaces, and the leaves of a parameter tree by dotted path.
import { generate } from 'mdml-engine/data'
import { circles, moons, xor } from 'mdml-engine/datasets'
import type { Key } from 'mdml-engine/random'
import { isTensor, toFlat, type Value } from 'mdml-engine/tensor'

export const DATASETS = ['moons', 'circles', 'xor'] as const
export type DatasetName = (typeof DATASETS)[number]
const GENERATORS = { moons, circles, xor } as const

/** Rows in the plane with class indices, and a padded bounding box. */
export interface TwoClass {
  readonly x: readonly (readonly [number, number])[]
  readonly y: readonly number[]
  readonly k: number
  readonly bounds: readonly [readonly [number, number], readonly [number, number]]
}

/** A named dataset of `n` rows, generated from the key. */
export function twoClassData(name: DatasetName, n: number, key: Key): TwoClass {
  const g = generate(GENERATORS[name], { knobs: { n }, key })
  const d = g.data.x.shape[1] ?? 2
  const flat = toFlat(g.data.x)
  const x = Array.from({ length: n }, (_, i): readonly [number, number] => [flat[i * d] ?? 0, flat[i * d + 1] ?? 0])
  const y = toFlat(g.data.y)
  const pad = (lo: number, hi: number): readonly [number, number] => [lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo)]
  const xs = x.map((r) => r[0])
  const ys = x.map((r) => r[1])
  return {
    x,
    y,
    k: Math.max(...y) + 1,
    bounds: [pad(Math.min(...xs), Math.max(...xs)), pad(Math.min(...ys), Math.max(...ys))],
  }
}

/** A grid of `resolution` by `resolution` points over bounds, row by row (one row per y). */
export function gridOver(
  bounds: TwoClass['bounds'],
  resolution: number,
): { readonly xs: number[]; readonly ys: number[]; readonly points: [number, number][] } {
  const axis = ([lo, hi]: readonly [number, number]): number[] =>
    Array.from({ length: resolution }, (_, i) => lo + ((hi - lo) * i) / (resolution - 1))
  const xs = axis(bounds[0])
  const ys = axis(bounds[1])
  const points: [number, number][] = []
  for (const y of ys) for (const x of xs) points.push([x, y])
  return { xs, ys, points }
}

/** Cut a flat list into rows of `width`. */
export function rowsOf<T>(flat: readonly T[], width: number): T[][] {
  const out: T[][] = []
  for (let i = 0; i < flat.length; i += width) out.push(flat.slice(i, i + width))
  return out
}

/** The tensor leaves of a parameter tree, by dotted path. */
export function leaves(tree: unknown, prefix = ''): [string, Value][] {
  if (isTensor(tree)) return [[prefix, tree]]
  if (Array.isArray(tree)) return tree.flatMap((v, i) => leaves(v, prefix === '' ? String(i) : `${prefix}.${i}`))
  if (typeof tree === 'object' && tree !== null)
    return Object.entries(tree as Record<string, unknown>).flatMap(([k, v]) =>
      leaves(v, prefix === '' ? k : `${prefix}.${k}`),
    )
  return []
}
