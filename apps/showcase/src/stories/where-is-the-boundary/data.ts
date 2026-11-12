/**
 * The data of 'Where is the boundary?': forty points in the plane that belong to one of two classes, drawn from the library's
 * two-moons generator (two interleaved half circles with noise), with the inputs rescaled to a mean of zero and a standard
 * deviation of one on each axis. The points are the same every time, so the page can draw them before anything runs; the run's
 * seed only decides the order in which the points are handed to the classifier.
 */

import { generate } from 'omai-engine/data'
import { moons } from 'omai-engine/datasets'
import { foldIn, key, uniform } from 'omai-engine/random'
import { argsort, toFlat } from 'omai-engine/tensor'

/** The points of the two classes. */
export interface Points {
  /** The two coordinates of each point. */
  readonly x: readonly (readonly [number, number])[]
  /** The class of each point: 0 or 1. */
  readonly y: readonly number[]
}

/** The number of points, how much noise the moons have, and the seed they are drawn from. */
export const POINTS = 40
export const NOISE = 0.3
export const POINTS_SEED = 1

/** The part of the plane the pictures show (the rescaled inputs lie well inside it). */
export const EXTENT = {
  x: [-2.6, 2.6],
  y: [-2.6, 2.6],
} as const

/**
 * The points everyone sees.
 *
 * @param n - How many points.
 * @returns The points, with inputs rescaled to a mean of zero and a standard deviation of one.
 */
export function drawPoints(n: number = POINTS): Points {
  const { data } = generate(moons, { knobs: { n, noise: NOISE }, key: foldIn(key(POINTS_SEED), 0) })
  const flat = toFlat(data.x)
  const labels = toFlat(data.y)
  const rows = labels.map((_, i): [number, number] => [flat[2 * i] as number, flat[2 * i + 1] as number])
  const mean = (j: 0 | 1): number => rows.reduce((s, r) => s + r[j], 0) / rows.length
  const sd = (j: 0 | 1): number => Math.sqrt(rows.reduce((s, r) => s + (r[j] - mean(j)) ** 2, 0) / rows.length)
  const [m0, m1, s0, s1] = [mean(0), mean(1), sd(0), sd(1)]
  return {
    x: rows.map((r): [number, number] => [(r[0] - m0) / s0, (r[1] - m1) / s1]),
    y: labels.map((v) => (v > 0.5 ? 1 : 0)),
  }
}

/**
 * The same points in another order, drawn from a seed: the order the classifier takes them in.
 *
 * @param points - The points.
 * @param seed - Decides the order.
 * @returns The points, shuffled.
 */
export function shuffled(points: Points, seed: number): Points {
  const order = toFlat(argsort(uniform(foldIn(key(seed), 5), [points.y.length]), 0))
  return { x: order.map((i) => points.x[i] as readonly [number, number]), y: order.map((i) => points.y[i] as number) }
}
