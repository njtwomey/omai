/**
 * Planted layouts for the tests of this folder, each with the truth it was built from: clusters on a sphere, two nested
 * circles lifted into more dimensions, and clusters at two scales (groups far apart, clusters near within a group), whose
 * true centre distances are what a layout that keeps global structure should reproduce.
 */

import { foldIn, type Key, normal, uniform } from '../random/index.ts'
import { toArray } from '../tensor/index.ts'

/** A planted point set: the rows, a label per row and the truth of the layout. */
export interface Planted {
  readonly x: number[][]
  readonly labels: number[]
  /** The cluster centres, when the layout has them (the rows are drawn around them). */
  readonly centres: number[][]
}

/**
 * Clusters on the unit sphere in 3 dimensions (the vertices of a regular tetrahedron, each with Gaussian scatter
 * projected back onto the sphere), padded with `pad` small-noise dimensions.
 */
export function sphereClusters(key: Key, perCluster: number, pad = 3): Planted {
  const s = 1 / Math.sqrt(3)
  const centres = [
    [s, s, s],
    [s, -s, -s],
    [-s, s, -s],
    [-s, -s, s],
  ]
  const g = toArray(normal(foldIn(key, 0), [centres.length * perCluster, 3 + pad])) as number[][]
  const x: number[][] = []
  const labels: number[] = []
  for (const [c, centre] of centres.entries())
    for (let i = 0; i < perCluster; i++) {
      const row = g[c * perCluster + i] as number[]
      const p = centre.map((v, q) => v + 0.2 * (row[q] ?? 0))
      const norm = Math.hypot(...p)
      x.push([...p.map((v) => v / norm), ...row.slice(3).map((v) => 0.02 * v)])
      labels.push(c)
    }
  return { x, labels, centres }
}

/** Two concentric circles of radii 1 and 3 in the plane, noise 0.05, lifted into `pad` more dimensions of noise 0.02. */
export function nestedCircles(key: Key, perCircle: number, pad = 2): Planted {
  const u = toArray(uniform(foldIn(key, 0), [2 * perCircle])) as number[]
  const g = toArray(normal(foldIn(key, 1), [2 * perCircle, 2 + pad])) as number[][]
  const x: number[][] = []
  const labels: number[] = []
  for (const [c, r] of [1, 3].entries())
    for (let i = 0; i < perCircle; i++) {
      const a = 2 * Math.PI * (u[c * perCircle + i] ?? 0)
      const row = g[c * perCircle + i] as number[]
      x.push([
        r * Math.cos(a) + 0.05 * (row[0] ?? 0),
        r * Math.sin(a) + 0.05 * (row[1] ?? 0),
        ...row.slice(2).map((v) => 0.02 * v),
      ])
      labels.push(c)
    }
  return {
    x,
    labels,
    centres: [
      [0, 0],
      [0, 0],
    ],
  }
}

/**
 * Six Gaussian clusters (standard deviation 0.5) in 10 dimensions at two scales: two groups of three clusters whose
 * centres are 40 apart, the three clusters of a group at the corners of an equilateral triangle of side 6. The true
 * distances between the cluster centres are `centres`' pairwise distances: 6 within a group and about 40 between.
 */
export function twoScaleClusters(key: Key, perCluster: number): Planted {
  const tri = [
    [0, 0],
    [6, 0],
    [3, 3 * Math.sqrt(3)],
  ]
  const centres: number[][] = []
  for (const group of [0, 1])
    for (const [a = 0, b = 0] of tri) centres.push([a + 40 * group, b, group === 0 ? 0 : 5, 0, 0, 0, 0, 0, 0, 0])
  const g = toArray(normal(foldIn(key, 0), [centres.length * perCluster, 10])) as number[][]
  const x: number[][] = []
  const labels: number[] = []
  for (const [c, centre] of centres.entries())
    for (let i = 0; i < perCluster; i++) {
      x.push(centre.map((v, q) => v + 0.5 * (g[c * perCluster + i]?.[q] ?? 0)))
      labels.push(c)
    }
  return { x, labels, centres }
}
