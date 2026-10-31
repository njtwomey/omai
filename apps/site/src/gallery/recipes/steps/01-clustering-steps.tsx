import { clusteringSteps } from 'mdml-engine/exposition'
import { ControlPanel, ExpositionView, NumberField, SeedInput, Select } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { blobs } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Clustering steps',
  question: 'How do I show a clustering algorithm at work: assignments and centres moving, merges, reachability?',
  explain:
    'A `clusteringSteps` record has an assignment of every row at each step, and what the algorithm kept besides: centres (k-means), the merges of a hierarchy, or a reachability ordering. `ExpositionView` picks the view from what is there: `ClusteringPlayer` plays the assignments and centres with the objective beside it, `DendrogramView` draws the merge tree, `ReachabilityView` the reachability plot whose valleys are clusters. All three run on the same blobs; raise the spread and the blobs merge, and k-means with too few or too many clusters shows what it does when the guess is wrong.',
  shows: ['clusteringSteps', 'ClusteringPlayer', 'DendrogramView', 'ReachabilityView', 'ExpositionView'],
}

type Algorithm = 'k-means' | 'hierarchy' | 'reachability'
const d = (a: readonly number[], b: readonly number[]): number =>
  Math.hypot((a[0] ?? 0) - (b[0] ?? 0), (a[1] ?? 0) - (b[1] ?? 0))

export default function ClusteringStepsRecipe(): ReactNode {
  const [algorithm, setAlgorithm] = useState<Algorithm>('k-means')
  const [k, setK] = useState(3)
  const [spread, setSpread] = useState(0.7)
  const [seed, setSeed] = useState(1)
  const { record, points } = useMemo(() => {
    const { points: pts } = blobs(seed, 3, 20, spread)
    const rows = { points: pts, features: ['x', 'y'] as [string, string] }
    if (algorithm === 'k-means') {
      let centres = pts.slice(0, k).map((p) => [...p])
      const assignments: number[][] = []
      const allCentres: number[][][] = []
      const objective: number[] = []
      for (let t = 0; t < 12; t++) {
        const a = pts.map((p) => centres.reduce((best, c, j) => (d(p, c) < d(p, centres[best] ?? c) ? j : best), 0))
        assignments.push(a)
        allCentres.push(centres.map((c) => [...c]))
        objective.push(pts.reduce((s, p, i) => s + d(p, centres[a[i] ?? 0] ?? p) ** 2, 0))
        centres = centres.map((c, j) => {
          const mine = pts.filter((_, i) => a[i] === j)
          return mine.length === 0 ? c : [0, 1].map((q) => mine.reduce((s, p) => s + (p[q] ?? 0), 0) / mine.length)
        })
      }
      return {
        points: pts,
        record: clusteringSteps({ method: 'k-means', assignments, centres: allCentres, objective, rows }),
      }
    }
    if (algorithm === 'hierarchy') {
      // Average-linkage merges: [left, right, distance, size], clusters numbered from n upwards.
      const n = pts.length
      const members = new Map<number, number[]>(pts.map((_, i) => [i, [i]]))
      const merges: [number, number, number, number][] = []
      let next = n
      while (members.size > 1) {
        let best: [number, number, number] = [-1, -1, Number.POSITIVE_INFINITY]
        const ids = [...members.keys()]
        for (const [x, a] of ids.entries())
          for (const b of ids.slice(x + 1)) {
            const ma = members.get(a) ?? []
            const mb = members.get(b) ?? []
            const link =
              ma.reduce((s, i) => s + mb.reduce((u, j) => u + d(pts[i] ?? [], pts[j] ?? []), 0), 0) /
              (ma.length * mb.length)
            if (link < best[2]) best = [a, b, link]
          }
        const joined = [...(members.get(best[0]) ?? []), ...(members.get(best[1]) ?? [])]
        members.delete(best[0])
        members.delete(best[1])
        members.set(next, joined)
        merges.push([best[0], best[1], best[2], joined.length])
        next++
      }
      // The assignment at the cut with k clusters: undo the last k - 1 merges.
      const groups = new Map<number, number[]>(pts.map((_, i) => [i, [i]]))
      merges.slice(0, n - k).forEach(([a, b], m) => {
        groups.set(n + m, [...(groups.get(a) ?? []), ...(groups.get(b) ?? [])])
        groups.delete(a)
        groups.delete(b)
      })
      const assign = new Array<number>(n).fill(0)
      ;[...groups.values()].forEach((g, c) => {
        for (const i of g) assign[i] = c
      })
      return {
        points: pts,
        record: clusteringSteps({ method: 'average linkage', assignments: [assign], merges, rows }),
      }
    }
    // Reachability: walk from row 0 to the nearest unvisited row; the plot's heights are the steps' lengths.
    const order = [0]
    const reach = [Number.POSITIVE_INFINITY]
    const left = new Set(pts.map((_, i) => i).slice(1))
    while (left.size > 0) {
      let bestJ = -1
      let bestD = Number.POSITIVE_INFINITY
      for (const j of left)
        for (const i of order) {
          const dist = d(pts[i] ?? [], pts[j] ?? [])
          if (dist < bestD) [bestJ, bestD] = [j, dist]
        }
      order.push(bestJ)
      reach.push(bestD)
      left.delete(bestJ)
    }
    const byRow = new Array<number>(pts.length).fill(0)
    order.forEach((row, pos) => {
      byRow[row] = Number.isFinite(reach[pos]) ? (reach[pos] ?? 0) : Math.max(...reach.filter(Number.isFinite))
    })
    return {
      points: pts,
      record: clusteringSteps({
        method: 'reachability ordering',
        assignments: [pts.map(() => 0)],
        reachability: order.map((r) => byRow[r] ?? 0),
        ordering: order,
        rows,
      }),
    }
  }, [algorithm, k, spread, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={4}>
        <Select
          label="algorithm"
          value={algorithm}
          options={['k-means', 'hierarchy', 'reachability'] as const}
          onChange={setAlgorithm}
        />
        <NumberField label="clusters k" value={k} onChange={setK} min={1} max={8} integer />
        <NumberField label="blob spread" value={spread} onChange={setSpread} min={0.3} max={2} step={0.1} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView record={record} points={points} title={`Clustering by ${algorithm}`} />
    </div>
  )
}
