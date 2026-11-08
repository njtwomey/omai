import { pipelineStages } from 'omai-engine/exposition'
import { eigh } from 'omai-engine/numerics/linalg'
import { tensor, toArray } from 'omai-engine/tensor'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { gaussianRows } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Pipeline stages',
  question: 'How do I show what each step of a pipeline did to the data and what it learned?',
  explain:
    'A `pipelineStages` record lists the stages of a pipeline (a transformer, a selector, an estimator), each with the columns it took and gave, a sample of the rows before and after, its shape change and what it learned. `PipelineView` draws the stages as a chain of boxes with the shapes on the arrows and the learned values in each box. This pipeline standardises four measurements in very different units, keeps the leading principal components and then assigns each row to the nearest of two class centres. More components keep more of the variance and the box says how much.',
  shows: ['pipelineStages', 'PipelineView'],
}

const COLS = ['height cm', 'weight kg', 'age y', 'income k']
const SCALE = [
  [170, 10],
  [70, 12],
  [40, 14],
  [50, 20],
] as const

export default function PipelineStagesRecipe(): ReactNode {
  const [keep, setKeep] = useState(2)
  const [seed, setSeed] = useState(1)
  const record = useMemo(() => {
    const n = 100
    const z = gaussianRows(seed, n, 4)
    const raw = z.map((r) =>
      r.map((v, j) => (SCALE[j]?.[0] ?? 0) + (SCALE[j]?.[1] ?? 0) * (v + (j === 1 ? 0.8 * (r[0] ?? 0) : 0))),
    )
    const mean = COLS.map((_, j) => raw.reduce((s, r) => s + (r[j] ?? 0), 0) / n)
    const sd = COLS.map((_, j) => Math.sqrt(raw.reduce((s, r) => s + ((r[j] ?? 0) - (mean[j] ?? 0)) ** 2, 0) / n))
    const std = raw.map((r) => r.map((v, j) => (v - (mean[j] ?? 0)) / (sd[j] || 1)))
    const cov = COLS.map((_, i) => COLS.map((_, j) => std.reduce((s, r) => s + (r[i] ?? 0) * (r[j] ?? 0), 0) / n))
    const e = eigh(tensor(cov))
    const vals = (toArray(e.eigenvalues) as number[]).slice().reverse()
    const vecs = toArray(e.eigenvectors) as number[][]
    const comps = Array.from({ length: keep }, (_, c) => COLS.map((_, j) => vecs[j]?.[4 - 1 - c] ?? 0))
    const proj = std.map((r) => comps.map((c) => c.reduce((s, w, j) => s + w * (r[j] ?? 0), 0)))
    const total = vals.reduce((a, b) => a + b, 0)
    const pcNames = comps.map((_, c) => `pc${c + 1}`)
    const groups = proj.map((p) => ((p[0] ?? 0) > 0 ? 1 : 0))
    const centre = [0, 1].map((g) =>
      pcNames.map((_, c) => {
        const mine = proj.filter((_, i) => groups[i] === g)
        return mine.reduce((s, p) => s + (p[c] ?? 0), 0) / Math.max(1, mine.length)
      }),
    )
    const sample = (rows: readonly (readonly number[])[]): number[][] => rows.slice(0, 5).map((r) => [...r])
    const shapes = (rows: number[][] | readonly (readonly number[])[], cols: number): [number, number] => [
      rows.length,
      cols,
    ]
    return pipelineStages({
      classes: ['group 0', 'group 1'],
      input: { columns: COLS, rows: sample(raw), n },
      stages: [
        {
          name: 'standardise',
          kind: 'transformer',
          inputColumns: COLS,
          outputColumns: COLS,
          inputSample: sample(raw),
          outputSample: sample(std),
          inputShape: shapes(raw, 4),
          outputShape: shapes(std, 4),
          output: { columns: COLS, rows: sample(std), n },
          learned: [
            { name: 'mean', value: mean },
            { name: 'std', value: sd },
          ],
        },
        {
          name: 'pca',
          kind: 'transformer',
          inputColumns: COLS,
          outputColumns: pcNames,
          inputSample: sample(std),
          outputSample: sample(proj),
          inputShape: shapes(std, 4),
          outputShape: shapes(proj, keep),
          output: { columns: pcNames, rows: sample(proj), n },
          learned: [{ name: 'variance kept', value: vals.slice(0, keep).reduce((a, b) => a + b, 0) / total }],
          explainedVarianceRatio: vals.slice(0, keep).map((v) => v / total),
        },
        {
          name: 'nearest centre',
          kind: 'estimator',
          inputColumns: pcNames,
          outputColumns: ['group'],
          inputSample: sample(proj),
          outputSample: sample(groups.map((g) => [g])),
          inputShape: shapes(proj, keep),
          outputShape: [groups.length, 1],
          output: { columns: ['group'], rows: sample(groups.map((g) => [g])), n },
          learned: [{ name: 'centres', value: centre.flat() }],
        },
      ],
    })
  }, [keep, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="components kept" value={keep} onChange={setKeep} min={1} max={4} integer />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView record={record} title="Three stages and what each learned" />
    </div>
  )
}
