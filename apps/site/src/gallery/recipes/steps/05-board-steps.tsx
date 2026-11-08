import { boardSteps } from 'omai-engine/exposition'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { uniforms } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Board steps',
  question: 'How do I show a grid of cells changing step by step, with a number tracked beside it?',
  explain:
    'A `boardSteps` record is a layout (a grid, or a graph of nodes), a list of categories and, per step, the category of every cell with a caption, plus metrics over the steps. `BoardPlayer` plays the board with the metrics as curves that follow the step. The board here is Schelling’s model of segregation: two kinds of neighbour on a grid with some empty cells, where anyone with too few of their own kind nearby moves to an empty cell. Even a mild preference, a third of neighbours, produces clumps; tolerance 0 keeps the mix.',
  shows: ['boardSteps', 'BoardPlayer'],
}

export default function BoardStepsRecipe(): ReactNode {
  const [wants, setWants] = useState(0.4)
  const [empty, setEmpty] = useState(0.15)
  const [seed, setSeed] = useState(1)
  const record = useMemo(() => {
    const size = 16
    const n = size * size
    const u = uniforms(seed, n + 40 * n)
    let cells = Array.from({ length: n }, (_, i) => ((u[i] ?? 0) < empty ? 0 : (u[n - 1 - i] ?? 0) < 0.5 ? 1 : 2))
    const neighbours = (i: number): number[] => {
      const r = Math.floor(i / size)
      const c = i % size
      const out: number[] = []
      for (let dr = -1; dr <= 1; dr++)
        for (let dc = -1; dc <= 1; dc++) {
          const rr = r + dr
          const cc = c + dc
          if ((dr !== 0 || dc !== 0) && rr >= 0 && rr < size && cc >= 0 && cc < size) out.push(rr * size + cc)
        }
      return out
    }
    const unhappy = (cs: readonly number[]): number[] =>
      cs.flatMap((v, i) => {
        if (v === 0) return []
        const near = neighbours(i).filter((j) => cs[j] !== 0)
        return near.length > 0 && near.filter((j) => cs[j] === v).length / near.length < wants ? [i] : []
      })
    const steps = []
    const metrics: number[] = []
    for (let t = 0; t < 40; t++) {
      const bad = unhappy(cells)
      steps.push({ cells: [...cells], caption: `${bad.length} unhappy` })
      metrics.push(bad.length)
      const free = cells.flatMap((v, i) => (v === 0 ? [i] : []))
      const next = [...cells]
      bad.forEach((i, m) => {
        const j = free.splice(Math.floor((u[n + t * n + m] ?? 0) * free.length), 1)[0]
        if (j !== undefined) {
          next[j] = cells[i] ?? 0
          next[i] = 0
          free.push(i)
        }
      })
      cells = next
    }
    return boardSteps({
      method: 'Schelling model',
      layout: { type: 'grid', rows: size, columns: size },
      categories: ['empty', 'kind A', 'kind B'],
      steps,
      metrics: [{ name: 'unhappy residents', values: metrics }],
      stepName: 'round',
    })
  }, [wants, empty, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="share of own kind wanted" value={wants} onChange={setWants} min={0} max={0.8} step={0.05} />
        <NumberField label="empty cells" value={empty} onChange={setEmpty} min={0.05} max={0.5} step={0.05} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView record={record} title="Neighbourhoods sorting themselves" />
    </div>
  )
}
