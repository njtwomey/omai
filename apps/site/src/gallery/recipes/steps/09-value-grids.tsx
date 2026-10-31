import { valueGrids } from 'mdml-engine/exposition'
import { ControlPanel, ExpositionView, NumberField } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Value grids',
  question: 'How do I show a value spreading over a grid, sweep by sweep?',
  explain:
    'A `valueGrids` record is one or more named grids of numbers and, optionally, the grids after each step. `ValueGridView` draws them as heatmaps on a shared colour scale and plays the steps. Here value iteration runs on a small grid world with a goal in one corner, a pit and a wall: every sweep spreads the value of the goal one cell further, so the picture is a wave front, and a discount close to 1 lets it carry far while a small one dies out within a few cells. A step cost makes the values negative everywhere but the goal.',
  shows: ['valueGrids', 'ValueGridView'],
}

const ROWS = 7
const COLS = 9
const WALLS = new Set([2 * COLS + 3, 3 * COLS + 3, 4 * COLS + 3, 4 * COLS + 4, 4 * COLS + 5])
const GOAL = 0 * COLS + (COLS - 1)
const PIT = 3 * COLS + (COLS - 1)

export default function ValueGridsRecipe(): ReactNode {
  const [gamma, setGamma] = useState(0.9)
  const [cost, setCost] = useState(0.04)
  const record = useMemo(() => {
    let v = new Array<number>(ROWS * COLS).fill(0)
    const sweeps: number[][] = [v]
    for (let t = 0; t < 30; t++) {
      v = v.map((_, i) => {
        if (WALLS.has(i)) return 0
        if (i === GOAL) return 1
        if (i === PIT) return -1
        const r = Math.floor(i / COLS)
        const c = i % COLS
        const next = [
          [r - 1, c],
          [r + 1, c],
          [r, c - 1],
          [r, c + 1],
        ].map(([rr = 0, cc = 0]) => {
          const j = rr * COLS + cc
          return rr < 0 || rr >= ROWS || cc < 0 || cc >= COLS || WALLS.has(j) ? (v[i] ?? 0) : (v[j] ?? 0)
        })
        return -cost + gamma * Math.max(...next)
      })
      sweeps.push(v)
    }
    const asGrid = (flat: readonly number[]): number[][] =>
      Array.from({ length: ROWS }, (_, r) => flat.slice(r * COLS, (r + 1) * COLS))
    return valueGrids({
      grids: [{ name: 'state value', values: asGrid(v) }],
      diverging: true,
      axes: ['row', 'column'],
      valueName: 'V(s)',
      steps: sweeps.map((s, t) => ({ iteration: t, grids: [asGrid(s)] })),
      stepName: 'sweep',
    })
  }, [gamma, cost])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="discount" value={gamma} onChange={setGamma} min={0.3} max={0.99} step={0.01} />
        <NumberField label="cost per step" value={cost} onChange={setCost} min={0} max={0.2} step={0.01} />
      </ControlPanel>
      <ExpositionView record={record} title="Value iteration on a grid world" />
    </div>
  )
}
