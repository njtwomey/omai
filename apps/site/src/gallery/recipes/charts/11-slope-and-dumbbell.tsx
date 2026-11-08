import { cartesian, lines, points } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, Choice, ControlPanel, FigureFrame, NumberField, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Slope and dumbbell charts',
  question: 'How do I show before and after for many items?',
  explain:
    "A slope chart joins each item's before and after with a line, so rises and falls read as slopes; a dumbbell chart puts both on one row per item, sorted by the change. Both are a few `lines` and `points`; the dumbbell labels its rows with a category axis.",
}

const FIGURE = figure({ width: 720, height: 460 })
const ITEMS = ['Arts', 'Biology', 'Chemistry', 'Drama', 'Economics', 'French', 'Geography', 'History']
const BEFORE = [62, 71, 58, 66, 54, 60, 69, 57]

export default function SlopeAndDumbbell(): ReactNode {
  const mode = useColourMode()
  const [view, setView] = useState<'slope' | 'dumbbell'>('slope')
  const [shake, setShake] = useState(8)
  const option = useMemo(() => {
    const after = BEFORE.map((b, i) => Math.round(b + shake * Math.sin(i * 2.3 + 1)))
    if (view === 'slope')
      return cartesian(
        [
          lines(
            ITEMS.map((name, i) => ({ x: [0, 1], y: [BEFORE[i] ?? 0, after[i] ?? 0], name, width: 2 })),
            { mode },
          ),
          points(
            ITEMS.map((name, i) => ({ x: [0, 1], y: [BEFORE[i] ?? 0, after[i] ?? 0], name: `__${name}`, size: 7 })),
            { mode },
          ),
        ],
        {
          mode,
          x: { label: '', limits: [-0.15, 1.15], format: (v) => (v === 0 ? 'before' : v === 1 ? 'after' : '') },
          y: { label: 'score' },
        },
      )
    const order = ITEMS.map((_, i) => i).sort(
      (a, b) => (after[a] ?? 0) - (BEFORE[a] ?? 0) - ((after[b] ?? 0) - (BEFORE[b] ?? 0)),
    )
    return cartesian(
      [
        lines(
          order.map((i, r) => ({
            x: [BEFORE[i] ?? 0, after[i] ?? 0],
            y: [r, r],
            name: `__${ITEMS[i]}`,
            colour: 'muted',
            width: 3,
          })),
          { mode },
        ),
        points([{ x: order.map((i) => BEFORE[i] ?? 0), y: order.map((_, r) => r), name: 'before', size: 9 }], { mode }),
        points([{ x: order.map((i) => after[i] ?? 0), y: order.map((_, r) => r), name: 'after', size: 9 }], {
          mode,
          colourOffset: 1,
        }),
      ],
      {
        mode,
        x: { label: 'score' },
        y: { label: '', limits: [-0.5, ITEMS.length - 0.5], format: (v) => ITEMS[order[Math.round(v)] ?? 0] ?? '' },
      },
    )
  }, [view, shake, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice label="view" value={view} onChange={setView} options={['slope', 'dumbbell'] as const} />
        <NumberField label="change" value={shake} onChange={setShake} min={0} max={25} integer />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="slope-and-dumbbell">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
