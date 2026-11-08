import { confusionMatrix } from 'omai-engine/metrics'
import { ContingencyTable, ControlPanel, Select, Toggle } from 'omai-render/react'
import { categoricalColour } from 'omai-render/theme'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Contingency table',
  question: 'How do I show classification results with their margins and rates?',
  explain:
    '`ContingencyTable` takes any confusion matrix (`confusionMatrix(yTrue, yPred)` from `omai-engine/metrics`) and draws it with its margins: row totals with each class’s recall and miss rate, column totals with precision and false discovery rate, and the grand total with accuracy. With two classes and a `positive` index the cells are named TP, FN, FP and TN, and the rates read sensitivity (recall), specificity, precision and their complements. Hover a rate for its acronym (TPR, TNR, PPV, …) and formula, and to outline what it divides: the numerator solid, the rest of the denominator dashed.',
}

// Fixed predictions: a three-class problem and a two-class screening test.
const repeat = (pairs: readonly [string, string, number][]): [string[], string[]] => {
  const t: string[] = []
  const p: string[] = []
  for (const [a, b, n] of pairs) {
    for (let i = 0; i < n; i++) {
      t.push(a)
      p.push(b)
    }
  }
  return [t, p]
}
const tables = {
  'three classes (iris)': {
    data: repeat([
      ['setosa', 'setosa', 50],
      ['versicolor', 'versicolor', 46],
      ['versicolor', 'virginica', 4],
      ['virginica', 'versicolor', 3],
      ['virginica', 'virginica', 47],
    ]),
    positive: undefined,
  },
  'two classes (screening)': {
    data: repeat([
      ['healthy', 'healthy', 880],
      ['healthy', 'ill', 70],
      ['ill', 'healthy', 8],
      ['ill', 'ill', 42],
    ]),
    positive: 1,
  },
} as const
const names = Object.keys(tables) as (keyof typeof tables)[]
const shares = ['all', 'row', 'column', 'none'] as const

export default function ContingencyTableRecipe(): ReactNode {
  const [which, setWhich] = useState<(typeof names)[number]>('two classes (screening)')
  const [share, setShare] = useState<(typeof shares)[number]>('all')
  const [dots, setDots] = useState(true)
  const { data, positive } = tables[which]
  const table = useMemo(() => confusionMatrix(data[0], data[1]), [data])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <Select label="results" value={which} options={names} onChange={setWhich} />
        <Select label="share under each count" value={share} options={shares} onChange={setShare} />
        <Toggle label="class colours" value={dots} onChange={setDots} />
      </ControlPanel>
      <ContingencyTable
        table={table}
        {...(positive === undefined ? {} : { positive })}
        share={share}
        {...(dots ? { classColours: table.labels.map((_, i) => categoricalColour(i)) } : {})}
      />
    </div>
  )
}
