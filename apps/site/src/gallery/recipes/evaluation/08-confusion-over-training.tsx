import { confusionCounts, confusionMatrix } from 'omai-engine/exposition'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { uniforms } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Confusion matrix over training',
  question: 'How do I show which classes a model confuses, and how that changes as it learns?',
  explain:
    'A `confusionMatrix` record is the counts of true class against predicted class and, optionally, `steps`: the counts after each iteration of training. `ConfusionMatrixView` draws the matrix as counts, each cell shaded by its share of the true class, and plays the steps with the player, so the diagonal can be watched filling in. The toy classifier starts at chance and improves at the rate you set, except that two look-alike classes keep being mistaken for each other. Raise their confusability and the pair stays off the diagonal while everything else converges.',
  shows: ['confusionMatrix', 'ConfusionMatrixView'],
}

const NAMES = ['cat', 'dog', 'fox', 'owl', 'bat', 'eel', 'ant', 'yak']

export default function ConfusionOverTraining(): ReactNode {
  const [classes, setClasses] = useState(5)
  const [skill, setSkill] = useState(0.9)
  const [confusable, setConfusable] = useState(0.35)
  const [seed, setSeed] = useState(1)
  const record = useMemo(() => {
    const k = classes
    const perClass = 60
    const n = k * perClass
    const truth = Array.from({ length: n }, (_, i) => i % k)
    const u = uniforms(seed, 2 * n)
    const stepsCount = 12
    const steps = Array.from({ length: stepsCount }, (_, s) => {
      const learned = skill * (1 - Math.exp(-(s + 0.5) / 3))
      const predicted = truth.map((t, i) => {
        const r = u[i] ?? 0
        // Cat and dog look alike: a share of each is called the other even once the model is good.
        if (t < 2 && (u[n + i] ?? 0) < confusable * learned) return 1 - t
        return r < learned ? t : Math.floor((u[n + i] ?? 0) * k)
      })
      return { iteration: s * 10, counts: confusionCounts(truth, predicted, k) }
    })
    const last = steps[steps.length - 1]
    return confusionMatrix({
      classes: NAMES.slice(0, k),
      counts: last?.counts ?? [],
      steps,
      stepName: 'epoch',
      split: 'validation',
    })
  }, [classes, skill, confusable, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={4}>
        <NumberField label="classes" value={classes} onChange={setClasses} min={3} max={8} integer />
        <NumberField label="final skill" value={skill} onChange={setSkill} min={0.3} max={1} step={0.05} />
        <NumberField
          label="cat and dog confusable"
          value={confusable}
          onChange={setConfusable}
          min={0}
          max={0.8}
          step={0.05}
        />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView record={record} title="What is mistaken for what" />
    </div>
  )
}
