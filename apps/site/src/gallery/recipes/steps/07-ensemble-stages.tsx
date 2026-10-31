import { decisionSurface, ensembleStages } from 'mdml-engine/exposition'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { crescents } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Ensemble stages',
  question: 'How do I show an ensemble being built, one member at a time?',
  explain:
    'An `ensembleStages` record has the weight of each member, the ensemble’s training and validation score after each stage and, optionally, its decision surface after each stage. `EnsembleView` plays the stages: the surface sharpening as members are added, the weights as bars and both scores as curves. These members are one-split stumps added by AdaBoost on two crescents: each stage picks the split that the rows still wrongly classified need most, so the surface gets staircase boundaries, and the validation curve flattens or turns while training keeps improving once the members only fit the noise.',
  shows: ['ensembleStages', 'EnsembleView'],
}

export default function EnsembleStagesRecipe(): ReactNode {
  const [members, setMembers] = useState(30)
  const [flip, setFlip] = useState(0.1)
  const [seed, setSeed] = useState(1)
  const { record, points, labels } = useMemo(() => {
    const train = crescents(seed, 160, flip)
    const valid = crescents(seed + 100, 160, flip)
    const n = train.x.length
    let w = new Array<number>(n).fill(1 / n)
    const stumps: { f: number; t: number; s: number; alpha: number }[] = []
    const m = 30
    const gx = Array.from({ length: m }, (_, i) => -1.8 + (4.6 * i) / (m - 1))
    const gy = Array.from({ length: m }, (_, i) => -1.4 + (3.2 * i) / (m - 1))
    const vote = (x: readonly number[]): number =>
      stumps.reduce((a, st) => a + st.alpha * (st.s * ((x[st.f] ?? 0) > st.t ? 1 : -1)), 0)
    const trainScores: number[] = []
    const validationScores: number[] = []
    const weights: number[] = []
    const steps = []
    for (let r = 0; r < members; r++) {
      let best = { f: 0, t: 0, s: 1, err: 2 }
      for (const f of [0, 1])
        for (const t of train.x.map((p) => p[f] ?? 0).filter((_, i) => i % 3 === 0))
          for (const s of [1, -1]) {
            const err = train.x.reduce(
              (a, p, i) => a + ((s * ((p[f] ?? 0) > t ? 1 : -1) > 0 ? 1 : 0) !== train.y[i] ? (w[i] ?? 0) : 0),
              0,
            )
            if (err < best.err) best = { f, t, s, err }
          }
      const e = Math.min(0.499, Math.max(1e-6, best.err))
      const alpha = 0.5 * Math.log((1 - e) / e)
      stumps.push({ f: best.f, t: best.t, s: best.s, alpha })
      w = w.map(
        (v, i) =>
          v *
          Math.exp(
            alpha * ((best.s * ((train.x[i]?.[best.f] ?? 0) > best.t ? 1 : -1) > 0 ? 1 : 0) === train.y[i] ? -1 : 1),
          ),
      )
      const z = w.reduce((a, b) => a + b, 0)
      w = w.map((v) => v / z)
      weights.push(alpha)
      const acc = (d: { x: number[][]; y: number[] }): number =>
        d.x.filter((p, i) => (vote(p) > 0 ? 1 : 0) === d.y[i]).length / d.x.length
      trainScores.push(acc(train))
      validationScores.push(acc(valid))
      steps.push({ iteration: r + 1, labels: gy.map((y) => gx.map((x) => (vote([x, y]) > 0 ? 1 : 0))) })
    }
    return {
      points: train.x,
      labels: train.y,
      record: ensembleStages({
        method: 'AdaBoost with stumps',
        weights,
        trainScores,
        validationScores,
        scoreName: 'accuracy',
        higherIsBetter: true,
        surface: decisionSurface({
          features: ['x', 'y'],
          x: gx,
          y: gy,
          labels: steps[steps.length - 1]?.labels ?? [],
          steps,
          stepName: 'members',
          classes: ['upper crescent', 'lower crescent'],
          rows: {
            points: train.x,
            features: ['x', 'y'],
            labels: train.y,
            classes: ['upper crescent', 'lower crescent'],
          },
        }),
      }),
    }
  }, [members, flip, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="members" value={members} onChange={setMembers} min={2} max={60} integer />
        <NumberField label="labels flipped" value={flip} onChange={setFlip} min={0} max={0.3} step={0.01} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView record={record} points={points} labels={labels} title="Boosting stumps into a boundary" />
    </div>
  )
}
