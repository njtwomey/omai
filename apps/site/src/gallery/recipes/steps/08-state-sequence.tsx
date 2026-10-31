import { stateSequence } from 'mdml-engine/exposition'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { normals, uniforms } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Hidden states',
  question: 'How do I show what a model believes about a hidden state at every time, next to what was true?',
  explain:
    'A `stateSequence` record holds, per time step, the posterior probability of each hidden state, the path the model decoded, the true path when it is known and the observations. `StateSequenceView` draws the observations over a band of state colours whose height at each time is the posterior, with the decoded path and the truth as strips beneath: where the colours are mixed the model is unsure, where the strips disagree it is wrong. The model is a three-state hidden Markov model with Gaussian emissions, filtered by the forward-backward algorithm. Bring the state means together or raise the noise and the posteriors blur.',
  shows: ['stateSequence', 'StateSequenceView'],
}

const STATES = ['calm', 'busy', 'storm']
const MEANS = [0, 2, 4]

export default function StateSequenceRecipe(): ReactNode {
  const [noise, setNoise] = useState(0.8)
  const [stay, setStay] = useState(0.93)
  const [seed, setSeed] = useState(1)
  const record = useMemo(() => {
    const T = 120
    const u = uniforms(seed, T)
    const e = normals(seed + 1, T, noise)
    const K = STATES.length
    const trans = (i: number, j: number): number => (i === j ? stay : (1 - stay) / (K - 1))
    const truth: number[] = []
    let s = 0
    for (let t = 0; t < T; t++) {
      if ((u[t] ?? 0) > stay) s = (s + 1 + Math.floor(((u[t] ?? 0) * 1000) % (K - 1))) % K
      truth.push(s)
    }
    const obs = truth.map((st, t) => (MEANS[st] ?? 0) + (e[t] ?? 0))
    const lik = (x: number, k: number): number => Math.exp(-((x - (MEANS[k] ?? 0)) ** 2) / (2 * noise ** 2))
    const fwd: number[][] = []
    let a = MEANS.map((_, k) => lik(obs[0] ?? 0, k) / K)
    for (let t = 0; t < T; t++) {
      if (t > 0) a = MEANS.map((_, j) => lik(obs[t] ?? 0, j) * a.reduce((acc, v, i) => acc + v * trans(i, j), 0))
      const z = a.reduce((x, y) => x + y, 0) || 1
      a = a.map((v) => v / z)
      fwd.push(a)
    }
    const bwd: number[][] = new Array<number[]>(T)
    bwd[T - 1] = MEANS.map(() => 1)
    for (let t = T - 2; t >= 0; t--) {
      const b = MEANS.map((_, i) =>
        MEANS.reduce((acc, _, j) => acc + trans(i, j) * lik(obs[t + 1] ?? 0, j) * (bwd[t + 1]?.[j] ?? 1), 0),
      )
      const z = b.reduce((x, y) => x + y, 0) || 1
      bwd[t] = b.map((v) => v / z)
    }
    const posteriors = fwd.map((f, t) => {
      const p = f.map((v, k) => v * (bwd[t]?.[k] ?? 1))
      const z = p.reduce((x, y) => x + y, 0) || 1
      return p.map((v) => v / z)
    })
    return stateSequence({
      states: STATES,
      posteriors,
      decoded: posteriors.map((p) => p.indexOf(Math.max(...p))),
      truth,
      observations: { name: 'wave height', values: obs },
    })
  }, [noise, stay, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="observation noise" value={noise} onChange={setNoise} min={0.2} max={3} step={0.1} />
        <NumberField label="chance of staying" value={stay} onChange={setStay} min={0.6} max={0.99} step={0.01} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView record={record} title="Posterior over three weather states" />
    </div>
  )
}
