import { type ReactNode, useMemo, useState } from 'react'
import { bayesNet, factorValues, variableElimination } from 'whitebox-engine/pgm'
import { ControlPanel, SceneView, Select } from 'whitebox-render/react'
import { bayesNetCards } from 'whitebox-render/structures'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Bayes network with tables',
  question: 'How do I draw a Bayes network with its probability tables and the posteriors they give?',
  explain:
    '`bayesNetCards` draws each variable as a card holding its conditional probability table, one row per assignment of its parents. Observing a variable shades its card and highlights the rows that apply; the posteriors (exact, from `variableElimination` in whitebox-engine’s `pgm`) sit above each table. Observe wet grass, then rain, and watch the sprinkler explain away.',
}

const bn = bayesNet([
  { name: 'cloudy', states: ['no', 'yes'], parents: [], cpt: [0.5, 0.5] },
  { name: 'sprinkler', states: ['off', 'on'], parents: ['cloudy'], cpt: [0.5, 0.5, 0.9, 0.1] },
  { name: 'rain', states: ['no', 'yes'], parents: ['cloudy'], cpt: [0.8, 0.2, 0.2, 0.8] },
  {
    name: 'wet grass',
    states: ['no', 'yes'],
    parents: ['sprinkler', 'rain'],
    cpt: [1, 0, 0.1, 0.9, 0.1, 0.9, 0.01, 0.99],
  },
])
const SEEN = ['unobserved', 'no', 'yes']

export default function BayesNetTables(): ReactNode {
  const [wet, setWet] = useState('yes')
  const [rain, setRain] = useState('unobserved')
  const drawing = useMemo(() => {
    const evidence: Record<string, string> = {}
    if (wet !== 'unobserved') evidence['wet grass'] = wet
    if (rain !== 'unobserved') evidence['rain'] = rain
    const marginals = Object.fromEntries(
      bn.nodes.map((n) => [
        n.name,
        n.name in evidence
          ? n.states.map((s) => (s === evidence[n.name] ? 1 : 0))
          : factorValues(variableElimination(bn, [n.name], { evidence }).posterior),
      ]),
    )
    return bayesNetCards(bn, { evidence, marginals })
  }, [wet, rain])
  return (
    <div className="flex w-fit max-w-full flex-col gap-3">
      <ControlPanel columns={2}>
        <Select label="wet grass" value={wet} options={SEEN} onChange={setWet} />
        <Select label="rain" value={rain} options={SEEN} onChange={setRain} />
      </ControlPanel>
      <SceneView scene={drawing} width={Math.min(720, drawing.width)} />
    </div>
  )
}
