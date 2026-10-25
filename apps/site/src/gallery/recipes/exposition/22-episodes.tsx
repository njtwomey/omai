import { episodeRecord } from 'mdml-engine/exposition'
import { frozenLake } from 'mdml-engine/methods/rl'
import { greedyTable, qLearning, valueTable } from 'mdml-engine/methods/rl/agents'
import { make, trainAgent } from 'mdml-engine/rl'
import { ControlPanel, EpisodeView, NumberField } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Episodes',
  question: 'How do I show an agent learning: returns, and its values and policy?',
  explain:
    'An `episodes` record holds every episode’s return and length, with snapshots of a tabular agent’s values and greedy policy. Here Q-learning on FrozenLake (non-slippery) is recorded from its training checkpoints; `EpisodeView` draws the returns with their rolling mean and, for the snapshot, the lake as a map of values with the greedy action in each cell.',
}

// region
export default function EpisodesRecipe(): ReactNode {
  const [epsilon, setEpsilon] = useState(0.2)
  const [snapshot, setSnapshot] = useState(-1)
  const record = useMemo(() => {
    const env = make(frozenLake, { isSlippery: false })
    const agent = qLearning(env, { alpha: 0.5, gamma: 0.95, epsilon })
    const run = trainAgent(env, agent, { episodes: 400, seed: 3 })
    const checkpoints = run.state.checkpoints.filter((c) => c.state.episode > 0)
    return episodeRecord({
      environment: 'FrozenLake 4x4',
      agent: 'Q-learning',
      returns: run.returns,
      lengths: run.lengths,
      snapshots: [
        ...checkpoints.map((c) => ({
          episode: c.state.episode,
          values: valueTable(c.state.agent),
          policy: greedyTable(c.state.agent),
        })),
        { episode: run.episodes, values: valueTable(run.agent), policy: greedyTable(run.agent) },
      ],
    })
  }, [epsilon])
  const count = record.snapshots?.length ?? 1
  const shown = snapshot < 0 || snapshot >= count ? count - 1 : snapshot
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="exploration ε" value={epsilon} onChange={setEpsilon} min={0} max={1} />
        <NumberField label="snapshot" value={shown} onChange={setSnapshot} min={0} max={count - 1} integer />
      </ControlPanel>
      <EpisodeView record={record} snapshot={shown} gridShape={[4, 4]} />
    </div>
  )
}
// endregion
