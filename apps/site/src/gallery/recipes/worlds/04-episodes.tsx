import { cartPole } from 'mdml-engine/methods/rl'
import { make } from 'mdml-engine/rl'
import { ControlPanel, EpisodePlayer, NumberField, SeedInput } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Episodes',
  question: 'How do I play back an episode of an environment?',
  explain:
    '`EpisodePlayer` runs one episode of an environment under a policy (reproducible from its seed), renders every state with the environment’s own `render`, and plays the frames with each step’s action, reward and return under them. The policy here is a hand-written controller for the cart-pole: push towards the side the pole is falling, more eagerly as the gain rises.',
}

// region
const env = make(cartPole)

export default function Episodes(): ReactNode {
  const [gain, setGain] = useState(0.5)
  const [seed, setSeed] = useState(0)
  // Observation: cart position and velocity, pole angle and angular velocity. Action: 0 pushes left, 1 right.
  const policy = useMemo(() => (o: readonly number[]) => ((o[2] ?? 0) + gain * (o[3] ?? 0) > 0 ? 1 : 0), [gain])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="angular velocity gain" value={gain} onChange={setGain} min={0} max={2} step={0.05} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <EpisodePlayer env={env} policy={policy} seed={seed} maxSteps={300} fps={30} width={480} />
    </div>
  )
}
// endregion
