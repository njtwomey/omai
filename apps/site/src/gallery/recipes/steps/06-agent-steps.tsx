import { agentSteps } from 'mdml-engine/exposition'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { uniforms } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Agent steps',
  question: 'How do I show many moving agents with a state of their own, and the counts of each state over time?',
  explain:
    'An `agentSteps` record has the world’s size, and per frame every agent’s position, heading and state, with series of counts over time. `AgentPlayer` plays the agents as coloured points in the world and draws the series beside it, so a change in the crowd and the curve that summarises it are one scrubber apart. These agents are walkers in an epidemic (susceptible, infected, recovered): a susceptible one next to an infected one catches it with the chance you set, and an infected one recovers after a while. Raise the infection chance and the infected curve peaks sooner and higher.',
  shows: ['agentSteps', 'AgentPlayer'],
}

export default function AgentStepsRecipe(): ReactNode {
  const [infect, setInfect] = useState(0.3)
  const [agents, setAgents] = useState(80)
  const [seed, setSeed] = useState(1)
  const record = useMemo(() => {
    const W = 20
    const T = 80
    const u = uniforms(seed, 4 * agents * T + 2 * agents)
    let x = Array.from({ length: agents }, (_, i) => (u[i] ?? 0) * W)
    let y = Array.from({ length: agents }, (_, i) => (u[agents + i] ?? 0) * W)
    let state: number[] = Array.from({ length: agents }, (_, i) => (i < 3 ? 1 : 0))
    let since = state.map(() => 0)
    const frames = []
    const counts = [[] as number[], [] as number[], [] as number[]]
    for (let t = 0; t < T; t++) {
      frames.push({ t, x: [...x], y: [...y], state: [...state] })
      for (let s = 0; s < 3; s++) counts[s]?.push(state.filter((v) => v === s).length)
      const base = 2 * agents + 4 * agents * t
      const nextState = [...state]
      state.forEach((s, i) => {
        if (s === 0) {
          const near = state.some(
            (o, j) => o === 1 && Math.hypot((x[i] ?? 0) - (x[j] ?? 0), (y[i] ?? 0) - (y[j] ?? 0)) < 1.5,
          )
          if (near && (u[base + i] ?? 1) < infect) nextState[i] = 1
        } else if (s === 1) {
          since[i] = (since[i] ?? 0) + 1
          if ((since[i] ?? 0) > 14) nextState[i] = 2
        }
      })
      x = x.map((v, i) => (v + 1.2 * ((u[base + agents + i] ?? 0) - 0.5) * 2 + W) % W)
      y = y.map((v, i) => (v + 1.2 * ((u[base + 2 * agents + i] ?? 0) - 0.5) * 2 + W) % W)
      state = nextState
      since = since.map((v, i) => (state[i] === 1 ? v : 0))
    }
    return agentSteps({
      method: 'random walkers with contagion',
      world: { width: W, height: W, wrap: true },
      states: ['susceptible', 'infected', 'recovered'],
      frames,
      series: ['susceptible', 'infected', 'recovered'].map((name, s) => ({ name, values: counts[s] ?? [], state: s })),
      seriesName: 'agents',
      timeName: 'day',
    })
  }, [infect, agents, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="chance of infection" value={infect} onChange={setInfect} min={0} max={1} step={0.05} />
        <NumberField label="agents" value={agents} onChange={setAgents} min={20} max={300} step={10} integer />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView record={record} title="An epidemic among walkers" />
    </div>
  )
}
