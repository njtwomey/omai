import { frozenLake } from 'mdml-engine/methods/rl'
import { make } from 'mdml-engine/rl'
import { ControlPanel, EnvView, SeedInput, Toggle, useEnvStepper, useHeldKeys } from 'mdml-render/react'
import { type ReactNode, useEffect, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Play an environment',
  question: 'How do I let the reader step an environment by hand?',
  explain:
    '`useEnvStepper(env, seed)` resets and steps an environment with keys derived from the seed, so the same moves replay the same episode; `EnvView` draws its state. Use the buttons, or click the figure and use the arrow keys (`useHeldKeys`), to cross the frozen lake; on slippery ice the move may go sideways.',
}

const ACTIONS = [
  { label: '←', key: 'ArrowLeft', action: 0 },
  { label: '↓', key: 'ArrowDown', action: 1 },
  { label: '→', key: 'ArrowRight', action: 2 },
  { label: '↑', key: 'ArrowUp', action: 3 },
]

export default function PlayAnEnvironment(): ReactNode {
  const [slippery, setSlippery] = useState(false)
  const [seed, setSeed] = useState(0)
  const [env, setEnv] = useState(() => make(frozenLake, { isSlippery: false }))
  useEffect(() => setEnv(make(frozenLake, { isSlippery: slippery })), [slippery])
  const run = useEnvStepper(env, seed)
  const keys = useHeldKeys(
    ACTIONS.map((a) => a.key),
    { onPress: (k) => run.step(ACTIONS.find((a) => a.key === k)?.action ?? 0) },
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Toggle label="slippery ice" value={slippery} onChange={setSlippery} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <div {...keys.props} className="w-fit rounded-lg outline-none focus-visible:ring-2 focus-visible:ring-accent/60">
        <EnvView env={env} state={run.state} width={320}>
          <div className="flex flex-col gap-2 text-sm">
            <div className="grid w-32 grid-cols-3 gap-1">
              {[null, ACTIONS[3], null, ACTIONS[0], ACTIONS[1], ACTIONS[2]].map((a, i) =>
                a === null || a === undefined ? (
                  // biome-ignore lint/suspicious/noArrayIndexKey: fixed pad layout
                  <span key={i} />
                ) : (
                  <button
                    key={a.key}
                    type="button"
                    disabled={run.done}
                    onClick={() => run.step(a.action)}
                    className="h-9 rounded-md border border-line bg-bg text-base hover:bg-surface disabled:opacity-40"
                  >
                    {a.label}
                  </button>
                ),
              )}
            </div>
            <span className="text-muted">
              step {run.t} · return {run.return}
              {run.done ? (run.lastReward > 0 ? ' · reached the goal' : ' · fell in') : ''}
            </span>
            <button
              type="button"
              onClick={run.reset}
              className="w-fit rounded-md border border-line px-2 py-1 text-xs hover:bg-surface"
            >
              Reset
            </button>
          </div>
        </EnvView>
      </div>
    </div>
  )
}
