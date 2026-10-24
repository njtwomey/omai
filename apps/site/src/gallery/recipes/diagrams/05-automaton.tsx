import { type ReactNode, useMemo, useState } from 'react'
import { structure } from 'whitebox-engine/scene'
import { ControlPanel, Player, Select, StructureView, usePlayback } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Automaton',
  question: 'How do I draw a state machine and step it through an input?',
  explain:
    'Automata are structures of type `automaton`: states as circles, the start state ringed in the accent, accepting states doubled, transitions labelled with their symbols. The player steps through the input one symbol at a time; the overlay marks the current state and the transition just taken.',
}

// A DFA over {a, b} accepting strings that end in "ab".
const states = ['q0', 'q1', 'q2']
const delta: Record<string, Record<string, string>> = {
  q0: { a: 'q1', b: 'q0' },
  q1: { a: 'q1', b: 'q2' },
  q2: { a: 'q1', b: 'q0' },
}
const dfa = structure(
  'automaton',
  states.map((id) => ({
    id,
    label: id,
    role: id === 'q0' ? ('input' as const) : id === 'q2' ? ('output' as const) : ('state' as const),
  })),
  Object.entries(delta).flatMap(([from, out]) => {
    const byTarget = new Map<string, string[]>()
    for (const [symbol, to] of Object.entries(out)) byTarget.set(to, [...(byTarget.get(to) ?? []), symbol])
    return [...byTarget].filter(([to]) => to !== from).map(([to, symbols]) => ({ from, to, label: symbols.join(', ') }))
  }),
  { direction: 'right' },
)

const INPUTS = ['abab', 'aabba', 'bbab', 'ababa']

export default function Automaton(): ReactNode {
  const [input, setInput] = useState('abab')
  const path = useMemo(() => {
    const out = ['q0']
    for (const c of input) out.push(delta[out[out.length - 1] ?? 'q0']?.[c] ?? 'q0')
    return out
  }, [input])
  const playback = usePlayback(path.length, { fps: 1 })
  const t = playback.frame
  const state = path[t] ?? 'q0'
  const prev = t > 0 ? path[t - 1] : undefined
  const overlay = { active: [state], edges: prev === undefined ? [] : [{ from: prev, to: state }] }
  const read = input.slice(0, t)
  return (
    <div className="flex w-fit max-w-full flex-col gap-3">
      <ControlPanel>
        <Select label="input" value={input} onChange={setInput} options={INPUTS} />
      </ControlPanel>
      <StructureView structure={dfa} overlay={overlay} details={false} width={460} />
      <Player
        controller={playback}
        stamp={() =>
          `read “${read}” → ${state}${t === path.length - 1 ? (state === 'q2' ? ' (accept)' : ' (reject)') : ''}`
        }
      />
    </div>
  )
}
