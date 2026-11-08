import { diagramRecord } from 'omai-engine/exposition'
import { structure } from 'omai-engine/scene'
import { ControlPanel, ExpositionView, NumberField } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Diagram steps',
  question: 'How do I walk through a structure, lighting the part that is working at each step?',
  explain:
    'A `diagram` record is a structure (nodes with roles, edges) and, optionally, a list of steps, each an overlay naming the active nodes and edges, badges on nodes and a caption. `DiagramPlayer` draws the structure and plays the steps, so the same picture explains a forward pass, a message-passing round or a procedure. This one is a small network: the input is clamped, each hidden unit lights in turn with a badge of its activation, and the output follows. More hidden units make the picture wider and the pass longer.',
  shows: ['diagram', 'DiagramPlayer'],
}

export default function DiagramStepsRecipe(): ReactNode {
  const [hidden, setHidden] = useState(5)
  const record = useMemo(() => {
    const ids = Array.from({ length: hidden }, (_, i) => `h${i + 1}`)
    const nodes = [
      { id: 'x1', label: 'x₁', role: 'input' as const },
      { id: 'x2', label: 'x₂', role: 'input' as const },
      ...ids.map((id, i) => ({ id, label: `h${i + 1}`, role: 'operation' as const, group: 'hidden' })),
      { id: 'y', label: 'ŷ', role: 'output' as const },
    ]
    const edges = [
      ...['x1', 'x2'].flatMap((a) => ids.map((b) => ({ from: a, to: b }))),
      ...ids.map((a) => ({ from: a, to: 'y' })),
    ]
    const act = (i: number): number => Math.round(100 * Math.tanh(0.8 * (i + 1) - 1.2)) / 100
    return diagramRecord({
      structure: structure('network', nodes, edges, {
        direction: 'right',
        groups: [{ id: 'hidden', label: 'hidden layer', kind: 'block' as const }],
      }),
      stepName: 'stage',
      steps: [
        {
          overlay: { active: ['x1', 'x2'], badges: { x1: { value: 0.5 }, x2: { value: -1.2 } } },
          caption: 'the input is set',
        },
        ...ids.map((id, i) => ({
          overlay: {
            active: [id],
            edges: ['x1', 'x2'].map((a) => ({ from: a, to: id })),
            badges: { [id]: { value: act(i) } },
          },
          caption: `${id} weighs the inputs and applies tanh`,
        })),
        {
          overlay: { active: ['y'], edges: ids.map((a) => ({ from: a, to: 'y' })), badges: { y: { value: 0.42 } } },
          caption: 'the output sums the hidden units',
        },
      ],
    })
  }, [hidden])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={1}>
        <NumberField label="hidden units" value={hidden} onChange={setHidden} min={1} max={8} integer />
      </ControlPanel>
      <ExpositionView record={record} title="A forward pass" />
    </div>
  )
}
