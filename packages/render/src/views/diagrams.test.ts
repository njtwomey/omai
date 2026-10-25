import { diagramRecord } from 'mdml-engine/exposition'
import { structure } from 'mdml-engine/scene'
import { describe, expect, it } from 'vitest'
import { diagramScene } from './diagrams.ts'
import { stillOf, viewFor } from './registry.ts'

describe('diagram views', () => {
  const s = structure(
    'net',
    [
      { id: 'x', label: 'x', role: 'input' },
      { id: 'y', label: 'y', role: 'output' },
    ],
    [{ from: 'x', to: 'y' }],
  )

  it('draw the structure with a step’s overlay, sized for every step', () => {
    const r = diagramRecord({
      structure: s,
      steps: [
        { overlay: { active: ['x'], badges: { x: { a: '1.000' } } }, caption: 'forward' },
        { overlay: { active: ['y'], badges: { y: { δ: '0.250' } } }, caption: 'back' },
      ],
    })
    const first = diagramScene(r, 0)
    const last = diagramScene(r)
    expect(first.kind).toBe('scene')
    expect([first.width, first.height]).toEqual([last.width, last.height])
    expect(viewFor('diagram').component).toBe('DiagramPlayer')
    expect((stillOf(r) as { kind: string }).kind).toBe('scene')
  })

  it('reject overlays naming nodes the structure lacks', () => {
    expect(() => diagramRecord({ structure: s, steps: [{ overlay: { active: ['z'] } }] })).toThrow(/z/)
  })
})
