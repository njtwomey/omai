import { describe, expect, it } from 'vitest'
import type { SceneNode } from 'whitebox-engine/scene'
import { fromAscii } from 'whitebox-engine/worlds'
import { pathFrames, raycastViews, worldView } from './worlds.ts'

const flatten = (nodes: readonly SceneNode[]): SceneNode[] =>
  nodes.flatMap((n) => (n.kind === 'group' ? [n, ...flatten(n.children)] : [n]))

describe('world views', () => {
  const w = fromAscii(['S..', '.#.', '..G'])

  it('lays a policy, labels, frontier and explored cells over the map', () => {
    const s = worldView(w, {
      policy: [1, 2, 2, 2, 0, 2, 1, 1, 0],
      labels: [4, 3, 2, 3, Number.NaN, 1, 2, 1, 0],
      frontier: [2],
      explored: [0, 1],
    })
    const all = flatten(s.children)
    const policy = all.find((n) => n.id === 'policy')
    expect(policy?.kind === 'group' && policy.children.length).toBe(9)
    const labels = all.find((n) => n.id === 'labels')
    expect(labels?.kind === 'group' && labels.children.length).toBe(8)
    expect(all.filter((n) => n.kind === 'path' && n.stroke === 'accent')).toHaveLength(1)
  })

  it('draws a distribution over actions as several arrows', () => {
    const probs = Array.from({ length: 9 }, () => [0.25, 0.25, 0.25, 0.25])
    const policy = flatten(worldView(w, { policy: probs }).children).find((n) => n.id === 'policy')
    expect(policy?.kind === 'group' && policy.children.length).toBe(36)
  })

  it('casts a first-person view with its cone on the map', () => {
    const room = fromAscii(['#####', '#...#', '#####'])
    const v = raycastViews(room, { x: 1.5, y: 1.5, heading: 0 }, { width: 16, height: 12, rayEvery: 4 })
    expect(v.firstPerson.children[0]?.kind).toBe('image')
    const rays = flatten(v.topDown.children).find((n) => n.id === 'rays')
    expect(rays?.kind === 'group' && rays.children.length).toBe(5)
    expect(v.frame.columns).toHaveLength(16)
  })

  it('makes one frame per step of a path', () => {
    expect(pathFrames(w, [0, 1, 2])).toHaveLength(3)
  })
})
