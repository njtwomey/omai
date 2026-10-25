import { bool, box, choice, dict, int, real, tuple } from 'mdml-engine/space'
import { describe, expect, it } from 'vitest'
import { settleValue, spaceControls, valueAt, withValueAt } from './space-controls.ts'

const space = dict({
  model: choice(['linear', 'tree'], { label: 'model' }),
  alpha: real(1e-4, 10, { log: true, label: 'penalty', when: { key: 'model', in: ['linear'] } }),
  depth: int(1, 8, { when: { key: 'model', in: ['tree'] } }),
  kernel: choice(['rbf', 'polynomial', 'linear', 'sigmoid', 'laplacian']),
  standardise: bool({ default: true }),
})

describe('spaceControls', () => {
  it('maps each dimension to a control, conditional keys only while active', () => {
    const controls = spaceControls(space, { model: 'linear', alpha: 0.1, kernel: 'rbf', standardise: true })
    expect(controls.map((c) => [c.id, c.kind])).toEqual([
      ['model', 'choice'],
      ['alpha', 'number'],
      ['kernel', 'select'],
      ['standardise', 'toggle'],
    ])
    const alpha = controls[1]
    expect(alpha?.kind === 'number' ? alpha.spec : undefined).toEqual({ min: 1e-4, max: 10, log: true })
    expect(alpha?.label).toBe('penalty')
  })

  it('turns boxes and tuples into groups with paths', () => {
    const [group] = spaceControls(tuple([box(-1, 1, [2]), int(0, 3)]), [[0, 0], 1])
    expect(group?.kind).toBe('group')
    if (group?.kind !== 'group') return
    const first = group.children[0]
    expect(first?.kind).toBe('group')
    if (first?.kind !== 'group') return
    expect(first.children.map((c) => c.path)).toEqual([
      [0, 0],
      [0, 1],
    ])
  })
})

describe('settling values', () => {
  it('brings dependent keys in with defaults and drops inactive ones', () => {
    const v = { model: 'linear', alpha: 0.1, kernel: 'rbf', standardise: true }
    const next = withValueAt(space, v, ['model'], 'tree')
    expect(next).toEqual({ model: 'tree', depth: 5, kernel: 'rbf', standardise: true })
    expect(settleValue(space, { model: 'linear', kernel: 'rbf', standardise: false })).toEqual({
      model: 'linear',
      alpha: expect.closeTo(0.0316, 3),
      kernel: 'rbf',
      standardise: false,
    })
  })

  it('reads and writes nested paths', () => {
    const s = tuple([box(-1, 1, [2]), int(0, 3)])
    const v = withValueAt(s, [[0, 0], 1], [0, 1], 0.5)
    expect(v).toEqual([[0, 0.5], 1])
    expect(valueAt(v, [0, 1])).toBe(0.5)
  })
})
