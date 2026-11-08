import { bool, box, choice, dict, int, real, tuple, union } from 'omai-engine/space'
import { describe, expect, it } from 'vitest'
import { greyOptions, settleValue, spaceControls, valueAt, withValueAt } from './space-controls.ts'

const space = dict({
  model: choice(['linear', 'tree'], { label: 'model' }),
  alpha: real(1e-4, 10, { log: true, label: 'penalty', when: { key: 'model', in: ['linear'] } }),
  depth: int(1, 8, { when: { key: 'model', in: ['tree'] } }),
  kernel: choice(['rbf', 'polynomial', 'linear', 'sigmoid', 'laplacian']),
  standardise: bool({ default: true }),
})

const withSolver = dict({
  solver: union(
    {
      closedForm: dict({}),
      sgd: dict({
        batch: int(1, 256),
        optimiser: union({ plain: dict({}), momentum: dict({ beta: real(0, 1) }) }, { tag: 'name' }),
      }),
    },
    { label: 'solver' },
  ),
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

  it('draws a union as its choice followed by the chosen option’s knobs at the same level', () => {
    const flat = spaceControls(withSolver, { solver: { kind: 'closedForm' } })
    expect(flat.map((c) => [c.id, c.kind])).toEqual([['solver', 'group']])
    const [group] = flat
    if (group?.kind !== 'group') return
    expect(group.children.map((c) => [c.id, c.kind, c.path])).toEqual([['solver.kind', 'choice', ['solver', 'kind']]])
    const deep = spaceControls(withSolver, {
      solver: { kind: 'sgd', batch: 8, optimiser: { name: 'momentum', beta: 0.9 } },
    })[0]
    if (deep?.kind !== 'group') return
    expect(deep.children.map((c) => c.id)).toEqual(['solver.kind', 'solver.batch', 'solver.optimiser'])
    const optimiser = deep.children[2]
    if (optimiser?.kind !== 'group') return
    expect(optimiser.children.map((c) => c.id)).toEqual(['solver.optimiser.name', 'solver.optimiser.beta'])
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

  it('switching a union’s option swaps its knobs for the new option’s defaults', () => {
    const v = { solver: { kind: 'sgd', batch: 8, optimiser: { name: 'momentum', beta: 0.9 } } }
    expect(withValueAt(withSolver, v, ['solver', 'kind'], 'closedForm')).toEqual({ solver: { kind: 'closedForm' } })
    expect(withValueAt(withSolver, v, ['solver', 'optimiser', 'name'], 'plain')).toEqual({
      solver: { kind: 'sgd', batch: 8, optimiser: { name: 'plain' } },
    })
    expect(withValueAt(withSolver, { solver: { kind: 'closedForm' } }, ['solver', 'kind'], 'sgd')).toEqual({
      solver: { kind: 'sgd', batch: 129, optimiser: { name: 'plain' } },
    })
  })

  it('reads and writes nested paths', () => {
    const s = tuple([box(-1, 1, [2]), int(0, 3)])
    const v = withValueAt(s, [[0, 0], 1], [0, 1], 0.5)
    expect(v).toEqual([[0, 0.5], 1])
    expect(valueAt(v, [0, 1])).toBe(0.5)
  })
})

describe('option descriptions', () => {
  it('travel from the space to the control, and stay on the available options of a grey-out', () => {
    const space = dict({
      basis: choice(['ps', 'cr'], { descriptions: { ps: 'P-spline', cr: 'cubic regression spline' } }),
    })
    const [control] = spaceControls(space, { basis: 'ps' })
    expect(control).toMatchObject({ kind: 'choice', descriptions: { ps: 'P-spline' } })
    const options = greyOptions(
      ['ps', 'cr'],
      ['basis'],
      undefined,
      (control as { descriptions: Record<string, string> }).descriptions,
    )
    expect(options).toEqual([
      { value: 'ps', description: 'P-spline' },
      { value: 'cr', description: 'cubic regression spline' },
    ])
    const grey = greyOptions(
      ['ps', 'cr'],
      ['basis'],
      { admissible: (_p, v) => v === 'ps', why: () => 'no' },
      { ps: 'P-spline' },
    )
    expect(grey).toEqual([
      { value: 'ps', description: 'P-spline' },
      { value: 'cr', disabled: true, description: 'no' },
    ])
  })
})

describe('greyOptions', () => {
  it('leaves the options as they are without a rule', () => {
    expect(greyOptions(['a', 'b'], ['x'], undefined)).toEqual(['a', 'b'])
  })

  it('disables the inadmissible options with the reason as their description', () => {
    const rule = {
      admissible: (path: readonly (string | number)[], o: string | number | boolean) =>
        !(path[0] === 'solver' && o === 'lars'),
      why: (_path: readonly (string | number)[], o: string | number | boolean) => `${o} needs an l1 penalty`,
    }
    expect(greyOptions(['lars', 'lbfgs'], ['solver', 'kind'], rule)).toEqual([
      { value: 'lars', disabled: true, description: 'lars needs an l1 penalty' },
      'lbfgs',
    ])
    expect(greyOptions(['lars'], ['other'], rule)).toEqual(['lars'])
    expect(greyOptions(['x'], [], { admissible: () => false })[0]).toMatchObject({
      disabled: true,
      description: 'not available with the current settings',
    })
  })
})
