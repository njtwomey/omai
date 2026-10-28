import { describe, expect, it } from 'vitest'
import { lookup } from '../base/index.ts'
import { key } from '../random/index.ts'
import { listEnvironments } from '../rl/index.ts'
import { box, dict, discrete, multiBinary, multiDiscrete, real, tuple } from '../space/index.ts'
import { pendulum } from './classic-control.ts'
import * as environments from './index.ts'
import {
  actionKindOf,
  ENVIRONMENT_FAMILIES,
  environmentEntry,
  environmentRegistry,
  findEnvironment,
  matchesEnvironmentQuery,
  observationKindOf,
  registerEnvironments,
  whereEnvironments,
} from './registry.ts'

describe('the environment registry', () => {
  it('registers every definition the module exports, single- and multi-agent, under its export name', () => {
    const registered = new Set(environmentRegistry.entries.map((e) => e.definition))
    for (const d of listEnvironments(environments)) expect(registered.has(d), d.id).toBe(true)
    for (const [name, value] of Object.entries(environments)) {
      const d = value as { readonly id?: unknown; readonly create?: unknown }
      if (typeof d.id !== 'string' || typeof d.create !== 'function') continue
      expect(findEnvironment(environmentRegistry, name)?.definition, name).toBe(value)
    }
    for (const e of environmentRegistry.entries) {
      expect(e.spec.id, e.name).toBe(e.definition.id)
      expect(e.title.length, e.name).toBeGreaterThan(0)
      expect(ENVIRONMENT_FAMILIES, e.name).toContain(e.family)
      if (e.multiAgent !== undefined) expect(e.agents.length, e.name).toBeGreaterThan(1)
    }
  })

  it('derives the action and observation kinds from the spaces', () => {
    const tags = (name: string): string => {
      const e = lookup(environmentRegistry, name)
      return `${e.actions}/${e.observations}`
    }
    expect(tags('cartPole')).toBe('discrete/vector')
    expect(tags('pendulum')).toBe('continuous/vector')
    expect(tags('frozenLake')).toBe('discrete/discrete')
    expect(tags('blackjack')).toBe('discrete/discrete')
    expect(tags('contextualLinearBandit')).toBe('discrete/vector')
    expect(tags('breakoutMinAtar')).toBe('discrete/grid')
    expect(tags('proceduralMaze')).toBe('discrete/vector')
    expect(tags('ticTacToe')).toBe('discrete/vector')
    expect(tags('cartPoleModel')).toBe('continuous/vector')
    expect(actionKindOf(multiDiscrete([2, 3]))).toBe('multi')
    expect(actionKindOf(real(0, 1))).toBe('continuous')
    expect(observationKindOf(box(0, 255, [8, 8, 3]))).toBe('image')
    expect(observationKindOf(box([0, 0], [255, 1], [2]))).toBe('vector')
    expect(observationKindOf(multiBinary(4))).toBe('vector')
    expect(observationKindOf(tuple([discrete(2), box(0, 1, [2])]))).toBe('vector')
    expect(observationKindOf(dict({ a: discrete(2), b: multiDiscrete([2]) }))).toBe('discrete')
  })

  it('makes environments of every form that reset', () => {
    const single = lookup(environmentRegistry, 'mountainCar')
    if (single.multiAgent !== undefined) throw new Error('single-agent expected')
    expect(single.make().reset(key(0)).observation).toHaveLength(2)
    expect(single.make({ goalVelocity: 0.1 }).spec.id).toBe('MountainCar-v0')
    const game = lookup(environmentRegistry, 'ticTacToe')
    if (game.multiAgent !== 'aec') throw new Error('aec expected')
    const g = game.make()
    expect(g.currentAgent(g.reset(key(0)))).toBe('player_0')
    const parallel = lookup(environmentRegistry, 'matrixGame')
    if (parallel.multiAgent !== 'parallel') throw new Error('parallel expected')
    expect(Object.keys(parallel.make().reset(key(0)).observations)).toEqual(['player_0', 'player_1'])
  })

  it('returns everything for an empty query and filters by family, kinds, episodic, multi-agent and text', () => {
    expect(whereEnvironments(environmentRegistry)).toEqual(environmentRegistry.entries)
    expect(
      whereEnvironments(environmentRegistry, { actions: 'continuous', episodic: true }).map((e) => e.name),
    ).toEqual(['mountainCarContinuous', 'cartDoublePendulum'])
    expect(whereEnvironments(environmentRegistry, { observations: 'grid' }).map((e) => e.family)).toEqual([
      ...Array<string>(5).fill('minatar'),
    ])
    expect(whereEnvironments(environmentRegistry, { family: ['toyText', 'bandit'], episodic: false }).length).toBe(7)
    expect(whereEnvironments(environmentRegistry, { multiAgent: true }).length).toBe(9)
    expect(whereEnvironments(environmentRegistry, { multiAgent: false, family: 'game' })).toEqual([])
    expect(whereEnvironments(environmentRegistry, { text: 'frozen LAKE' }).map((e) => e.name)).toEqual([
      'frozenLake',
      'frozenLake8x8',
    ])
    expect(whereEnvironments(environmentRegistry, { text: 'Taxi-v4' }).map((e) => e.name)).toEqual(['taxi'])
    expect(
      matchesEnvironmentQuery(lookup(environmentRegistry, 'go'), { observations: 'vector', actions: 'discrete' }),
    ).toBe(true)
  })

  it('derives entries from definitions and extends into a new registry', () => {
    const entry = environmentEntry({ definition: pendulum, title: 'P', family: 'classicControl', episodic: false })
    expect(entry.name).toBe('Pendulum-v1')
    expect(entry.description).toBe(pendulum.description)
    expect(entry.options).toBe(pendulum.options)
    const mine = registerEnvironments(environmentRegistry, [
      { definition: pendulum, name: 'myPendulum', title: 'My pendulum', family: 'classicControl', episodic: false },
    ])
    expect(mine.entries.length).toBe(environmentRegistry.entries.length + 1)
    expect(findEnvironment(environmentRegistry, 'myPendulum')).toBeUndefined()
    expect(findEnvironment(mine, 'myPendulum')?.spec.id).toBe('Pendulum-v1')
    expect(() =>
      environmentEntry({
        definition: {
          id: 'Empty-v0',
          description: 'no agents',
          multiAgent: 'parallel',
          create: () => ({ ...lookup(environmentRegistry, 'matrixGame').make(), possibleAgents: [] }) as never,
        },
        title: 'Empty',
        family: 'multiAgent',
        episodic: false,
      }),
    ).toThrow(/no agents/)
  })
})
