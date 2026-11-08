import { environmentRegistry } from 'omai-engine/environments'
import { bool, box, choice, dict, discrete, int, multiBinary, multiDiscrete, real, tuple } from 'omai-engine/space'
import { describe, expect, it } from 'vitest'
import { entryOf, firstFrame, formatSpace, makeSelection, selectEnvironment } from './environments.ts'

describe('environment selections', () => {
  it('start an entry from its default options and make it', () => {
    const s = selectEnvironment(environmentRegistry, 'frozenLake')
    expect(s.name).toBe('frozenLake')
    expect(s.options['isSlippery']).toBe(true)
    expect(selectEnvironment(environmentRegistry).name).toBe(environmentRegistry.entries[0]?.name)
    expect(selectEnvironment(environmentRegistry, 'acrobot').options).toEqual({})
    const env = makeSelection(environmentRegistry, { ...s, options: { ...s.options, isSlippery: false } })
    expect(env.spec.id).toBe('FrozenLake-v1')
    expect(() => entryOf(environmentRegistry, 'nope')).toThrow(/no entry 'nope'/)
  })

  it('draw a first frame of an environment of any form', () => {
    for (const name of ['cartPole', 'ticTacToe', 'matrixGame', 'pendulumModel']) {
      const scene = firstFrame(makeSelection(environmentRegistry, selectEnvironment(environmentRegistry, name)), 1)
      expect(scene.width, name).toBeGreaterThan(0)
    }
  })

  it('format spaces as gymnasium prints them', () => {
    expect(formatSpace(discrete(4))).toBe('Discrete(4)')
    expect(formatSpace(discrete(4, { start: 1 }))).toBe('Discrete(4, start 1)')
    expect(formatSpace(box(-1, 1, [2]))).toBe('Box(-1, 1, [2])')
    expect(formatSpace(box([-1.2, -0.07], [0.6, 0.07], [2]))).toBe('Box([-1.2, -0.07], [0.6, 0.07], [2])')
    expect(formatSpace(box(-Number.MAX_VALUE, Number.MAX_VALUE, [4]))).toBe('Box(-inf, inf, [4])')
    expect(formatSpace(box([0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6], [6]))).toBe('Box(0, [1 to 6], [6])')
    expect(formatSpace(box(0, 1, [10, 10, 4]))).toBe('Box(0, 1, [10, 10, 4])')
    expect(formatSpace(multiDiscrete([2, 3]))).toBe('MultiDiscrete([2, 3])')
    expect(formatSpace(multiBinary(4))).toBe('MultiBinary(4)')
    expect(formatSpace(tuple([discrete(32), discrete(11)]))).toBe('Tuple(Discrete(32), Discrete(11))')
    expect(formatSpace(dict({ a: discrete(2), b: box(0, 1, [1]) }))).toBe('Dict(a: Discrete(2), b: Box(0, 1, [1]))')
    expect(formatSpace(real(1e-4, 10, { log: true }))).toBe('Real(0.0001, 10, log)')
    expect(formatSpace(int(1, 10))).toBe('Int(1, 10)')
    expect(formatSpace(choice(['a', 'b']))).toBe('Choice(a | b)')
    expect(formatSpace(bool())).toBe('Bool')
  })
})
