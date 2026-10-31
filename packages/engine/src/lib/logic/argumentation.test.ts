import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { runAlgorithm } from '../trace/index.ts'
import {
  accepted,
  admissible,
  characteristic,
  completeExtensions,
  conflictFree,
  extensions,
  framework,
  groundedAlgorithm,
  groundedExtension,
  preferredExtensions,
  stableExtensions,
} from './index.ts'

const names = (e: readonly (readonly string[])[]): string[] => e.map((x) => x.join(',')).sort()

describe('classic frameworks', () => {
  it('a mutual attack (the Nixon diamond): nothing grounded, two preferred and two stable extensions', () => {
    const f = framework(
      ['a', 'b'],
      [
        ['a', 'b'],
        ['b', 'a'],
      ],
    )
    expect(groundedExtension(f)).toEqual([])
    expect(names(preferredExtensions(f))).toEqual(['a', 'b'])
    expect(names(stableExtensions(f))).toEqual(['a', 'b'])
    expect(names(completeExtensions(f))).toEqual(['', 'a', 'b'])
    expect(accepted(f, 'a', 'preferred', 'credulous')).toBe(true)
    expect(accepted(f, 'a', 'preferred', 'sceptical')).toBe(false)
  })

  it('an odd cycle has no stable extension and the empty preferred one', () => {
    const f = framework(
      ['a', 'b', 'c'],
      [
        ['a', 'b'],
        ['b', 'c'],
        ['c', 'a'],
      ],
    )
    expect(stableExtensions(f)).toEqual([])
    expect(preferredExtensions(f)).toEqual([[]])
    expect(groundedExtension(f)).toEqual([])
  })

  it('a chain a -> b -> c reinstates c: grounded {a, c}, the only stable one too', () => {
    const f = framework(
      ['a', 'b', 'c'],
      [
        ['a', 'b'],
        ['b', 'c'],
      ],
    )
    expect(groundedExtension(f)).toEqual(['a', 'c'])
    expect(names(stableExtensions(f))).toEqual(['a,c'])
    expect(characteristic(f, [])).toEqual(['a'])
  })

  it('a self-attacking argument is never accepted, and an attacked-only-by-it argument is reinstated', () => {
    const f = framework(
      ['a', 'b'],
      [
        ['a', 'a'],
        ['a', 'b'],
      ],
    )
    expect(groundedExtension(f)).toEqual([])
    expect(accepted(f, 'b', 'preferred', 'credulous')).toBe(false)
    expect(conflictFree(f, ['a'])).toBe(false)
  })

  it('an added argument flips the outcome: attacking a defender removes the grounded acceptance', () => {
    const base = framework(['a', 'b'], [['a', 'b']])
    expect(groundedExtension(base)).toEqual(['a'])
    const attacked = framework(
      ['a', 'b', 'c'],
      [
        ['a', 'b'],
        ['c', 'a'],
      ],
    )
    expect(groundedExtension(attacked)).toEqual(
      ['c', 'b'].sort((x, y) => attacked.arguments.indexOf(x) - attacked.arguments.indexOf(y)),
    )
  })

  it('rejects unknown arguments and duplicate names', () => {
    expect(() => framework(['a'], [['a', 'z']])).toThrow()
    expect(() => framework(['a', 'a'], [])).toThrow()
  })
})

describe('the theory of the semantics', () => {
  const arbFramework = fc.integer({ min: 1, max: 6 }).chain((n) => {
    const args = Array.from({ length: n }, (_, i) => `x${i}`)
    return fc
      .array(fc.tuple(fc.constantFrom(...args), fc.constantFrom(...args)), { maxLength: 10 })
      .map((attacks) => framework(args, attacks))
  })

  it('the grounded extension is the least complete extension, contained in every preferred one', () => {
    fc.assert(
      fc.property(arbFramework, (f) => {
        const g = groundedExtension(f)
        const complete = completeExtensions(f)
        expect(complete.some((c) => c.length === g.length && g.every((x) => c.includes(x)))).toBe(true)
        for (const c of complete) expect(g.every((x) => c.includes(x))).toBe(true)
        for (const p of preferredExtensions(f)) expect(g.every((x) => p.includes(x))).toBe(true)
      }),
      { numRuns: 200 },
    )
  })

  it('stable extensions are preferred, preferred are complete, and all are admissible', () => {
    fc.assert(
      fc.property(arbFramework, (f) => {
        const complete = names(completeExtensions(f))
        const preferred = names(preferredExtensions(f))
        for (const s of names(stableExtensions(f))) expect(preferred).toContain(s)
        for (const p of preferred) expect(complete).toContain(p)
        for (const c of completeExtensions(f)) expect(admissible(f, c)).toBe(true)
        expect(preferredExtensions(f).length).toBeGreaterThan(0)
      }),
      { numRuns: 200 },
    )
  })

  it('the labelling algorithm agrees with the grounded extension, one round per step', () => {
    fc.assert(
      fc.property(arbFramework, (f) => {
        const { state } = runAlgorithm(groundedAlgorithm(f), { maxSteps: 50, seed: 0 })
        const ins = f.arguments.filter((a) => state.labels[a] === 'in')
        expect(ins).toEqual(groundedExtension(f))
        // An argument is out exactly when attacked by an in argument.
        for (const a of f.arguments) {
          if (state.labels[a] === 'out')
            expect(f.attacks.some(([b, t]) => t === a && state.labels[b] === 'in')).toBe(true)
        }
      }),
      { numRuns: 200 },
    )
  })

  it('extensions() dispatches by semantics', () => {
    const f = framework(
      ['a', 'b'],
      [
        ['a', 'b'],
        ['b', 'a'],
      ],
    )
    expect(extensions(f, 'grounded')).toEqual([[]])
    expect(names(extensions(f, 'stable'))).toEqual(['a', 'b'])
  })
})
