// The key of a group operator's tilt: one key for the cavities of the ports of a sweep (equal to rounding), a different one for cavities
// that differ in a small entry, in the power, or in what kind of argument they are.
import { describe, expect, it } from 'vitest'
import { messageArg, valueArg } from '../../core/args.ts'
import { gaussian, gaussianEta } from '../../core/families.ts'
import { gaussianMv, gaussianMvEta } from '../../core/families-vector.ts'
import type { Arg, Eta, OperatorContext } from '../../core/types.ts'
import { tiltKey } from './_group.ts'

const context = (args: readonly Arg[], power = 1): OperatorContext => ({ mode: 'ep', to: 0, args, params: {}, power })
const mv = (eta: Eta): Arg => messageArg(gaussianMv(2), eta)

describe('tiltKey', () => {
  it('is one key for the cavity of each port: the same numbers to rounding, and a zero that comes back as 1e-17', () => {
    // [h1, h2, J11, J12, J21, J22]-shaped message of a belief of size 1e3
    const a = mv([3.5, 0, 1000, 0, 0, 2000])
    const b = mv([3.5 * (1 + 3e-16), 1e-17, 1000 * (1 - 2e-16), -1e-17, 1e-17, 2000 * (1 + 1e-16)])
    expect(tiltKey('g', context([a]))).toBe(tiltKey('g', context([b])))
  })

  it('tells apart cavities that differ only in entries far below the largest of their argument', () => {
    // the old key rounded to 1e-10 of the largest entry (1e3, so a step of 1e-7): both entries below it were 0 and the cavities one key
    const small = (j: number): Arg => mv([1e-8, 2e-8, 1000, 0, 0, j])
    expect(tiltKey('g', context([small(1e-9)]))).not.toBe(tiltKey('g', context([small(5e-8)])))
    const h = (v: number): Arg => mv([v, 0, 1000, 0, 0, 1000])
    expect(tiltKey('g', context([h(1e-9)]))).not.toBe(tiltKey('g', context([h(3e-9)])))
  })

  it('tells apart different powers, kinds of argument and families', () => {
    const w = gaussianMvEta([0, 0], [1, 0, 0, 1])
    const message = mv(w)
    expect(tiltKey('g', context([message], 1))).not.toBe(tiltKey('g', context([message], 0.5)))
    // an observed value and a message with the same numbers are not the same input
    const value = messageArg(gaussian, gaussianEta(1, 2))
    const observed = valueArg(gaussian, 1)
    expect(tiltKey('g', context([value]))).not.toBe(tiltKey('g', context([observed])))
    expect(tiltKey('g', context([messageArg(gaussian, [1, 2])]))).not.toBe(
      tiltKey('g', context([mv([1, 2, 0, 0, 0, 0])])),
    )
    expect(tiltKey('a', context([message]))).not.toBe(tiltKey('b', context([message])))
  })
})
