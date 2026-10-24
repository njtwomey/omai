import { describe, expect, it } from 'vitest'
import { tf } from '../../systems/index.ts'
import { cohenCoon, pid, relayAutotune, simulatePid, zieglerNichols } from './pid.ts'

describe('PID control', () => {
  it('tracks a step with zero steady-state error and filters the derivative', () => {
    const plant = tf([1], [1, 2, 1])
    const r = simulatePid(plant, pid({ kp: 3, ki: 1.5, kd: 1, dt: 0.01 }), { duration: 20 })
    expect(Math.abs((r.y.at(-1) ?? 0) - 1)).toBeLessThan(1e-3)
    // A derivative kick is absent with derivative on measurement (c = 0): the first output is the P + I terms.
    expect(r.u[0]).toBeCloseTo(3 + 1.5 * 0.01, 12)
  })

  it('anti-windup limits the overshoot under saturation', () => {
    const plant = tf([1], [10, 1])
    const base = { kp: 5, ki: 2, dt: 0.01, outputMin: -1, outputMax: 1 }
    const none = simulatePid(plant, pid({ ...base, antiWindup: 'none' }), { duration: 60, setpoint: 0.8 })
    const clamp = simulatePid(plant, pid({ ...base, antiWindup: 'clamping' }), { duration: 60, setpoint: 0.8 })
    const back = simulatePid(plant, pid({ ...base, antiWindup: 'back-calculation' }), { duration: 60, setpoint: 0.8 })
    expect(clamp.overshoot).toBeLessThan(none.overshoot)
    expect(back.overshoot).toBeLessThan(none.overshoot)
    expect(none.u.every((u) => Math.abs(u) <= 1)).toBe(true)
  })

  it('tuning rules and relay auto-tuning', () => {
    expect(zieglerNichols(4, 2, 'P')).toEqual({ kp: 2, ki: 0, kd: 0 })
    const pi = zieglerNichols(4, 2.4, 'PI')
    expect(pi.ki).toBeCloseTo(0.9, 12)
    const cc = cohenCoon(2, 5, 1, 'PI')
    expect(cc.kp).toBeCloseTo((1 / (2 * 0.2)) * (0.9 + 0.2 / 12), 12)
    expect(() => cohenCoon(1, 0, 1)).toThrow()
    const plant = tf([1], [1, 3, 3, 1])
    const r = relayAutotune(plant, { dt: 0.005, duration: 60 })
    expect(r.oscillating).toBe(true)
    expect(Math.abs(r.tu - (2 * Math.PI) / Math.sqrt(3))).toBeLessThan(0.1)
    // The ZN-tuned loop is stable.
    const loop = simulatePid(plant, pid({ ...r.gains, dt: 0.005 }), { duration: 40 })
    expect(Math.abs((loop.y.at(-1) ?? 0) - 1)).toBeLessThan(0.02)
    expect(() => pid({ kp: 1, dt: 0 })).toThrow()
  })
})
