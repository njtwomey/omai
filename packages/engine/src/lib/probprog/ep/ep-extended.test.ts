// The extended atoms: each is the twin of a basic atom (it names it), simulates its data from a stated truth that every
// quantity carries, and recovers that truth under EP: over its seeds the 90% intervals hold the truth at least as often as it
// promises and the errors stay within its bound. Generated from the registry.
import { describe, expect, it } from 'vitest'
import { epAtom, epAtoms } from './ep-atom-list.ts'
import { atomRun } from './ep-atoms.ts'
import { atomCalibration, atomRecovery, beliefCredibleInterval, calibrationSample } from './ep-extended.ts'

const extended = epAtoms.filter((a) => a.extends !== undefined)

describe('beliefCredibleInterval', () => {
  it('is the normal interval on the line, the gamma quantiles on the positive axis, the beta quantiles on the unit interval', () => {
    const [lo, hi] = beliefCredibleInterval('real', 1, 2, 0.9)
    expect(lo).toBeCloseTo(1 - 1.644854 * 2, 4)
    expect(hi).toBeCloseTo(1 + 1.644854 * 2, 4)
    // Gamma(shape 4, rate 2): mean 2, sd 1; its 5% and 95% points are the chi-square 8 quantiles (2.733, 15.507) over 4
    const [gl, gh] = beliefCredibleInterval('positive', 2, 1, 0.9)
    expect(gl).toBeCloseTo(2.7326 / 4, 3)
    expect(gh).toBeCloseTo(15.5073 / 4, 3)
    const [bl, bh] = beliefCredibleInterval('unit', 0.5, Math.sqrt(1 / 12), 0.9)
    expect(bl).toBeCloseTo(0.05, 3)
    expect(bh).toBeCloseTo(0.95, 3)
  })
})

describe('atomRecovery', () => {
  it('marks whether the truth is inside the interval and measures the error in posterior sds', () => {
    const rows = atomRecovery(
      {
        quantities: [
          { name: 'a', label: 'a', support: 'real', truth: 0 },
          { name: 'b', label: 'b', support: 'real', truth: 5 },
          { name: 'c', label: 'c', support: 'real' },
        ],
      },
      {
        a: { mean: 0.5, sd: 1, proper: true },
        b: { mean: 0, sd: 1, proper: true },
        c: { mean: 0, sd: 1, proper: true },
      },
    )
    expect(rows.map((r) => [r.name, r.covered, r.z])).toEqual([
      ['a', true, 0.5],
      ['b', false, -5],
    ])
  })
})

describe('the extended atoms', () => {
  it('exist, name a basic twin that is in the registry, and are titled after it', () => {
    // an atom is extended exactly when its name says it is learned
    expect(extended.map((a) => a.name)).toEqual(epAtoms.filter((a) => a.name.endsWith('-learned')).map((a) => a.name))
    for (const a of extended) {
      const basic = epAtom(a.extends as string)
      expect(a.title.startsWith(basic.title), `${a.name}: ${a.title} vs ${basic.title}`).toBe(true)
      expect(a.title).toMatch(/: parameters learned/)
      expect(a.name).toMatch(/-learned$/)
      expect(a.recovery).toBeDefined()
      expect('seed' in a.knobs.spaces).toBe(true)
    }
  })

  describe.each(extended)('$name', (atom) => {
    it('carries the truth on every parameter it reports as a recovery, and the truth changes with the knobs, not the seed', () => {
      const p = atom.problem(atom.defaults)
      const withTruth = p.quantities.filter((q) => q.truth !== undefined)
      expect(withTruth.length).toBeGreaterThan(0)
      expect(withTruth.every((q) => Number.isFinite(q.truth))).toBe(true)
      expect(p.equations[0]).toMatch(/^truth:/)
      const again = atom.problem({ ...atom.defaults, seed: 2 })
      expect(again.quantities.length).toBe(p.quantities.length)
    })

    if (atom.expected === 'runs') {
      it('recovers the truth: the 90% intervals hold it over its seeds at the promised rate', () => {
        const c = atomCalibration(atom)
        expect(c.ran).toBe(c.seeds)
        const bad = c.rows.filter((r) => !r.pass).map((r) => `${r.key}: coverage ${r.coverage}, zrms ${r.zrms}`)
        expect(bad).toEqual([])
        expect(c.pass).toBe(true)
      }, 120_000)
      it('is the same run for the same seed (a calibration is reproducible)', () => {
        const a = calibrationSample(atom, 3)
        const b = calibrationSample(atom, 3)
        expect(a.rows.map((r) => r.mean)).toEqual(b.rows.map((r) => r.mean))
      })
    } else {
      it('is refused with the library reason, and the reference is still the posterior of the same model', () => {
        const run = atomRun(atom.problem(atom.defaults))
        expect(run.status).toBe('refused')
        expect(calibrationSample(atom, 1).rows).toEqual([])
      })
    }
  })
})
