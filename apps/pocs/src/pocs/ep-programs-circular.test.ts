// The circular-times program (the post "Circular statistics without circular distributions"): its claims as assertions, simulated from the
// post's two truths over a few seeds and fitted by EP. Shifting the means (the paper) and shifting the data (the post) are the same
// arithmetic; extra rows fit a far too wide sd; the plain mixture misfits data that straddle midnight; the two that wrap recover the truth.
import { describe, expect, it } from 'vitest'
import {
  type CircularRun,
  circularGap,
  copiesError,
  heldOutHours,
  heldOutLoss,
  lossesAt,
  lossGap,
  matchToTruth,
  mixtureOf,
  randomMixtures,
  runCircular,
  truthScale,
} from './ep-program-circular/compute.ts'
import { simulate, TRUTHS, type TruthName, type Variant } from './ep-program-circular/program.ts'

const fit = (truth: TruthName, components: number, variant: Variant, seed: number, draws = 200): Promise<CircularRun> =>
  runCircular({ seed, truth, components, draws, variant })

const lossOf = (truth: TruthName, run: CircularRun): number =>
  heldOutLoss(run.setup.variant, truth, mixtureOf(run.found), heldOutHours(truth, run.setup.seed))

describe('the arithmetic: shift the means is shift the data', () => {
  it('the log likelihoods agree to rounding at random parameters, for one and for two components', async () => {
    for (const [truth, k] of [
      ['one', 1],
      ['two', 2],
    ] as const) {
      const hours = simulate(3, truth, 300)
      expect(lossGap(hours, k)).toBeLessThan(1e-9)
    }
  })
  it('extra rows are a different number: the sum of the logs, not the log of the sum', async () => {
    const hours = simulate(3, 'one', 300)
    const [mix] = randomMixtures(1, 1)
    const l = lossesAt(hours, mix as NonNullable<typeof mix>)
    expect(Math.abs(l.extraRows - l.shiftData)).toBeGreaterThan(10)
  })
  it('the three-copy bound is 8e-11 at sd 3 h, under 0.005 at 6 h and about 0.4 at 12 h', async () => {
    expect(copiesError(3)).toBeLessThan(1e-10)
    expect(copiesError(6)).toBeLessThan(0.005)
    expect(copiesError(12)).toBeGreaterThan(0.35)
  })
})

describe('a normal at 23:00, sd 3 h (it straddles midnight)', () => {
  it('ignoring the circle misfits: one component lands mid-day with a huge sd, and the held-out density is far below the truth', async () => {
    const run = await fit('one', 1, 'ignore', 1)
    expect(circularGap((run.found.means[0] as { mean: number }).mean, 23)).toBeGreaterThan(5)
    expect(run.found.sd.mean).toBeGreaterThan(7)
    expect(lossOf('one', run)).toBeGreaterThan(0.3)
  }, 60_000)

  it('copies as extra rows fit an sd far too wide (the post: 21.7 h against 3)', async () => {
    const run = await fit('one', 1, 'rows', 1)
    expect(run.found.sd.mean).toBeGreaterThan(15)
    expect(run.found.sd.mean).toBeLessThan(30)
  }, 60_000)

  it.each([
    ['means', [1, 2, 3]],
    ['data', [1, 2]],
  ] as const)(
    'shifting the %s recovers the mean 23 h and the sd 3 h on seeds %j',
    async (variant, seeds) => {
      for (const seed of seeds) {
        const run = await fit('one', 1, variant, seed)
        expect(run.fit.run.status, `seed ${seed}`).toBe('converged')
        const mean = (run.found.means[0] as { mean: number }).mean
        expect(circularGap(mean, 23), `seed ${seed} mean ${mean}`).toBeLessThan(0.6)
        expect(Math.abs(run.found.sd.mean - 3) / 3, `seed ${seed} sd ${run.found.sd.mean}`).toBeLessThan(0.15)
        expect(lossOf('one', run), `seed ${seed}`).toBeLessThan(0.03)
      }
    },
    120_000,
  )

  it('shift the means and shift the data give the same EP fit to a few hundredths of an hour', async () => {
    const a = await fit('one', 1, 'means', 2)
    const b = await fit('one', 1, 'data', 2, 200)
    expect(
      Math.abs((a.found.means[0] as { mean: number }).mean - (b.found.means[0] as { mean: number }).mean),
    ).toBeLessThan(0.25)
    expect(Math.abs(a.found.sd.mean - b.found.sd.mean)).toBeLessThan(0.25)
  }, 120_000)
})

describe('two normals: 22 h (0.6, sd 2.5) and 9 h (0.4, sd 2), one shared sd in the model', () => {
  it.each([
    ['means', [1, 2, 3]],
    ['data', [1, 3]],
  ] as const)(
    'shifting the %s recovers the means and weights (seeds %j)',
    async (variant, seeds) => {
      for (const seed of seeds) {
        const run = await fit('two', 2, variant, seed)
        const matched = matchToTruth('two', run.found)
        expect(run.fit.run.status, `seed ${seed}`).toBe('converged')
        TRUTHS.two.means.forEach((m, k) => {
          const f = run.found.means[matched[k] as number] as { mean: number }
          expect(circularGap(f.mean, m), `seed ${seed} mean ${m}`).toBeLessThan(1.2)
          const w = run.found.weights[matched[k] as number] as { mean: number }
          expect(Math.abs(w.mean - (TRUTHS.two.weights[k] as number)), `seed ${seed} weight ${k}`).toBeLessThan(0.12)
        })
        // One sd for two scales: close to their root mean square.
        expect(Math.abs(run.found.sd.mean - truthScale('two')) / truthScale('two'), `seed ${seed}`).toBeLessThan(0.25)
        expect(lossOf('two', run), `seed ${seed}`).toBeLessThan(0.1)
      }
    },
    240_000,
  )

  it('the plain mixture has no idea of midnight: its held-out density is further below the truth than the wrapped fit’s', async () => {
    const plain = await fit('two', 2, 'ignore', 1)
    const wrapped = await fit('two', 2, 'means', 1)
    expect(lossOf('two', plain)).toBeGreaterThan(lossOf('two', wrapped))
  }, 120_000)

  it('a known limit: EP on the data form can land in a bad mode (seed 2: one component at 14 h, sd 5.9). Flip this when EP improves', async () => {
    const run = await fit('two', 2, 'data', 2)
    expect(lossOf('two', run)).toBeGreaterThan(0.3)
    // The same readings under the mean form are fine.
    expect(lossOf('two', await fit('two', 2, 'means', 2))).toBeLessThan(0.1)
  }, 120_000)
})
