import { mcmc } from 'omai-engine/probprog'
import { toProgram } from 'omai-engine/probprog/model'
import { key } from 'omai-engine/random'
import { mean, std, toFlat } from 'omai-engine/tensor'
import { describe, expect, it } from 'vitest'
import { dataOf, ratingModel } from './compute.ts'
import { drawRatings } from './data.ts'
import { REFERENCE } from './reference.ts'

// The sampler takes about three minutes, so this runs only when asked: OMAI_REFERENCE=1.
describe.skipIf(process.env['OMAI_REFERENCE'] === undefined)('the stored sampler reference', () => {
  it('is what the sampler gives for the ratings of the page', () => {
    const posterior = mcmc(toProgram(ratingModel.bind(dataOf(drawRatings()))), undefined, {
      key: key(1),
      numChains: REFERENCE.chains,
      numWarmup: REFERENCE.warmup,
      numDraws: REFERENCE.draws,
    })
    const cuts = posterior.draws?.['cuts']
    if (cuts === undefined) throw new Error('the sampler returned no draws')
    const m = toFlat(mean(cuts, 0))
    const s = toFlat(std(cuts, 0))
    for (const [k, v] of REFERENCE.cutMean.entries()) expect(m[k]).toBeCloseTo(v, 1)
    for (const [k, v] of REFERENCE.cutSd.entries()) expect(Math.abs((s[k] as number) - v)).toBeLessThan(0.05)
  }, 900000)
})
