/**
 * The sampler's answer for the ratings the page shows, stored so the page can compare with it. NUTS (the library's `mcmc`, key 1)
 * was run once on all the ratings of `drawRatings()` with the model of `compute.ts`: 4 chains, 500 warm-up iterations and
 * 1000 draws each, which took about three minutes. A sampler is the slow way to fit this model, so it is not run in the page.
 * `reference.test.ts` repeats the run when the environment variable `OMAI_REFERENCE` is set and checks these numbers.
 */

export const REFERENCE = {
  chains: 4,
  warmup: 500,
  draws: 1000,
  /** The posterior mean and standard deviation of each of the four cut points. */
  cutMean: [-2.232, -1.479, -0.592, 1.174],
  cutSd: [0.379, 0.337, 0.312, 0.323],
  /** The posterior standard deviation of each item's quality. */
  qualitySd: [
    0.481, 0.542, 0.493, 0.479, 0.558, 0.51, 0.523, 0.587, 0.427, 0.588, 0.407, 0.541, 0.72, 0.472, 0.519, 0.825, 0.54,
    0.644, 0.503, 0.478,
  ],
} as const
