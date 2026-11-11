// The code shown at the foot of every Bayesian example page: its regions exist in the library's source for every choice of the
// knobs that picks a different piece, contain the model's `defineModel`, the inference lines, and keep the shown style.

import { defaults } from 'omai-engine/space'
import type { Config } from 'omai-engine/trace'
import { describe, expect, it } from 'vitest'
import {
  classifierExample,
  coinExample,
  gpClassificationExample,
  irtExample,
  linearExample,
  mixtureExample,
  radonExample,
  robustExample,
  schoolsExample,
} from '../examples/index.ts'
import { codeOf, exampleCode, sourcesOf } from '../shared/exampleCode.ts'
import { styleProblems } from '../shared/styleScan.ts'
import { codeOf as sbcCode } from './calibration-sbc/shown.ts'

const knobs: Record<string, Config> = {
  classify: defaults(classifierExample.method.config) as Config,
  'coin-flipping': defaults(coinExample.method.config) as Config,
  'linear-regression': defaults(linearExample.method.config) as Config,
  'robust-regression': defaults(robustExample.method.config) as Config,
  'eight-schools': defaults(schoolsExample.method.config) as Config,
  radon: defaults(radonExample.method.config) as Config,
  'gaussian-mixture': defaults(mixtureExample.method.config) as Config,
  'item-response': defaults(irtExample.method.config) as Config,
  'gp-classification': defaults(gpClassificationExample.method.config) as Config,
  'bayesian-gam': {},
}

const cases = Object.entries(exampleCode).flatMap(([slug, code]) =>
  code.variants.map((variant) => [slug, JSON.stringify(variant), variant] as const),
)

describe('the code shown on the example pages', () => {
  it('has an entry for every example page', () => {
    expect(Object.keys(exampleCode).sort()).toEqual(Object.keys(knobs).sort())
  })

  it.each(cases)(
    '%s %s: every piece exists, the model is defined and the inference follows',
    async (slug, _, variant) => {
      const entry = exampleCode[slug]
      if (entry === undefined) throw new Error(slug)
      const config = { ...knobs[slug], ...variant }
      const { code, file, missing } = codeOf(entry, config, await sourcesOf(entry))
      expect(missing).toEqual([])
      expect(file).not.toBe('')
      expect(code).toMatch(/defineModel/)
      expect(code).toMatch(slug === 'bayesian-gam' ? /useSession/ : /inferenceMethodFor/)
      expect(code).toMatch(/methodSpace/)
      expect(code.split('\n').length).toBeLessThan(110)
    },
  )

  it.each(cases)(
    '%s %s: one statement per line, no latent inside a call, comments above, short lines',
    async (slug, _, variant) => {
      const entry = exampleCode[slug]
      if (entry === undefined) throw new Error(slug)
      const config = { ...knobs[slug], ...variant }
      const { code } = codeOf(entry, config, await sourcesOf(entry))
      expect(styleProblems(code)).toEqual([])
    },
  )
})

describe('the code shown on the calibration page', () => {
  it.each(['nuts', 'mean-field'])('%s: the model, the fit and the run, in the shown style', (id) => {
    const code = sbcCode(id)
    expect(code).toMatch(/defineModel/)
    expect(code).toMatch(id === 'nuts' ? /nutsMethod/ : /sviMethod/)
    expect(code).toMatch(/sbcMethod/)
    expect(styleProblems(code)).toEqual([])
  })
})
