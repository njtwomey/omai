// Every EP atom of the registry has a page of the group, with the atom's title: the pages are generated from the registry.
import { epAtoms } from 'omai-engine/bayesian-examples'
import { describe, expect, it } from 'vitest'
import { POCS, pageOf } from '../registry.ts'
import { atomSource } from '../shared/ep/atomSource.ts'

describe('the pages of the EP atoms', () => {
  it.each(epAtoms.map((a) => [a.name, a.title] as const))(
    '%s has a page in the group, titled as the atom',
    (name, title) => {
      const entry = POCS.find((p) => p.slug === `ep-${name}`)
      expect(entry?.group).toBe('expectation-propagation')
      expect(entry?.title).toBe(title)
      expect(pageOf(`ep-${name}`)).toBeDefined()
    },
  )

  it('has a coverage page too', () => {
    expect(POCS.find((p) => p.slug === 'ep-coverage')?.group).toBe('expectation-propagation')
  })
})

describe('the code shown on the atom pages', () => {
  it.each(epAtoms.map((a) => [a.name, a] as const))(
    '%s has a region of its own model, short, and the inference lines',
    (_, atom) => {
      const { code, file } = atomSource(atom, atom.problem(atom.defaults))
      expect(file).toMatch(/^ep-atoms/)
      expect(code).toMatch(/defineModel|trueskillModel/)
      expect(code).toMatch(/epSpecMethod\(bound\)/)
      expect(code).toMatch(/session\(method/)
      expect(code.split('\n').length).toBeLessThan(60)
    },
  )
})

describe('the style of the code shown on the atom pages', () => {
  const lines = (code: string): string[] => code.split('\n')
  it.each(epAtoms.map((a) => [a.name, a] as const))(
    '%s: one statement per line, no latent inside a call, no trailing comments, short lines',
    (_, atom) => {
      const bad: string[] = []
      for (const line of lines(atomSource(atom, atom.problem(atom.defaults)).code)) {
        const t = line.trim()
        if (t.startsWith('//')) continue
        if ((t.match(/\bm\.(latent|observe)\(/g) ?? []).length > 1) bad.push(`two statements: ${t}`)
        if (/\bm\.latent\(/.test(t) && !/^(const \w+ = )m\.latent\(/.test(t))
          bad.push(`latent inside another expression: ${t}`)
        if (/\S\s+\/\/ /.test(t.replace(/'[^']*'|`[^`]*`|"[^"]*"/g, ''))) bad.push(`trailing comment: ${t}`)
        if (line.length > 100) bad.push(`long line (${line.length}): ${t}`)
      }
      expect(bad).toEqual([])
    },
  )
})
