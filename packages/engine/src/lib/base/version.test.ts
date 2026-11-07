import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { VERSION } from './version.ts'

describe('VERSION', () => {
  it('is the version of the package manifest it is recorded from', () => {
    const manifest = JSON.parse(readFileSync(new URL('../../../package.json', import.meta.url), 'utf8')) as {
      readonly version: string
    }
    expect(VERSION).toBe(manifest.version)
  })
})
