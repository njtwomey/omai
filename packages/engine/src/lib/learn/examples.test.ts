// Runs every TSDoc @example of the learn domain (and its subfolders) and checks each `// => value` line.
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import * as autodiff from '../autodiff/index.ts'
import * as base from '../base/index.ts'
import * as data from '../data/index.ts'
import * as exposition from '../exposition/index.ts'
import * as metrics from '../metrics/index.ts'
import * as random from '../random/index.ts'
import * as space from '../space/index.ts'
import * as stats from '../stats/index.ts'
import * as tensor from '../tensor/index.ts'
import * as trace from '../trace/index.ts'
import * as learn from './index.ts'

const MODULES: Readonly<Record<string, Readonly<Record<string, unknown>>>> = {
  'whitebox-engine/autodiff': autodiff,
  'whitebox-engine/base': base,
  'whitebox-engine/data': data,
  'whitebox-engine/exposition': exposition,
  'whitebox-engine/metrics': metrics,
  'whitebox-engine/random': random,
  'whitebox-engine/space': space,
  'whitebox-engine/stats': stats,
  'whitebox-engine/tensor': tensor,
  'whitebox-engine/trace': trace,
  'whitebox-engine/learn': learn,
  ...Object.fromEntries(
    Object.entries(learn)
      .filter(([, v]) => typeof v === 'object' && v !== null && !Array.isArray(v))
      .flatMap(([k, v]) => {
        const folder = k.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`)
        return [[`whitebox-engine/learn/${folder}`, v as Readonly<Record<string, unknown>>]]
      }),
  ),
}

interface Example {
  readonly file: string
  readonly line: number
  readonly code: string[]
}

function examples(file: string): Example[] {
  const text = readFileSync(join(import.meta.dirname, file), 'utf8').split('\n')
  const out: Example[] = []
  for (let i = 0; i < text.length; i++) {
    if (!/^\s*\*\s*```ts\s*$/.test(text[i] ?? '')) continue
    const code: string[] = []
    const start = i
    for (i++; i < text.length && !/^\s*\*\s*```\s*$/.test(text[i] ?? ''); i++) {
      code.push((text[i] ?? '').replace(/^\s*\*\s?/, ''))
    }
    out.push({ file, line: start + 1, code })
  }
  return out
}

function compileExample(code: readonly string[]): string {
  const body: string[] = []
  for (const line of code) {
    const imp = /^import\s+\{([^}]*)\}\s+from\s+'([^']+)'/.exec(line)
    if (imp !== null) {
      const names = (imp[1] ?? '').replace(/\btype\s+\w+\s*,?/g, '').replace(/\s+as\s+/g, ': ')
      body.push(`const {${names}} = __mods[${JSON.stringify(imp[2])}];`)
      continue
    }
    const throws = /^(.*?)\s*\/\/ => throws (\w[\w-]*)$/.exec(line)
    if (throws !== null) {
      body.push(`__throws(() => (${throws[1]}), ${JSON.stringify(throws[2])}, ${JSON.stringify(line)});`)
      continue
    }
    const traced = /^(.*?)\s*\/\/ => (trace|figure)$/.exec(line)
    if (traced !== null) {
      body.push(`${traced[1]};`)
      continue
    }
    const check = /^(.*?)\s*\/\/ => (.*)$/.exec(line)
    if (check !== null) {
      body.push(`__check(() => (${check[1]}), () => (${check[2]}), ${JSON.stringify(line)});`)
      continue
    }
    body.push(line)
  }
  return body.join('\n')
}

function sources(dir: string, prefix = ''): string[] {
  const out: string[] = []
  for (const name of readdirSync(join(import.meta.dirname, dir)).sort()) {
    const rel = prefix === '' ? name : `${prefix}/${name}`
    if (statSync(join(import.meta.dirname, rel)).isDirectory()) out.push(...sources(rel, rel))
    else if (name.endsWith('.ts') && !name.includes('.test') && !name.startsWith('_') && name !== 'index.ts')
      out.push(rel)
  }
  return out
}

const files = sources('.')

describe('documentation examples', () => {
  for (const file of files) {
    for (const ex of examples(file)) {
      it(`${file}:${ex.line}`, () => {
        const run = new Function('__mods', '__check', '__throws', compileExample(ex.code)) as (
          mods: typeof MODULES,
          check: (actual: () => unknown, expected: () => unknown, text: string) => void,
          throws: (f: () => unknown, code: string, text: string) => void,
        ) => void
        run(
          MODULES,
          (actual, expected, text) => {
            expect(actual(), text).toEqual(expected())
          },
          (f, code, text) => {
            expect(f, text).toThrow()
            try {
              f()
            } catch (e) {
              expect(base.isWhiteboxError(e, code as base.ErrorCode), text).toBe(true)
            }
          },
        )
      })
    }
  }
})
