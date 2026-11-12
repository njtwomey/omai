/**
 * The list of what the model-to-factor-graph compiler maps, written from its mapping tables: `docs/probprog/compiler-mappings.md`.
 *
 *   make ep-compiler-doc          write the page
 *   make ep-compiler-doc-check    fail (exit 1) when the page is out of date
 *
 * The page holds names and sentences only, so two runs of the same code give the same text.
 */
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { compilerMappingsDoc } from '../packages/engine/src/lib/probprog/model/factor-graph/_doc.ts'

const PAGE = join(import.meta.dirname, '../docs/probprog/compiler-mappings.md')
const text = compilerMappingsDoc()

if (process.argv.includes('--check')) {
  if (!existsSync(PAGE) || readFileSync(PAGE, 'utf8') !== text) {
    process.stdout.write('docs/probprog/compiler-mappings.md is out of date: run make ep-compiler-doc\n')
    process.exit(1)
  }
  process.stdout.write('compiler-mappings.md: up to date\n')
} else {
  writeFileSync(PAGE, text)
  process.stdout.write(`wrote ${PAGE}\n`)
}
