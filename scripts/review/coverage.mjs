/**
 * Reads a v8/istanbul `coverage-final.json` (vitest `--coverage.reporter=json`) and reports the core domains: statement,
 * function and branch coverage per domain, the files with the most uncovered statements, and the functions no test or
 * example executes.
 *
 *   node scripts/review/coverage.mjs <coverage-final.json> [minimum statements for a file to be listed = 40]
 */
import { readFileSync } from 'node:fs'
import { CORE, domainOf, isBarrel, isData, isTest, LIB, listFor, relOf } from './_shared.mjs'

const MIN = Number(process.argv[3] ?? 40)
const data = JSON.parse(readFileSync(process.argv[2], 'utf8'))
const files = Object.entries(data)
  .filter(([f]) => f.startsWith(LIB) && !isTest(f) && !isData(f) && !isBarrel(f) && CORE.has(domainOf(f)))
  .map(([f, c]) => {
    const statements = Object.values(c.s)
    const branches = Object.values(c.b).flat()
    const missed = Object.entries(c.f)
      .filter(([, n]) => n === 0)
      .map(([id]) => c.fnMap[id])
    return {
      file: f,
      domain: domainOf(f),
      st: statements.length,
      stHit: statements.filter((n) => n > 0).length,
      fn: Object.keys(c.f).length,
      fnHit: Object.values(c.f).filter((n) => n > 0).length,
      br: branches.length,
      brHit: branches.filter((n) => n > 0).length,
      missed,
    }
  })
const pct = (a, b) => (b === 0 ? '  -  ' : `${((100 * a) / b).toFixed(1)}%`.padStart(5))
const sum = (list, k) => list.reduce((a, x) => a + x[k], 0)
const byDomain = new Map()
for (const x of files) listFor(byDomain, x.domain).push(x)
console.log(`CORE COVERAGE (${files.length} files; tests and examples of the engine, CI mode)`)
console.log('domain       files  statements  functions  branches   never-run functions')
for (const [d, xs] of [...byDomain].sort((a, b) => sum(b[1], 'st') - sum(a[1], 'st'))) {
  console.log(
    `${d.padEnd(11)} ${String(xs.length).padStart(5)}  ${pct(sum(xs, 'stHit'), sum(xs, 'st'))}      ${pct(sum(xs, 'fnHit'), sum(xs, 'fn'))}     ${pct(sum(xs, 'brHit'), sum(xs, 'br'))}     ${sum(xs, 'fn') - sum(xs, 'fnHit')}`,
  )
}
console.log(
  `TOTAL        ${String(files.length).padStart(5)}  ${pct(sum(files, 'stHit'), sum(files, 'st'))}      ${pct(sum(files, 'fnHit'), sum(files, 'fn'))}     ${pct(sum(files, 'brHit'), sum(files, 'br'))}     ${sum(files, 'fn') - sum(files, 'fnHit')}`,
)
const big = files.filter((x) => x.st >= MIN)
console.log(`\nFILES WITH THE MOST UNCOVERED STATEMENTS (files of ${MIN}+ statements):`)
for (const x of [...big].sort((a, b) => b.st - b.stHit - (a.st - a.stHit)).slice(0, 25)) {
  console.log(
    `  ${pct(x.stHit, x.st)} of ${String(x.st).padStart(4)} statements, ${String(x.fn - x.fnHit).padStart(3)} functions never run: ${relOf(x.file)}`,
  )
}
const zero = big.filter((x) => x.stHit === 0)
console.log(`\nFILES NO TEST REACHES (0 statements run, ${MIN}+ statements): ${zero.length}`)
for (const x of zero.slice(0, 20)) console.log(`  ${x.st} statements: ${relOf(x.file)}`)
console.log('\nNEVER-RUN FUNCTIONS, by file (the most, with names):')
for (const x of [...files].sort((a, b) => b.missed.length - a.missed.length).slice(0, 12)) {
  console.log(
    `  ${x.missed.length}/${x.fn} in ${relOf(x.file)}: ${x.missed
      .slice(0, 6)
      .map((m) => m.name)
      .join(', ')}`,
  )
}
