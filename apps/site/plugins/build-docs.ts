/**
 * Precompile the docs catalogue (`make docs-catalogue`): extract the engine's catalogue and write the docs' static
 * files (docs-data.ts).
 *
 *   node apps/site/plugins/build-docs.ts [--if-stale] [--quiet]
 *
 * `--if-stale` returns at once when the files are newer than every source (so `make dev` stays instant).
 */
import { isStale, OUT, ROOT, writeDocsData } from './docs-data.ts'

const args = process.argv.slice(2)
const quiet = args.includes('--quiet')
const kb = (n: number): string => `${(n / 1024).toFixed(0)} KB`

if (args.includes('--if-stale') && !isStale()) {
  if (!quiet) process.stdout.write('docs catalogue: up to date\n')
} else {
  const t0 = performance.now()
  // Loaded only when needed: ts-morph takes a moment to import.
  const { extractCatalogue } = await import('../../../packages/meta/src/extract/index.ts')
  const catalogue = extractCatalogue({ root: ROOT, packages: ['engine'] })
  const t1 = performance.now()
  const r = writeDocsData(catalogue)
  const t2 = performance.now()
  if (!quiet)
    process.stdout.write(
      `docs catalogue: ${catalogue.files.length} files, ${catalogue.symbols.length} symbols in ${OUT}\n` +
        `  extracted in ${((t1 - t0) / 1000).toFixed(1)} s, written in ${((t2 - t1) / 1000).toFixed(2)} s\n` +
        `  index ${kb(r.index)}, search ${kb(r.search)}, file data ${kb(r.files)} (largest ${kb(r.largestFile)}), types ${kb(r.types)}\n`,
    )
}
