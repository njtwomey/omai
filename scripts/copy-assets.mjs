/**
 * Copies the stylesheets that `tsc` leaves behind: every `.css` file under `src/` goes to the same place under `dist/`, so
 * the `import './maths.css'` that stays in the emitted JavaScript resolves. Run from a package folder after `tsc` (the
 * `build` script of omai-render does). It then checks that every relative `.css` import in `dist/` has its file, and fails
 * the build if one does not.
 */
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'

const root = process.cwd()
const walk = (dir) =>
  readdirSync(dir).flatMap((name) => {
    const path = join(dir, name)
    return statSync(path).isDirectory() ? walk(path) : [path]
  })

for (const file of walk(join(root, 'src')).filter((f) => f.endsWith('.css'))) {
  const target = file.replace(join(root, 'src'), join(root, 'dist'))
  mkdirSync(dirname(target), { recursive: true })
  cpSync(file, target)
}

const missing = walk(join(root, 'dist'))
  .filter((f) => f.endsWith('.js'))
  .flatMap((f) =>
    [...readFileSync(f, 'utf8').matchAll(/import\s+['"](\.{1,2}\/[^'"]+\.css)['"]/g)]
      .map((m) => resolve(dirname(f), m[1]))
      .filter((css) => !existsSync(css))
      .map((css) => `${f} imports ${css}`),
  )
if (missing.length > 0) {
  console.error(`stylesheets imported from dist/ but not found:\n${missing.join('\n')}`)
  process.exit(1)
}
