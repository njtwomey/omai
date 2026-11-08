/**
 * Cut a release: the next version is the last released one plus 0.0.1.
 *
 *   node scripts/release.ts                 bump, commit and tag locally
 *   node scripts/release.ts --dry-run       print what would change and change nothing
 *   node scripts/release.ts --push          also push the commit and the tags to `origin`
 *
 * The last released version is the highest `v<version>` tag, on this clone or on `origin`. The three
 * packages are released in lockstep, as one version and one release, so the new version is written to `packages/{engine,render,meta}/package.json`
 * and to `packages/engine/src/lib/base/version.ts` (a test keeps the constant equal to the manifest), and the examples are
 * run again because their tapes carry the version. The changelog
 * must already have a `## <version>` entry. The commit is "Release <version>" and its tag is
 * `v<version>`; pushing the tag makes the `pack` workflow build the tarballs and publish the GitHub release. It refuses to run with uncommitted changes.
 */
import { spawnSync } from 'node:child_process'
import { readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

const ROOT = join(import.meta.dirname, '..')
const args = process.argv.slice(2)
const dry = args.includes('--dry-run')
const push = args.includes('--push')
const PACKAGES = ['engine', 'render', 'meta'] as const

function git(...a: string[]): string {
  const r = spawnSync('git', a, { cwd: ROOT, encoding: 'utf8' })
  if (r.status !== 0) throw new Error(`git ${a.join(' ')}: ${r.stderr.trim()}`)
  return r.stdout.trim()
}

type Version = readonly [number, number, number]
const parse = (v: string): Version | undefined => {
  const m = /^(\d+)\.(\d+)\.(\d+)$/.exec(v)
  return m ? [Number(m[1]), Number(m[2]), Number(m[3])] : undefined
}
const compare = (a: Version, b: Version): number => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]

/** The released versions: `v<version>` tags here and on origin. */
function released(): Version[] {
  const local = git('tag', '--list', 'v*').split('\n')
  let remote: string[] = []
  try {
    remote = git('ls-remote', '--tags', 'origin', 'v*')
      .split('\n')
      .map((l) => l.split('refs/tags/')[1] ?? '')
  } catch {
    // no network or no origin: the local tags are what there is
  }
  return [...local, ...remote]
    .map((t) => parse(t.replace(/^v/, '').replace(/\^\{\}$/, '')))
    .filter((v): v is Version => v !== undefined)
}

const last = released().sort(compare).at(-1)
if (last === undefined) throw new Error('release: no v<version> tag found')
const next = `${last[0]}.${last[1]}.${last[2] + 1}`
console.log(`release: last released ${last.join('.')}, next ${next}`)

const changelog = readFileSync(join(ROOT, 'CHANGELOG.md'), 'utf8')
if (!new RegExp(`^## ${next.replaceAll('.', '\\.')}\\s*$`, 'm').test(changelog)) {
  throw new Error(`release: CHANGELOG.md has no "## ${next}" entry: write it first`)
}
if (git('status', '--porcelain') !== '') {
  if (!dry) throw new Error('release: uncommitted changes: commit or stash them first')
  console.log('release (dry run): note that there are uncommitted changes, which a real run refuses')
}

const edits: { file: string; text: string }[] = []
for (const p of PACKAGES) {
  const file = `packages/${p}/package.json`
  const text = readFileSync(join(ROOT, file), 'utf8')
  const out = text.replace(/("version":\s*")[^"]+(")/, `$1${next}$2`)
  if (out === text && !text.includes(`"version": "${next}"`)) throw new Error(`release: no version field in ${file}`)
  edits.push({ file, text: out })
}
const versionFile = 'packages/engine/src/lib/base/version.ts'
const versionText = readFileSync(join(ROOT, versionFile), 'utf8')
const versionOut = versionText.replace(/(export const VERSION: string = ')[^']+(')/, `$1${next}$2`)
if (versionOut === versionText && !versionText.includes(`'${next}'`))
  throw new Error(`release: no VERSION in ${versionFile}`)
edits.push({ file: versionFile, text: versionOut })

const tag = `v${next}`
if (dry) {
  console.log(`release (dry run): would write ${edits.map((e) => e.file).join(', ')}`)
  console.log(`release (dry run): would commit "Release ${next}" and tag ${tag}${push ? ', then push' : ''}`)
  process.exit(0)
}
for (const e of edits) writeFileSync(join(ROOT, e.file), e.text)
// The tapes of the examples carry the library version, so they are written again for the new one and go in the commit
// (otherwise `make readme-examples-check` fails from the release on).
const examples = spawnSync(process.execPath, [join(ROOT, 'scripts/readme-examples.ts')], {
  cwd: ROOT,
  stdio: 'inherit',
})
if (examples.status !== 0)
  throw new Error('release: the examples failed to run again; the version files are written but not committed')
git('add', ...edits.map((e) => e.file), 'examples')
git('commit', '-m', `Release ${next}`)
git('tag', '-a', tag, '-m', next)
console.log(`release: committed "Release ${next}" and tagged ${tag}`)
if (push) {
  git('push', 'origin', 'HEAD')
  git('push', 'origin', tag)
  console.log('release: pushed the commit and the tag')
}
