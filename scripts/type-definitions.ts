/**
 * Fail on type definitions that restate a canonical type instead of using it, so the vector, point, colour and complex
 * types stay singular (`Vec2`, `Vec3`, `Vec4` of `mdml-engine/base`):
 *
 *   node scripts/type-definitions.ts --check
 *
 * 1. A non-exported `type X = Y` where Y is a bare type name is a purposeless local alias: use the name `Y`.
 * 2. A `type` alias whose structure is an unlabelled fixed-size number tuple of two, three or four entries restates
 *    `Vec2`, `Vec3` or `Vec4`: use the canonical type, or a labelled tuple (`readonly [r: number, g: number, ...]`)
 *    when the element names carry meaning.
 * 3. An exported `type A = B` of a bare name needs a doc comment that says what the specialisation adds.
 *
 * An alias of a typed array (`type Vec = Float64Array`) is not flagged: the choice of representation is a separate question.
 * The right-hand side is normalised as `scripts/review/types.mjs` does (comments and `readonly` dropped), through the
 * helpers in `scripts/review/_shared.mjs`. Tests are exempt. A line may opt out with a trailing `// allow: <reason>`.
 */
import { readFileSync } from 'node:fs'
import { join, relative } from 'node:path'
import { aliasLines, isData, isTest, ROOT, walk } from './review/_shared.mjs'

/** Names that are not another definition's name: a bare keyword type is a domain name for a primitive. */
const PRIMITIVE = new Set(
  'string number boolean bigint symbol object unknown never any void null undefined true false'.split(' '),
)
const BARE_NAME = /^[A-Za-z_$][\w$]*(\.[A-Za-z_$][\w$]*)*$/
const TUPLE = /^\[number(,number){1,3}\]$/
const MIN_DOC_WORDS = 4
/** The hot-path typed arrays: `type Vec = Float64Array` names a representation, which is a design question of its own. */
const TYPED_ARRAY = /^(Big)?(Int|Uint|Float)(8|16|32|64)(Clamped)?Array$/

const problems: string[] = []
for (const root of ['packages', 'apps']) {
  for (const file of walk(join(ROOT, root))) {
    if (isTest(file) || isData(file)) continue
    const where = relative(ROOT, file)
    for (const a of aliasLines(readFileSync(file, 'utf8'))) {
      if (a.optOut) continue
      const bare = !a.generic && BARE_NAME.test(a.shape) && !PRIMITIVE.has(a.shape) && !TYPED_ARRAY.test(a.shape)
      if (TUPLE.test(a.shape)) {
        const size = a.shape.split(',').length
        problems.push(
          `${where}:${a.line}: \`${a.name}\` restates Vec${size}; use it, or a labelled tuple if the names matter`,
        )
      } else if (bare && !a.exported) {
        problems.push(`${where}:${a.line}: \`${a.name}\` is a local alias of \`${a.shape}\`; use \`${a.shape}\``)
      } else if (bare && a.doc.split(' ').filter((w) => /\w/.test(w)).length < MIN_DOC_WORDS) {
        problems.push(`${where}:${a.line}: \`${a.name}\` aliases \`${a.shape}\`; a doc comment must say what it adds`)
      }
    }
  }
}
if (problems.length > 0) {
  for (const p of problems) console.error(p)
  process.exit(1)
}
console.log('type-definitions: clean')
