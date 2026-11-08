/**
 * For every exported function of the core: does its signature take or return native numeric arrays (`number[]`,
 * `number[][]`, `Float64Array`, `ArrayLike<number>`) rather than tensors (`TensorLike`, `Value`, `Tensor`)? The rule of
 * the library: tensors in public numeric signatures, native arrays only in the primitives (kernels, `blas.ts`, private
 * `_` modules). Fixed-size tuples (`Vec3`, `Rgba`) are not arrays here.
 *
 *   node scripts/review/signatures.mjs
 */
import { relative } from 'node:path'
import { CORE, domainOf, LIB, listFor, ROOT, relOf } from './_shared.mjs'

const { readProject } = await import('../../packages/meta/src/extract/project.ts')
const { readSymbols } = await import('../../packages/meta/src/extract/symbols.ts')
const { symbols } = readSymbols(readProject(ROOT, ['engine']))

const NATIVE = /\bnumber\[\]|\bFloat64Array\b|\bArrayLike<number>|\bFloat32Array\b/
const TENSOR = /\b(TensorLike|Value|Tensor)\b/
/** Splits one declaration line into its parameter text and its return type, by matching the first top-level parentheses. */
function split(line) {
  const open = line.indexOf('(')
  if (open < 0) return undefined
  let depth = 0
  for (let i = open; i < line.length; i++) {
    if (line[i] === '(') depth++
    else if (line[i] === ')' && --depth === 0) return { params: line.slice(open + 1, i), ret: line.slice(i + 1) }
  }
  return undefined
}
/** Private by convention: a kernel or helper module the rule allows to use raw buffers. */
const allowed = (file) => /\/_[^/]*$/.test(file) || /\/blas\.ts$/.test(file) || /\/tensor\/_cpu\//.test(file)

const rows = []
for (const s of symbols) {
  if (s.kind !== 'function' || s.package !== 'omai-engine') continue
  const file = relative(ROOT, s.source.file)
  const full = `${ROOT}/${file}`
  if (!full.startsWith(LIB) || !CORE.has(domainOf(full))) continue
  let params = false
  let ret = false
  let tensor = false
  for (const line of s.signature.split('\n')) {
    const parts = split(line)
    if (parts === undefined) continue
    params ||= NATIVE.test(parts.params)
    ret ||= NATIVE.test(parts.ret)
    tensor ||= TENSOR.test(line)
  }
  rows.push({ name: s.name, domain: domainOf(full), file: relOf(full), params, ret, tensor, allowed: allowed(file) })
}
const pub = rows.filter((r) => !r.allowed)
const bad = (r) => r.params || r.ret
console.log(
  `PUBLIC NUMERIC SIGNATURES (core exported functions outside private/kernel modules): ${pub.length} of ${rows.length} functions`,
)
console.log(
  `  native arrays in a signature: ${pub.filter(bad).length}; tensor types in a signature: ${pub.filter((r) => r.tensor).length}`,
)
console.log('\ndomain       functions  native in params  native in return  either   share')
const byDomain = new Map()
for (const r of pub) listFor(byDomain, r.domain).push(r)
for (const [d, xs] of [...byDomain].sort((a, b) => b[1].filter(bad).length - a[1].filter(bad).length)) {
  const p = xs.filter((r) => r.params).length
  const q = xs.filter((r) => r.ret).length
  const e = xs.filter(bad).length
  console.log(
    `${d.padEnd(11)} ${String(xs.length).padStart(9)} ${String(p).padStart(17)} ${String(q).padStart(17)} ${String(e).padStart(8)}  ${((100 * e) / xs.length).toFixed(0).padStart(4)}%`,
  )
}
console.log('\nFILES WITH THE MOST NATIVE-ARRAY SIGNATURES:')
const byFile = new Map()
for (const r of pub.filter(bad)) listFor(byFile, r.file).push(r)
for (const [f, xs] of [...byFile].sort((a, b) => b[1].length - a[1].length).slice(0, 22)) {
  console.log(
    `  ${String(xs.length).padStart(3)} in ${f}: ${xs
      .slice(0, 5)
      .map((r) => r.name)
      .join(', ')}`,
  )
}
