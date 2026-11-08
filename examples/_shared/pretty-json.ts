/**
 * JSON written for people to read: objects one key per line with two-space indentation, and arrays of numbers,
 * strings or other short values filled onto lines of a limited width, so a tape of a few hundred steps stays a few
 * hundred lines. The text parses back to the same value as `JSON.stringify`.
 */

const WIDTH = 110

const isLeaf = (v: unknown): boolean => v === null || typeof v !== 'object'

/** Whether an array can be written on one line: every element is a leaf, or a short array/object of leaves. */
const flat = (v: unknown): boolean =>
  isLeaf(v) || (Array.isArray(v) ? v.every(isLeaf) : Object.values(v as Record<string, unknown>).every(isLeaf))

/**
 * The text of a value, indented.
 *
 * @param value - Anything JSON can hold.
 * @param indent - The indentation of the line the text starts on.
 * @returns The pretty text, without a trailing newline.
 */
export function prettyJson(value: unknown, indent = ''): string {
  if (isLeaf(value)) return JSON.stringify(value) ?? 'null'
  const inner = `${indent}  `
  if (Array.isArray(value)) {
    if (value.length === 0) return '[]'
    const one = JSON.stringify(value)
    if (value.every(flat) && indent.length + one.length <= WIDTH && value.every(isLeaf)) return one
    if (value.every(isLeaf)) {
      // numbers and strings: fill lines up to the width
      const lines: string[] = []
      let line = ''
      for (const item of value) {
        const text = `${JSON.stringify(item)},`
        if (line !== '' && inner.length + line.length + 1 + text.length > WIDTH) {
          lines.push(line)
          line = text
        } else line = line === '' ? text : `${line} ${text}`
      }
      lines.push(line)
      lines[lines.length - 1] = (lines[lines.length - 1] as string).replace(/,$/, '')
      return `[\n${lines.map((l) => `${inner}${l}`).join('\n')}\n${indent}]`
    }
    return `[\n${value.map((v) => `${inner}${flat(v) && JSON.stringify(v).length + inner.length <= WIDTH ? JSON.stringify(v) : prettyJson(v, inner)}`).join(',\n')}\n${indent}]`
  }
  const entries = Object.entries(value as Record<string, unknown>).filter(([, v]) => v !== undefined)
  if (entries.length === 0) return '{}'
  const body = entries.map(([k, v]) => `${inner}${JSON.stringify(k)}: ${prettyJson(v, inner)}`).join(',\n')
  return `{\n${body}\n${indent}}`
}
