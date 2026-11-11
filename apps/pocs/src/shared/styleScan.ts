// The style the code shown on a page keeps (the standing rule: one statement per line, a latent never created inside another
// expression, comments above the code and never trailing, lines under about 100 characters), as a scan of the text.

/**
 * What a shown piece of code does against the style: one message per line that breaks it.
 *
 * @param code - The code as shown.
 * @returns The offending lines, with the rule each breaks (empty when the code keeps the style).
 */
export function styleProblems(code: string): string[] {
  const bad: string[] = []
  for (const line of code.split('\n')) {
    const t = line.trim()
    if (t.startsWith('//')) continue
    if ((t.match(/\bm\.(latent|observe)\(/g) ?? []).length > 1) bad.push(`two statements: ${t}`)
    if (/\bm\.latent\(/.test(t) && !/^(const \w+ = )m\.latent\(/.test(t))
      bad.push(`latent inside another expression: ${t}`)
    if (/\S\s+\/\/ /.test(t.replace(/'[^']*'|`[^`]*`|"[^"]*"/g, ''))) bad.push(`trailing comment: ${t}`)
    if (line.length > 100) bad.push(`long line (${line.length}): ${t}`)
  }
  return bad
}
