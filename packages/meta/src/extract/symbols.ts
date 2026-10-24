/**
 * Exported declarations as {@link SymbolDoc}s: each module's export map is followed through its barrels to the
 * declaring files, every declaration is documented once at its canonical home (the entry point whose folder holds
 * it), and every other name it is exported under is recorded as an alias. Only the syntax tree is needed (no type
 * checker), so the whole engine reads in a few seconds.
 */
import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { ts } from 'ts-morph'
import type { MemberDoc, ModuleDoc, ParamDoc, SymbolDoc, SymbolKind } from '../schema/catalogue.ts'
import type { Example } from '../schema/examples.ts'
import { type EntryPoint, type Project, relativeTo } from './project.ts'
import { type ParsedDoc, parseDocComment, unframeComment } from './tsdoc.ts'

/** A declaration in a source file: its name, kind and nodes (several for an overloaded function). */
interface Declaration {
  readonly key: string
  readonly file: string
  readonly name: string
  readonly kind: SymbolKind
  readonly nodes: ts.Node[]
  readonly sf: ts.SourceFile
}

/** An exported name: a declaration, or a namespace re-export (`export * as x`). */
type Exported = { readonly decl: Declaration } | { readonly namespace: string }

/** What reading declarations reports besides the symbols. */
export interface SymbolsResult {
  readonly modules: readonly ModuleDoc[]
  readonly symbols: readonly SymbolDoc[]
}

const oneLine = (s: string): string => s.replace(/\s+/g, ' ').trim()

function modifiersOf(node: ts.Node): readonly ts.ModifierLike[] {
  return ts.canHaveModifiers(node) ? (ts.getModifiers(node) ?? []) : []
}

function isExported(node: ts.Node): boolean {
  return modifiersOf(node).some((m) => m.kind === ts.SyntaxKind.ExportKeyword)
}

function hasModifier(node: ts.Node, kind: ts.SyntaxKind): boolean {
  return modifiersOf(node).some((m) => m.kind === kind)
}

/** The doc comment directly before a node, unframed, with the 0-based file line of its first line. */
function docOf(sf: ts.SourceFile, node: ts.Node): { text: string; line: number } | undefined {
  const ranges = ts.getLeadingCommentRanges(sf.text, node.getFullStart()) ?? []
  const last = ranges.filter((r) => sf.text.startsWith('/**', r.pos)).at(-1)
  if (last === undefined) return undefined
  return {
    text: unframeComment(sf.text.slice(last.pos, last.end)),
    line: sf.getLineAndCharacterOfPosition(last.pos).line,
  }
}

const EMPTY_DOC: ParsedDoc = parseDocComment('')

function parsedDoc(
  project: Project,
  sf: ts.SourceFile,
  node: ts.Node,
): { doc: ParsedDoc; examples: Example[]; found: boolean } {
  const raw = docOf(sf, node)
  if (raw === undefined) return { doc: EMPTY_DOC, examples: [], found: false }
  const doc = parseDocComment(raw.text)
  const file = relativeTo(project, sf.fileName)
  const examples = doc.examples.map((e) => ({ ...e, source: { file, line: raw.line + (e.source?.line ?? 0) + 1 } }))
  return { doc, examples, found: true }
}

class Reader {
  private readonly files = new Map<string, ts.SourceFile>()
  private readonly declarations = new Map<string, Map<string, Declaration>>()
  private readonly exportMaps = new Map<string, Map<string, Exported>>()

  source(file: string): ts.SourceFile {
    let sf = this.files.get(file)
    if (sf === undefined) {
      const kind = file.endsWith('.tsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS
      sf = ts.createSourceFile(file, readFileSync(file, 'utf8'), ts.ScriptTarget.ES2022, true, kind)
      this.files.set(file, sf)
    }
    return sf
  }

  /** The top-level declarations of a file by name (exported or not, since `export { x }` may export them later). */
  declarationsOf(file: string): Map<string, Declaration> {
    const had = this.declarations.get(file)
    if (had !== undefined) return had
    const sf = this.source(file)
    const out = new Map<string, Declaration>()
    const add = (name: string, kind: SymbolKind, node: ts.Node): void => {
      const d = out.get(name)
      if (d !== undefined && d.kind === kind) d.nodes.push(node)
      else if (d === undefined || d.kind === 'type' || d.kind === 'interface') {
        // A value sharing its name with a type (`type Normal` and `function Normal`) documents both.
        out.set(name, { key: `${file}#${name}`, file, name, kind, nodes: [node], sf })
      }
    }
    for (const node of sf.statements) {
      if (ts.isFunctionDeclaration(node) && node.name !== undefined) add(node.name.text, 'function', node)
      else if (ts.isClassDeclaration(node) && node.name !== undefined) add(node.name.text, 'class', node)
      else if (ts.isInterfaceDeclaration(node)) add(node.name.text, 'interface', node)
      else if (ts.isTypeAliasDeclaration(node)) add(node.name.text, 'type', node)
      else if (ts.isEnumDeclaration(node)) add(node.name.text, 'enum', node)
      else if (ts.isVariableStatement(node)) {
        for (const d of node.declarationList.declarations) {
          if (!ts.isIdentifier(d.name)) continue
          const init = d.initializer
          const fn = init !== undefined && (ts.isArrowFunction(init) || ts.isFunctionExpression(init))
          add(d.name.text, fn ? 'function' : 'const', node)
        }
      }
    }
    this.declarations.set(file, out)
    return out
  }

  private resolveSpecifier(from: string, spec: string): string | undefined {
    if (!spec.startsWith('.')) return undefined
    const base = resolve(dirname(from), spec)
    const candidates = [base, base.replace(/\.js$/, '.ts'), `${base}.ts`, `${base}.tsx`, `${base}/index.ts`]
    for (const c of candidates) {
      try {
        if (/\.tsx?$/.test(c)) {
          this.source(c)
          return c
        }
      } catch {
        // Not this candidate.
      }
    }
    return undefined
  }

  /** The names a file exports, followed through `export *`, `export { a as b } from` and local `export { a }`. */
  exportsOf(file: string, seen: Set<string> = new Set()): Map<string, Exported> {
    const had = this.exportMaps.get(file)
    if (had !== undefined) return had
    if (seen.has(file)) return new Map()
    seen.add(file)
    const sf = this.source(file)
    const local = this.declarationsOf(file)
    const out = new Map<string, Exported>()
    const imports = new Map<string, { file: string; name: string }>()
    for (const node of sf.statements) {
      if (!ts.isImportDeclaration(node) || !ts.isStringLiteral(node.moduleSpecifier)) continue
      const target = this.resolveSpecifier(file, node.moduleSpecifier.text)
      const named = node.importClause?.namedBindings
      if (target === undefined || named === undefined || !ts.isNamedImports(named)) continue
      for (const el of named.elements)
        imports.set(el.name.text, { file: target, name: (el.propertyName ?? el.name).text })
    }
    for (const node of sf.statements) {
      if (ts.isExportDeclaration(node)) {
        if (node.isTypeOnly && node.moduleSpecifier === undefined) continue
        const spec =
          node.moduleSpecifier !== undefined && ts.isStringLiteral(node.moduleSpecifier)
            ? node.moduleSpecifier.text
            : undefined
        const target = spec === undefined ? undefined : this.resolveSpecifier(file, spec)
        if (spec !== undefined && target === undefined) continue
        const clause = node.exportClause
        if (clause === undefined && target !== undefined) {
          for (const [name, e] of this.exportsOf(target, seen))
            if (name !== 'default' && !out.has(name)) out.set(name, e)
        } else if (clause !== undefined && ts.isNamespaceExport(clause) && target !== undefined) {
          out.set(clause.name.text, { namespace: target })
        } else if (clause !== undefined && ts.isNamedExports(clause)) {
          for (const el of clause.elements) {
            const from = (el.propertyName ?? el.name).text
            let found: Exported | undefined
            if (target !== undefined) found = this.exportsOf(target, seen).get(from)
            else {
              const d = local.get(from)
              const imp = imports.get(from)
              if (d !== undefined) found = { decl: d }
              else if (imp !== undefined) found = this.exportsOf(imp.file, seen).get(imp.name)
            }
            if (found !== undefined) out.set(el.name.text, found)
          }
        }
      } else if (
        (ts.isFunctionDeclaration(node) ||
          ts.isClassDeclaration(node) ||
          ts.isInterfaceDeclaration(node) ||
          ts.isTypeAliasDeclaration(node) ||
          ts.isEnumDeclaration(node)) &&
        node.name !== undefined &&
        isExported(node)
      ) {
        const d = local.get(node.name.text)
        if (d !== undefined) out.set(d.name, { decl: d })
      } else if (ts.isVariableStatement(node) && isExported(node)) {
        for (const v of node.declarationList.declarations) {
          if (!ts.isIdentifier(v.name)) continue
          const d = local.get(v.name.text)
          if (d !== undefined) out.set(d.name, { decl: d })
        }
      }
    }
    this.exportMaps.set(file, out)
    return out
  }
}

function paramsOf(sig: ts.SignatureDeclaration | undefined, doc: ParsedDoc): ParamDoc[] {
  if (sig === undefined) return []
  const docs = new Map(doc.params.map((p) => [p.name, p.description]))
  const out: ParamDoc[] = []
  const unnamed = doc.params.map((p) => p.name).filter((n) => !n.includes('.'))
  sig.parameters.forEach((p, i) => {
    const type = p.type === undefined ? '' : oneLine(p.type.getText())
    const defaultValue = p.initializer === undefined ? '' : oneLine(p.initializer.getText())
    const optional = p.questionToken !== undefined || p.initializer !== undefined || p.dotDotDotToken !== undefined
    const name = ts.isIdentifier(p.name)
      ? p.name.text
      : (unnamed[i] ?? (ts.isObjectBindingPattern(p.name) ? 'options' : `arg${i}`))
    const rest = p.dotDotDotToken === undefined ? '' : '...'
    out.push({ name: `${rest}${name}`, type, optional, defaultValue, description: docs.get(name) ?? '' })
    if (ts.isObjectBindingPattern(p.name)) {
      for (const el of p.name.elements) {
        if (!ts.isIdentifier(el.name)) continue
        const field = (el.propertyName ?? el.name).getText()
        out.push({
          name: `${name}.${field}`,
          type: '',
          optional: true,
          defaultValue: el.initializer === undefined ? '' : oneLine(el.initializer.getText()),
          description: docs.get(`${name}.${field}`) ?? '',
        })
      }
    }
  })
  // `@param options.x` entries for fields of an options type written out of the signature.
  for (const p of doc.params) {
    if (p.name.includes('.') && !out.some((o) => o.name === p.name))
      out.push({ name: p.name, type: '', optional: true, defaultValue: '', description: p.description })
  }
  return out
}

function headOf(sf: ts.SourceFile, node: ts.Node, bodyStart: number | undefined): string {
  const text = sf.text.slice(node.getStart(sf), bodyStart ?? node.getEnd())
  return text
    .replace(/^export\s+(default\s+)?/, '')
    .replace(/^declare\s+/, '')
    .trim()
    .replace(/\s*\{$/, '')
}

function membersOf(project: Project, sf: ts.SourceFile, node: ts.ClassDeclaration): MemberDoc[] {
  const out: MemberDoc[] = []
  for (const m of node.members) {
    if (hasModifier(m, ts.SyntaxKind.PrivateKeyword) || hasModifier(m, ts.SyntaxKind.ProtectedKeyword)) continue
    if (m.name !== undefined && ts.isPrivateIdentifier(m.name)) continue
    const { doc, examples } = parsedDoc(project, sf, m)
    if (doc.modifiers.includes('internal')) continue
    const isStatic = hasModifier(m, ts.SyntaxKind.StaticKeyword)
    const body = (m as { body?: ts.Node }).body
    const end =
      ts.isPropertyDeclaration(m) && m.initializer !== undefined ? m.initializer.getFullStart() : body?.getStart(sf)
    const signature = oneLine(headOf(sf, m, end).replace(/\s*=$/, ''))
    let kind: MemberDoc['kind']
    let name: string
    let sig: ts.SignatureDeclaration | undefined
    if (ts.isConstructorDeclaration(m)) {
      kind = 'constructor'
      name = 'constructor'
      sig = m
    } else if (ts.isMethodDeclaration(m)) {
      kind = 'method'
      name = m.name.getText(sf)
      sig = m
    } else if (ts.isGetAccessorDeclaration(m) || ts.isSetAccessorDeclaration(m)) {
      kind = 'accessor'
      name = m.name.getText(sf)
      if (out.some((o) => o.name === name && o.kind === 'accessor')) continue
    } else if (ts.isPropertyDeclaration(m)) {
      kind = 'property'
      name = m.name.getText(sf)
    } else continue
    // Overloads: keep the first signature that has a doc comment, else the first.
    const twin = out.findIndex((o) => o.name === name && o.kind === kind && o.static === isStatic)
    if (twin >= 0) {
      const prev = out[twin] as MemberDoc
      if (body === undefined || prev.summary !== '') {
        if (body === undefined) out[twin] = { ...prev, signature: `${prev.signature}\n${signature}` }
        continue
      }
    }
    out.push({
      name,
      kind,
      static: isStatic,
      signature,
      summary: doc.summary,
      description: doc.description,
      params: paramsOf(sig, doc),
      returns: doc.returns,
      examples,
    })
  }
  return out
}

function signatureOf(d: Declaration): { signature: string; sig?: ts.SignatureDeclaration } {
  const sf = d.sf
  const first = d.nodes[0] as ts.Node
  switch (d.kind) {
    case 'function': {
      if (ts.isVariableStatement(first)) {
        const v = first.declarationList.declarations.find((x) => ts.isIdentifier(x.name) && x.name.text === d.name)
        const init = v?.initializer
        if (v !== undefined && init !== undefined && (ts.isArrowFunction(init) || ts.isFunctionExpression(init))) {
          const head = sf.text.slice(init.getStart(sf), init.body.getStart(sf)).replace(/\s*=>\s*$/, '')
          const type = v.type === undefined ? '' : `: ${v.type.getText(sf)}`
          return { signature: oneLine(`const ${d.name}${type} = ${head}`), sig: init }
        }
      }
      const fns = d.nodes.filter(ts.isFunctionDeclaration)
      const overloads = fns.filter((f) => f.body === undefined)
      const shown = overloads.length > 0 ? overloads : fns
      return {
        signature: shown.map((f) => oneLine(headOf(sf, f, f.body?.getStart(sf)))).join('\n'),
        ...(shown[0] === undefined ? {} : { sig: shown[0] }),
      }
    }
    case 'const': {
      const s = first as ts.VariableStatement
      const v = s.declarationList.declarations.find((x) => ts.isIdentifier(x.name) && x.name.text === d.name)
      const keyword = (s.declarationList.flags & ts.NodeFlags.Const) !== 0 ? 'const' : 'let'
      const type = v?.type === undefined ? '' : `: ${oneLine(v.type.getText(sf))}`
      return { signature: `${keyword} ${d.name}${type}` }
    }
    case 'class': {
      const c = first as ts.ClassDeclaration
      const head = oneLine(sf.text.slice(c.getStart(sf), c.members.pos).replace(/\{\s*$/, ''))
      return { signature: head.replace(/^export\s+(default\s+)?/, '').replace(/^declare\s+/, '') }
    }
    default:
      return {
        signature: first
          .getText(sf)
          .replace(/^export\s+/, '')
          .replace(/^declare\s+/, ''),
      }
  }
}

function documentedNode(d: Declaration): ts.Node {
  for (const n of d.nodes) if (docOf(d.sf, n) !== undefined) return n
  return d.nodes[0] as ts.Node
}

/**
 * Read every entry point's exports into modules and symbols, each symbol at its canonical home. Declarations tagged
 * `@internal` (or `@hidden`) are left out.
 *
 * @param project - The packages and entry points, from `readProject`.
 * @returns The modules and symbols.
 */
export function readSymbols(project: Project): SymbolsResult {
  const reader = new Reader()
  const byDir = [...project.entries].sort((a, b) => b.dir.length - a.dir.length)
  const ownerOf = (file: string): EntryPoint | undefined =>
    byDir.find((e) => e.files.includes(file)) ?? byDir.find((e) => file.startsWith(`${e.dir}/`))
  const symbols = new Map<string, SymbolDoc & { aliases: string[] }>()
  const declOrder = new Map<string, number>()
  const moduleSymbols = new Map<string, string[]>()
  const reexports = new Map<string, Record<string, string>>()
  const pending: { entry: EntryPoint; name: string; decl: Declaration }[] = []

  for (const entry of project.entries) {
    let exports: Map<string, Exported>
    try {
      exports = reader.exportsOf(entry.index)
    } catch {
      continue
    }
    for (const [name, e] of exports) if ('decl' in e) pending.push({ entry, name, decl: e.decl })
  }

  const build = (decl: Declaration, home: EntryPoint): (SymbolDoc & { aliases: string[] }) | undefined => {
    const node = documentedNode(decl)
    const { doc, examples } = parsedDoc(project, decl.sf, node)
    if (doc.modifiers.includes('internal') || doc.modifiers.includes('hidden')) return undefined
    const { signature, sig } = signatureOf(decl)
    const first = decl.nodes[0] as ts.Node
    const line = decl.sf.getLineAndCharacterOfPosition(first.getStart(decl.sf)).line + 1
    return {
      id: `${home.specifier}#${decl.name}`,
      package: home.package,
      module: home.specifier,
      name: decl.name,
      kind: decl.kind,
      signature,
      summary: doc.summary,
      description: doc.description,
      params: paramsOf(sig, doc),
      returns: doc.returns,
      throws: doc.throws,
      examples,
      concepts: doc.concepts,
      see: doc.see,
      references: doc.references,
      ...(doc.deprecated === undefined ? {} : { deprecated: doc.deprecated }),
      aliases: [],
      members: ts.isClassDeclaration(first) ? membersOf(project, decl.sf, first) : [],
      source: { file: relativeTo(project, decl.file), line },
    }
  }

  // First the canonical homes, then the aliases (so an alias always finds its symbol).
  const homes = new Map<string, EntryPoint>()
  for (const { decl } of pending) {
    if (homes.has(decl.key)) continue
    const owner = ownerOf(decl.file)
    if (owner !== undefined) homes.set(decl.key, owner)
  }
  for (const [key, home] of homes) {
    const decl = pending.find((p) => p.decl.key === key)?.decl
    if (decl === undefined) continue
    const s = build(decl, home)
    if (s === undefined) continue
    symbols.set(key, s)
    const pos = (decl.nodes[0] as ts.Node).getStart(decl.sf)
    declOrder.set(key, home.files.indexOf(decl.file) * 1e9 + pos)
  }
  for (const { entry, name, decl } of pending) {
    const s = symbols.get(decl.key)
    if (s === undefined) continue
    const id = `${entry.specifier}#${name}`
    if (id === s.id) continue
    s.aliases.push(id)
    const r = reexports.get(entry.specifier) ?? {}
    r[name] = s.id
    reexports.set(entry.specifier, r)
  }
  const ordered = [...symbols.entries()].sort((a, b) => {
    const ma = a[1].module.localeCompare(b[1].module)
    return ma !== 0 ? ma : (declOrder.get(a[0]) ?? 0) - (declOrder.get(b[0]) ?? 0)
  })
  for (const [, s] of ordered) {
    const list = moduleSymbols.get(s.module) ?? []
    list.push(s.name)
    moduleSymbols.set(s.module, list)
  }

  const modules: ModuleDoc[] = project.entries.map((e) => {
    const header = /^\s*(\/\*\*[\s\S]*?\*\/)/.exec(reader.source(e.index).text)?.[1]
    const summary = e.manifest?.summary ?? (header === undefined ? '' : parseDocComment(unframeComment(header)).summary)
    return {
      id: e.specifier,
      package: e.package,
      path: e.path,
      name: e.path === '' ? e.package : (e.path.split('/').at(-1) as string),
      layer: e.layer,
      summary,
      ...(e.manifest?.tier === undefined ? {} : { tier: e.manifest.tier }),
      ...(e.parent === undefined ? {} : { parent: e.parent }),
      children: project.entries.filter((c) => c.parent === e.specifier).map((c) => c.specifier),
      symbols: moduleSymbols.get(e.specifier) ?? [],
      reexports: reexports.get(e.specifier) ?? {},
      files: e.files.map((f) => relativeTo(project, f)),
    }
  })
  return { modules, symbols: ordered.map(([, s]) => s) }
}
