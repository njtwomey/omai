/**
 * The documentation catalogue: every public module and exported symbol of the documented packages, as plain data the
 * docs app renders and the example checker runs (docs/structure.md section 7).
 */
import type { Example, SourceLocation } from './examples.ts'

/** The kinds of exported symbol. */
export type SymbolKind = 'function' | 'class' | 'interface' | 'type' | 'const' | 'enum'

/** A parameter of a function, method or constructor: its type and default from the signature, its `@param` text. */
export interface ParamDoc {
  /** The name; an options object written as a pattern is `options`, its fields `options.<field>`. */
  readonly name: string
  readonly type: string
  readonly optional: boolean
  /** The default value as written, or `''`. */
  readonly defaultValue: string
  readonly description: string
}

/** A documented member of a class: a constructor, method, property or accessor. */
export interface MemberDoc {
  readonly name: string
  readonly kind: 'constructor' | 'method' | 'property' | 'accessor'
  readonly static: boolean
  readonly signature: string
  readonly summary: string
  readonly description: string
  readonly params: readonly ParamDoc[]
  readonly returns: string
  readonly examples: readonly Example[]
}

/** One exported symbol, at its canonical home (the entry point whose folder defines it). */
export interface SymbolDoc {
  /** `<module>#<name>`, such as `whitebox-engine/tensor#add`. */
  readonly id: string
  /** The package, such as `whitebox-engine`. */
  readonly package: string
  /** The canonical entry point, such as `whitebox-engine/tensor`. */
  readonly module: string
  readonly name: string
  readonly kind: SymbolKind
  /** The declaration without its body (every overload, one per line, for an overloaded function). */
  readonly signature: string
  /** The first paragraph of the doc comment. */
  readonly summary: string
  /** The rest of the prose, with `@remarks`. */
  readonly description: string
  readonly params: readonly ParamDoc[]
  readonly returns: string
  readonly throws: readonly string[]
  readonly examples: readonly Example[]
  /** `@concept` ids. */
  readonly concepts: readonly string[]
  /** `@see` entries. */
  readonly see: readonly string[]
  /** `@reference` citations. */
  readonly references: readonly string[]
  /** The `@deprecated` text, when the symbol is deprecated. */
  readonly deprecated?: string
  /** Other names it is exported under: `<module>#<name>` of each re-export elsewhere. */
  readonly aliases: readonly string[]
  readonly members: readonly MemberDoc[]
  readonly source: SourceLocation
}

/** Where a module sits: the engine core (`lib/`), the engine methods (`methods/`), or a part of another package. */
export type ModuleLayer = 'core' | 'methods' | 'part'

/** One public entry point. */
export interface ModuleDoc {
  /** The import specifier, such as `whitebox-engine/numerics/linalg`. */
  readonly id: string
  readonly package: string
  /** The path below the package, such as `numerics/linalg` (`''` for a package root). */
  readonly path: string
  /** The last path segment (or the package name for a root). */
  readonly name: string
  readonly layer: ModuleLayer
  /** One sentence: the `domain.json`/`area.json` summary, else the opening comment of `index.ts`. */
  readonly summary: string
  /** The tier from `domain.json`, for core domains. */
  readonly tier?: number
  /** The enclosing entry point, if any. */
  readonly parent?: string
  /** The entry points directly below it. */
  readonly children: readonly string[]
  /** The names of the symbols whose canonical home it is, in source order. */
  readonly symbols: readonly string[]
  /** Symbols it re-exports from elsewhere, as `<name>` to the canonical id. */
  readonly reexports: Readonly<Record<string, string>>
  /** The source files it owns (repository-relative). */
  readonly files: readonly string[]
}

/** A documented package. */
export interface PackageDoc {
  readonly name: string
  readonly description: string
  readonly version: string
  /** Its top-level entry points. */
  readonly modules: readonly string[]
}

/** A source file of a module, with the comment that opens it (when it is the file's own, not a declaration's). */
export interface FileDoc {
  /** Repository-relative path. */
  readonly path: string
  /** The entry point that owns it. */
  readonly module: string
  readonly summary: string
  readonly description: string
}

/** Everything the docs know: packages, modules, their files and symbols. */
export interface Catalogue {
  readonly version: 1
  readonly packages: readonly PackageDoc[]
  readonly modules: readonly ModuleDoc[]
  readonly files: readonly FileDoc[]
  readonly symbols: readonly SymbolDoc[]
}
