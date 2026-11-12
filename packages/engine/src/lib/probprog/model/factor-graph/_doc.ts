/**
 * The page that lists what the compiler maps, written from the mapping tables of `_table.ts` (`docs/probprog/compiler-mappings.md`): one
 * table per table of the compiler, one row per entry in priority order, with the families it serves, what it maps, the factor operators it
 * needs and why it sits where it does. A script (`make ep-compiler-doc`) writes it and a test checks it, so the list cannot go stale.
 */

import { VALUE_ALGEBRA_OPERATORS } from './_builder.ts'
import type { Mapping } from './_mapping.ts'
import {
  latentMappings,
  nodeMappings,
  observedElementMappings,
  observedNodeMappings,
  observedRowMappings,
} from './_table.ts'

const cell = (text: string): string => text.replaceAll('|', '\\|')

function tableOf(entries: readonly Mapping<never[], unknown>[]): string {
  const header = '| # | mapping | families | what it maps | operators | order |'
  const rule = '| --- | --- | --- | --- | --- | --- |'
  const rows = entries.map((e, i) => {
    const families = e.families === undefined ? '' : e.families.join(', ')
    const operators = e.requires.map((r) => `\`${r}\``).join(', ')
    return `| ${i + 1} | \`${e.name}\` | ${cell(families)} | ${cell(e.maps)} | ${operators} | ${cell(e.order ?? '')} |`
  })
  return [header, rule, ...rows].join('\n')
}

/** The text of `docs/probprog/compiler-mappings.md`. */
export function compilerMappingsDoc(): string {
  const sections: [string, string, readonly Mapping<never[], unknown>[]][] = [
    [
      'Nodes',
      'What the walk does with each kind of node of the spec, in order. The latents and the observed nodes go to the tables below.',
      nodeMappings as unknown as Mapping<never[], unknown>[],
    ],
    [
      'Latent nodes',
      'The first entry that takes a latent node maps it. The refusals sit where a later entry could otherwise take the node.',
      latentMappings as unknown as Mapping<never[], unknown>[],
    ],
    [
      'Observed nodes, as a whole',
      'The forms that take the whole node (a vector with a Wishart precision, a bivariate normal, a value moved to the mean); the last entry maps the rows.',
      observedNodeMappings as unknown as Mapping<never[], unknown>[],
    ],
    [
      'Observed nodes, by rows',
      'Once the values are read (missing values and a batch are refused first): a normal-gamma pair, one plated factor, a table over discrete parents, else element by element.',
      observedRowMappings as unknown as Mapping<never[], unknown>[],
    ],
    [
      'Observed nodes, element by element',
      'The likelihoods, by family (the order across families does not matter, the order within a family does); the last entry refuses a family with no mapping.',
      observedElementMappings as unknown as Mapping<never[], unknown>[],
    ],
  ]
  const out = [
    '# What the compiler maps',
    '',
    'Generated from the mapping tables of `packages/engine/src/lib/probprog/model/factor-graph/_table.ts` by `make ep-compiler-doc` (a test checks that it is',
    'current): edit the tables, not this page. A compilation runs in passes: the events and scans are unrolled, the constants are found (`_known.ts`),',
    'the layout plan is made (`_plan.ts`), the nodes are walked in spec order through the tables below, the dead `apply` factors are pruned, and the',
    'opaque nodes are reported. In a table the first entry whose pattern the node is maps it; an entry that can only tell by trying answers `NOT_MINE`,',
    'and the graph is then restored to what it was (`_mapping.ts`). The operators of an entry are the ones it emits itself; every mapping can also',
    `cause the operators of the value algebra (${VALUE_ALGEBRA_OPERATORS.map((o) => `\`${o}\``).join(', ')}).`,
  ]
  for (const [title, intro, entries] of sections) out.push('', `## ${title}`, '', intro, '', tableOf(entries))
  return `${out.join('\n')}\n`
}
