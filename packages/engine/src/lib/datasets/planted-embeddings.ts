/**
 * Generators with a planted truth for the embedding and matching methods: a corpus whose word vectors must contain
 * parallelograms (planted analogies) and clusters, a knowledge graph whose relations have planted logical patterns
 * (symmetry, inversion, composition, a directed cycle) so that each scorer's expressiveness shows, and two tables of
 * records with known matches under typos for record linkage.
 */

import { type CorpusData, corpus, type DatasetGenerator, type TripleData, triples } from '../data/index.ts'
import { categorical, foldIn, uniform } from '../random/index.ts'
import { dict, int, real } from '../space/index.ts'
import { fromStorage } from '../tensor/index.ts'
import { stamp } from './_util.ts'

/**
 * A corpus with planted analogies and clusters. Target words `x{i}y{j}` combine a concept `i` and an attribute `j`;
 * each sentence is one target with context words drawn from the concept's own context words, the attribute's own, or
 * a shared filler pool, so a target's co-occurrence profile is the sum of a concept part and an attribute part and
 * the vector offset from `x{i}y{j}` to `x{i}y{j'}` is the same for every concept (the parallelogram). The truth has
 * the analogy quadruples `(x{i}y{j}, x{i}y{j'}, x{i'}y{j}, x{i'}y{j'})`, the concept and attribute of every target,
 * and the target list.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { analogyCorpus } from 'mdml-engine/datasets'
 * generate(analogyCorpus, { knobs: { concepts: 3, attributes: 2, sentences: 10 } }).data.documents?.length // => 10
 * ```
 */
export const analogyCorpus: DatasetGenerator<CorpusData> = {
  name: 'analogyCorpus',
  kind: 'corpus',
  description: 'Sentences whose targets combine a concept and an attribute, with the analogies and clusters planted.',
  knobs: dict({
    concepts: int(2, 50, { label: 'concepts', description: 'The number of concepts.', default: 5 }),
    attributes: int(2, 50, { label: 'attributes', description: 'The number of attributes.', default: 4 }),
    contexts: int(1, 50, {
      label: 'context words per concept and attribute',
      description: 'Context words owned by each concept and each attribute.',
      default: 4,
    }),
    fillers: int(0, 1000, { label: 'filler words', description: 'Words that carry no signal.', default: 6 }),
    sentences: int(1, 1_000_000, { label: 'sentences', description: 'The number of sentences.', default: 4000 }),
    length: int(2, 40, {
      label: 'context words per sentence',
      description: 'Context tokens around the target.',
      default: 6,
    }),
    noise: real(0, 1, {
      label: 'filler share',
      description: 'The chance a context token is a filler instead of a concept or attribute word.',
      default: 0.2,
    }),
  }),
  generate: (k, knobs) => {
    const C = knobs['concepts'] as number
    const A = knobs['attributes'] as number
    const M = knobs['contexts'] as number
    const F = knobs['fillers'] as number
    const S = knobs['sentences'] as number
    const L = knobs['length'] as number
    const noise = knobs['noise'] as number
    const targets = Array.from({ length: C * A }, (_, t) => `x${Math.floor(t / A)}y${t % A}`)
    const conceptWords = Array.from({ length: C }, (_, i) => Array.from({ length: M }, (_, m) => `c${i}_${m}`))
    const attributeWords = Array.from({ length: A }, (_, j) => Array.from({ length: M }, (_, m) => `a${j}_${m}`))
    const fillers = Array.from({ length: F }, (_, f) => `f${f}`)
    const vocabulary = [...targets, ...conceptWords.flat(), ...attributeWords.flat(), ...fillers].sort()
    const index = new Map(vocabulary.map((w, i) => [w, i]))
    const pick = Array.from(categorical(foldIn(k, 0), new Array<number>(C * A).fill(1), [S]).data)
    const documents: string[] = []
    const tokens: number[][] = []
    for (let s = 0; s < S; s++) {
      const t = pick[s] ?? 0
      const i = Math.floor(t / A)
      const j = t % A
      const u = uniform(foldIn(foldIn(k, 1), s), [3 * L]).data
      const sentence: string[] = []
      for (let p = 0; p < L; p++) {
        const kind = u[3 * p] ?? 0
        const m = Math.floor((u[3 * p + 1] ?? 0) * M)
        let word: string
        if (F > 0 && kind < noise) word = fillers[Math.floor((u[3 * p + 2] ?? 0) * F)] ?? 'f0'
        else if ((u[3 * p + 2] ?? 0) < 0.5) word = conceptWords[i]?.[m] ?? ''
        else word = attributeWords[j]?.[m] ?? ''
        sentence.push(word)
      }
      sentence.splice(Math.floor(L / 2), 0, targets[t] ?? '')
      documents.push(sentence.join(' '))
      tokens.push(sentence.map((w) => index.get(w) ?? 0))
    }
    const analogies: [string, string, string, string][] = []
    for (let i = 0; i < C; i++)
      for (let i2 = 0; i2 < C; i2++)
        for (let j = 0; j < A; j++)
          for (let j2 = 0; j2 < A; j2++)
            if (i !== i2 && j !== j2) analogies.push([`x${i}y${j}`, `x${i}y${j2}`, `x${i2}y${j}`, `x${i2}y${j2}`])
    return {
      data: corpus({ documents, tokens, vocabulary }, stamp('analogyCorpus', analogyCorpus.description, knobs)),
      truth: {
        clusters: fromStorage(
          Int32Array.from(vocabulary, (w) => {
            const t = targets.indexOf(w)
            return t >= 0 ? Math.floor(t / A) : -1
          }),
          [vocabulary.length],
          'int32',
        ),
        extra: {
          targets,
          analogies,
          conceptOf: Object.fromEntries(targets.map((w, t) => [w, Math.floor(t / A)])),
          attributeOf: Object.fromEntries(targets.map((w, t) => [w, t % A])),
        },
      },
    }
  },
}

/** The logical pattern a planted relation obeys. */
export type RelationPattern = 'symmetric' | 'inverse' | 'cycle' | 'composition'

/**
 * A knowledge graph with planted relation patterns. Entities are grouped; `sibling` is symmetric (some pairs of a
 * group), `parent` and `child` are inverses (older entities of a group parent younger ones), `next` is a directed
 * cycle through the group (antisymmetric, a rotation), and `follows` is the composition `parent` then `next`. A share
 * of the facts whose pattern lets them be inferred from the others (the reverse of a symmetric or inverse fact, a
 * composed fact) is held out as a test set, so a scorer that cannot express the pattern cannot rank them. The truth
 * has every fact, the held-out ones, each relation's pattern and each entity's group.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { plantedRelations } from 'mdml-engine/datasets'
 * generate(plantedRelations, { knobs: { groups: 2, perGroup: 6 } }).data.relationNames // => ['sibling', 'parent', 'child', 'next', 'follows']
 * ```
 */
export const plantedRelations: DatasetGenerator<TripleData> = {
  name: 'plantedRelations',
  kind: 'triples',
  description: 'Triples whose relations are symmetric, inverse, a directed cycle and a composition, partly held out.',
  knobs: dict({
    groups: int(1, 100, { label: 'groups', description: 'The number of entity groups.', default: 6 }),
    perGroup: int(4, 100, { label: 'entities per group', description: 'Entities in each group (even).', default: 10 }),
    density: real(0.05, 1, {
      label: 'sibling density',
      description: 'The chance a pair of a group are siblings.',
      default: 0.5,
    }),
    heldOut: real(0, 0.9, {
      label: 'held-out share',
      description: 'The share of inferable facts kept for testing.',
      default: 0.3,
    }),
  }),
  generate: (k, knobs) => {
    const G = knobs['groups'] as number
    const P = (knobs['perGroup'] as number) - ((knobs['perGroup'] as number) % 2)
    const density = knobs['density'] as number
    const heldOut = knobs['heldOut'] as number
    const relationNames = ['sibling', 'parent', 'child', 'next', 'follows']
    const pattern: Record<string, RelationPattern> = {
      sibling: 'symmetric',
      parent: 'inverse',
      child: 'inverse',
      next: 'cycle',
      follows: 'composition',
    }
    const all = new Map<string, [number, number, number]>()
    const add = (h: number, r: number, t: number): void => {
      all.set(`${h},${r},${t}`, [h, r, t])
    }
    const u = uniform(foldIn(k, 0), [G * P * P + 4 * G * P]).data
    let at = 0
    const parentOf = new Map<number, number[]>()
    for (let g = 0; g < G; g++) {
      const base = g * P
      for (let i = 0; i < P; i++)
        for (let j = i + 1; j < P; j++)
          if ((u[at++] ?? 1) < density) {
            add(base + i, 0, base + j)
            add(base + j, 0, base + i)
          }
      for (let i = 0; i < P / 2; i++) {
        const children = new Set<number>([P / 2 + Math.floor((u[at++] ?? 0) * (P / 2)), P / 2 + ((i + 1) % (P / 2))])
        for (const c of children) {
          add(base + i, 1, base + c)
          add(base + c, 2, base + i)
          parentOf.set(base + i, [...(parentOf.get(base + i) ?? []), base + c])
        }
      }
      for (let i = 0; i < P; i++) add(base + i, 3, base + ((i + 1) % P))
    }
    for (const [h, kids] of parentOf) for (const m of kids) add(h, 4, Math.floor(m / P) * P + (((m % P) + 1) % P))
    const facts = [...all.values()].sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2])
    const key = (h: number, r: number, t: number): string => `${h},${r},${t}`
    const draws = uniform(foldIn(k, 1), [facts.length]).data
    const test: [number, number, number][] = []
    const train: [number, number, number][] = []
    const taken = new Set<string>()
    for (const [i, f] of facts.entries()) {
      const [h, r, t] = f
      // A fact may be held out only if its pattern lets the rest imply it, and its partner stays in training.
      let inferable = false
      if (r === 0) inferable = all.has(key(t, 0, h)) && !taken.has(key(t, 0, h)) && h < t
      else if (r === 1) inferable = true
      else if (r === 4) inferable = true
      if (inferable && (draws[i] ?? 1) < heldOut) {
        test.push(f)
        taken.add(key(h, r, t))
      }
    }
    // Keep the partner of a held-out reverse in training: parent facts are held out, child facts stay.
    for (const f of facts) if (!taken.has(key(...f))) train.push(f)
    const entityNames = Array.from({ length: G * P }, (_, e) => `g${Math.floor(e / P)}e${e % P}`)
    return {
      data: triples(
        train.map((f) => f[0]),
        train.map((f) => f[1]),
        train.map((f) => f[2]),
        { entityNames, relationNames, meta: stamp('plantedRelations', plantedRelations.description, knobs) },
      ),
      truth: {
        extra: {
          allTriples: facts,
          heldOut: test,
          patterns: pattern,
          group: Array.from({ length: G * P }, (_, e) => Math.floor(e / P)),
        },
      },
    }
  },
}

/** A record of the linkage tables: its fields as text. */
interface LinkageRecord {
  readonly name: string
  readonly city: string
  readonly year: string
}

/**
 * Two tables of person records with known matches. Each of `entities` people has a name, a city and a birth year;
 * the left table holds one clean record per person, the right table a noisy copy of a share of them (typos in the
 * name, a city spelled differently, a year off by one) plus records of people who are not in the left table. The
 * dataset is a corpus of the left records followed by the right ones, each as `name|city|year`, with the person as the
 * label (records of one person share it); the truth has the matching `[left, right]` index pairs (right indices count
 * from the first right record) and the sizes of the two tables.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { linkageTables } from 'mdml-engine/datasets'
 * generate(linkageTables, { knobs: { entities: 20, overlap: 1, extras: 0 } }).data.documents?.length // => 40
 * ```
 */
export const linkageTables: DatasetGenerator<CorpusData> = {
  name: 'linkageTables',
  kind: 'corpus',
  description: 'Two tables of person records, the second with typos, and the known matching pairs.',
  knobs: dict({
    entities: int(2, 100_000, { label: 'people', description: 'Records in the left table.', default: 200 }),
    overlap: real(0, 1, {
      label: 'overlap',
      description: 'The share of people with a record in the right table.',
      default: 0.6,
    }),
    extras: int(0, 100_000, {
      label: 'unmatched right records',
      description: 'Right records of other people.',
      default: 40,
    }),
    typoRate: real(0, 1, {
      label: 'typo rate',
      description: 'The chance per name character of an edit.',
      default: 0.06,
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['entities'] as number
    const overlap = knobs['overlap'] as number
    const extras = knobs['extras'] as number
    const typo = knobs['typoRate'] as number
    const first = ['anna', 'bela', 'carl', 'dora', 'emil', 'fran', 'gita', 'hugo', 'iris', 'jan', 'kira', 'liam']
    const last = [
      'smith',
      'jones',
      'novak',
      'khan',
      'rossi',
      'meyer',
      'ito',
      'silva',
      'berg',
      'costa',
      'young',
      'lopez',
    ]
    const cities = ['london', 'paris', 'berlin', 'madrid', 'rome', 'vienna', 'prague', 'dublin']
    const person = (i: number, salt: number): LinkageRecord => {
      const u = uniform(foldIn(foldIn(k, 10 + salt), i), [4]).data
      return {
        name: `${first[Math.floor((u[0] ?? 0) * first.length)]} ${last[Math.floor((u[1] ?? 0) * last.length)]}`,
        city: cities[Math.floor((u[2] ?? 0) * cities.length)] ?? '',
        year: String(1940 + Math.floor((u[3] ?? 0) * 60)),
      }
    }
    const left = Array.from({ length: n }, (_, i) => person(i, 0))
    const letters = 'abcdefghijklmnopqrstuvwxyz'
    const noisy = (r: LinkageRecord, i: number): LinkageRecord => {
      const u = uniform(foldIn(foldIn(k, 20), i), [r.name.length * 3 + 3]).data
      let name = ''
      for (let c = 0; c < r.name.length; c++) {
        const ch = r.name[c] ?? ''
        if (ch === ' ' || (u[3 * c] ?? 1) >= typo) name += ch
        else {
          const op = Math.floor((u[3 * c + 1] ?? 0) * 3)
          const x = letters[Math.floor((u[3 * c + 2] ?? 0) * 26)] ?? 'a'
          if (op === 1) name += ch + x
          else if (op === 2) name += x
        }
      }
      const v = r.name.length * 3
      const city = (u[v] ?? 1) < 0.1 ? r.city.slice(0, -1) : r.city
      const year = (u[v + 1] ?? 1) < 0.1 ? String(Number(r.year) + ((u[v + 2] ?? 0) < 0.5 ? -1 : 1)) : r.year
      return { name: name.trim() === '' ? r.name : name, city, year }
    }
    const kept = Array.from(uniform(foldIn(k, 1), [n]).data, (x) => x < overlap)
    const right: LinkageRecord[] = []
    const who: number[] = []
    const matches: [number, number][] = []
    for (let i = 0; i < n; i++) {
      if (kept[i]) {
        matches.push([i, right.length])
        who.push(i)
        right.push(noisy(left[i] as LinkageRecord, i))
      }
    }
    for (let e = 0; e < extras; e++) {
      who.push(n + e)
      right.push(noisy(person(n + e, 1), n + e))
    }
    // Shuffle the right table so the order carries no signal.
    const order = Array.from(uniform(foldIn(k, 2), [right.length]).data)
      .map((x, i) => [x, i] as const)
      .sort((a, b) => a[0] - b[0])
      .map(([, i]) => i)
    const where = new Map(order.map((old, now) => [old, now]))
    const shuffled = order.map((i) => right[i] as LinkageRecord)
    const person0 = [...Array.from({ length: n }, (_, i) => i), ...order.map((i) => who[i] as number)]
    const line = (r: LinkageRecord): string => `${r.name}|${r.city}|${r.year}`
    return {
      data: corpus(
        { documents: [...left, ...shuffled].map(line), labels: person0 },
        stamp('linkageTables', linkageTables.description, knobs),
      ),
      truth: {
        clusters: fromStorage(Int32Array.from(person0), [person0.length], 'int32'),
        extra: { nLeft: n, nRight: right.length, matches: matches.map(([l, r]) => [l, where.get(r) ?? -1]) },
      },
    }
  },
}
