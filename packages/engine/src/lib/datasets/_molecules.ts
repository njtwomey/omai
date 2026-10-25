// A small SMILES reader (organic subset, aromatic atoms, branches, ring closures, bond symbols; no bracket atoms)
// producing heavy-atom graphs with implicit hydrogen counts, and the molecules of `smallMolecules`.
import { MdmlError } from '../base/index.ts'

/** A parsed molecule: heavy atoms, bonds with orders (1.5 for aromatic), and implicit hydrogens per atom. */
export interface ParsedMolecule {
  readonly atoms: readonly string[]
  readonly aromatic: readonly boolean[]
  readonly bonds: readonly (readonly [number, number, number])[]
  readonly hydrogens: readonly number[]
}

const VALENCES: Readonly<Record<string, readonly number[]>> = {
  B: [3],
  C: [4],
  N: [3, 5],
  O: [2],
  P: [3, 5],
  S: [2, 4, 6],
  F: [1],
  Cl: [1],
  Br: [1],
  I: [1],
}

/** Standard atomic weights (IUPAC, abridged). */
export const ATOMIC_WEIGHTS: Readonly<Record<string, number>> = {
  H: 1.008,
  B: 10.81,
  C: 12.011,
  N: 14.007,
  O: 15.999,
  F: 18.998,
  P: 30.974,
  S: 32.06,
  Cl: 35.45,
  Br: 79.904,
  I: 126.904,
}

/** Parse a SMILES string of the organic subset into a heavy-atom graph. */
export function parseSmiles(smiles: string): ParsedMolecule {
  const atoms: string[] = []
  const aromatic: boolean[] = []
  const bonds: [number, number, number][] = []
  const stack: number[] = []
  const rings = new Map<number, { atom: number; order: number | undefined }>()
  let prev = -1
  let pending: number | undefined
  const orderOf = (a: number, b: number, explicit: number | undefined) =>
    explicit ?? (aromatic[a] === true && aromatic[b] === true ? 1.5 : 1)
  let i = 0
  while (i < smiles.length) {
    const ch = smiles[i] as string
    const two = smiles.slice(i, i + 2)
    if (ch === '(') {
      stack.push(prev)
      i++
    } else if (ch === ')') {
      prev = stack.pop() ?? -1
      i++
    } else if (ch === '-' || ch === '=' || ch === '#' || ch === ':') {
      pending = ch === '-' ? 1 : ch === '=' ? 2 : ch === '#' ? 3 : 1.5
      i++
    } else if (/[0-9%]/.test(ch)) {
      const digits = ch === '%' ? smiles.slice(i + 1, i + 3) : ch
      const label = Number(digits)
      i += ch === '%' ? 3 : 1
      const open = rings.get(label)
      if (open === undefined) rings.set(label, { atom: prev, order: pending })
      else {
        bonds.push([open.atom, prev, orderOf(open.atom, prev, pending ?? open.order)])
        rings.delete(label)
      }
      pending = undefined
    } else if (two === 'Cl' || two === 'Br' || /[BCNOPSFI]/.test(ch) || /[cnosp]/.test(ch)) {
      const symbol = two === 'Cl' || two === 'Br' ? two : ch.toUpperCase()
      const isAromatic = /[cnosp]/.test(ch) && two !== 'Cl'
      atoms.push(symbol)
      aromatic.push(isAromatic)
      const idx = atoms.length - 1
      if (prev >= 0) bonds.push([prev, idx, orderOf(prev, idx, pending)])
      pending = undefined
      prev = idx
      i += symbol.length
    } else {
      throw new MdmlError('value', `parseSmiles: unsupported character ${ch} in ${smiles}`)
    }
  }
  if (rings.size > 0) throw new MdmlError('value', `parseSmiles: unclosed ring in ${smiles}`)
  const used = atoms.map(() => 0)
  for (const [a, b, o] of bonds) {
    used[a] = (used[a] ?? 0) + o
    used[b] = (used[b] ?? 0) + o
  }
  const hydrogens = atoms.map((s, j) => {
    const u = used[j] ?? 0
    const vals = VALENCES[s] ?? [0]
    if (aromatic[j] === true) return Math.max(0, Math.floor((vals[0] ?? 0) - u))
    const v = vals.find((x) => x >= u - 1e-9) ?? u
    return Math.max(0, Math.round(v - u))
  })
  return { atoms, aromatic, bonds, hydrogens }
}

/** The Hill-order molecular formula (C, then H, then the rest alphabetically). */
export function formulaOf(m: ParsedMolecule): string {
  const counts = new Map<string, number>()
  for (const a of m.atoms) counts.set(a, (counts.get(a) ?? 0) + 1)
  const h = m.hydrogens.reduce((s, v) => s + v, 0)
  if (h > 0) counts.set('H', h)
  const keys = [...counts.keys()].sort((a, b) => {
    const rank = (k: string) => (k === 'C' ? 0 : k === 'H' ? 1 : 2)
    return rank(a) - rank(b) || a.localeCompare(b)
  })
  return keys.map((k) => `${k}${(counts.get(k) ?? 0) > 1 ? counts.get(k) : ''}`).join('')
}

/** Common small molecules with their SMILES. */
export const MOLECULES: readonly { readonly name: string; readonly smiles: string }[] = [
  { name: 'water', smiles: 'O' },
  { name: 'ammonia', smiles: 'N' },
  { name: 'methane', smiles: 'C' },
  { name: 'ethane', smiles: 'CC' },
  { name: 'propane', smiles: 'CCC' },
  { name: 'butane', smiles: 'CCCC' },
  { name: 'isobutane', smiles: 'CC(C)C' },
  { name: 'ethylene', smiles: 'C=C' },
  { name: 'acetylene', smiles: 'C#C' },
  { name: 'methanol', smiles: 'CO' },
  { name: 'ethanol', smiles: 'CCO' },
  { name: 'dimethyl ether', smiles: 'COC' },
  { name: 'formaldehyde', smiles: 'C=O' },
  { name: 'carbon dioxide', smiles: 'O=C=O' },
  { name: 'acetone', smiles: 'CC(=O)C' },
  { name: 'acetic acid', smiles: 'CC(=O)O' },
  { name: 'urea', smiles: 'NC(=O)N' },
  { name: 'glycine', smiles: 'NCC(=O)O' },
  { name: 'alanine', smiles: 'CC(N)C(=O)O' },
  { name: 'chloroform', smiles: 'ClC(Cl)Cl' },
  { name: 'cyclohexane', smiles: 'C1CCCCC1' },
  { name: 'glucose (open chain)', smiles: 'OCC(O)C(O)C(O)C(O)C=O' },
  { name: 'benzene', smiles: 'c1ccccc1' },
  { name: 'toluene', smiles: 'Cc1ccccc1' },
  { name: 'phenol', smiles: 'Oc1ccccc1' },
  { name: 'aniline', smiles: 'Nc1ccccc1' },
  { name: 'benzoic acid', smiles: 'OC(=O)c1ccccc1' },
  { name: 'pyridine', smiles: 'c1ccncc1' },
  { name: 'furan', smiles: 'c1ccoc1' },
  { name: 'thiophene', smiles: 'c1ccsc1' },
  { name: 'naphthalene', smiles: 'c1ccc2ccccc2c1' },
  { name: 'aspirin', smiles: 'CC(=O)Oc1ccccc1C(=O)O' },
  { name: 'paracetamol', smiles: 'CC(=O)Nc1ccc(O)cc1' },
  { name: 'ibuprofen', smiles: 'CC(C)Cc1ccc(cc1)C(C)C(=O)O' },
  { name: 'caffeine', smiles: 'Cn1cnc2c1c(=O)n(C)c(=O)n2C' },
  { name: 'nicotine', smiles: 'CN1CCCC1c1cccnc1' },
]
