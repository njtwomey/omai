/** Helpers shared by the subword models. */

const nameCache = new WeakMap<ReadonlyMap<string, number>, Map<number, string>>()

/** The token of each id of a vocabulary (cached). */
export function namesOf(model: { readonly vocab: ReadonlyMap<string, number> }): ReadonlyMap<number, string> {
  let names = nameCache.get(model.vocab)
  if (names === undefined) {
    names = new Map()
    for (const [t, i] of model.vocab) names.set(i, t)
    nameCache.set(model.vocab, names)
  }
  return names
}
