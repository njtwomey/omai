import { GROUPS } from './groups.ts'
import { type PocEntry, searchPocs, splitSubgroups } from './poc.ts'
import { POCS } from './registry.ts'

/**
 * The pocs of a group in the order its page shows them: the subgroups in the order they are declared, each newest first and
 * then by title. The previous and next arrows of the top bar walk this list, so they go through a page as it reads.
 *
 * @param group - The slug of a group.
 * @returns The group's pocs in display order (empty for an unknown group).
 */
export function inDisplayOrder(group: string | undefined): PocEntry[] {
  const g = GROUPS.find((x) => x.slug === group)
  const pool = POCS.filter((p) => p.group === group)
  return splitSubgroups(searchPocs(pool, ''), g).flatMap((s) => [...s.entries])
}
