// The code that ran on the calibration page: the model, the fit of the chosen method and the run of the calibration, cut from
// compute.ts by its `// region` markers (the file is imported as text, so the code on screen is the code that runs).
import { snippetOf } from '../../../../../site/src/shell/snippet.ts'
import source from './compute.ts?raw'

/**
 * The code of the chosen method's calibration.
 *
 * @param id - The method's id (`nuts`, or a guide).
 * @returns The code.
 */
export function codeOf(id: string | undefined): string {
  const fit = id === 'nuts' ? 'sbc:nuts' : 'sbc:guide'
  return ['sbc', fit, 'sbc:run'].map((region) => snippetOf(source, { region })).join('\n')
}
