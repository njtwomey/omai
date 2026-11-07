// Internal: saving a file from the page (an anchor click, with an object URL freed afterwards) and waiting for a figure
// to draw, as the figure frame, the page export and the panels' exports share them.

/** How long a saved blob's object URL stays valid, in milliseconds: the browser has the bytes by the time it ends. */
const REVOKE_MS = 10_000

/**
 * Save what a URL holds as a file: a click on an anchor with a `download` name.
 *
 * @param url - A data URL or an object URL.
 * @param name - The file name offered to the reader.
 */
export function saveUrl(url: string, name: string): void {
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.click()
}

/**
 * Save bytes or text as a file through an object URL, which is revoked once the browser has the data.
 *
 * @param content - The contents.
 * @param type - The media type.
 * @param name - The file name offered to the reader.
 */
export function saveBlob(content: BlobPart, type: string, name: string): void {
  const url = URL.createObjectURL(new Blob([content], { type }))
  saveUrl(url, name)
  setTimeout(() => URL.revokeObjectURL(url), REVOKE_MS)
}

/**
 * Resolve after the page has drawn what was just changed: two animation frames, then a short settle (for charts whose
 * update is queued).
 *
 * @param settleMs - Milliseconds to wait after the frames (default 30).
 * @returns A promise of the moment.
 */
export function nextPaint(settleMs = 30): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(() => setTimeout(resolve, settleMs)))
  })
}
