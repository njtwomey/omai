/** The `localStorage` key holding the colour preference (shared by the three sites when they share an origin). */
export const COLOUR_STORAGE_KEY = 'whitebox-colour-mode'

/** The media query of the system's dark mode. */
export const DARK_QUERY = '(prefers-color-scheme: dark)'

/**
 * The inline script that marks the document before first paint, so a dark page never flashes white: the `dark` class
 * (Tailwind's dark variant), `data-theme` and `color-scheme`, from the stored preference or else the system setting.
 * It repeats the rule of `resolveMode` and `applyMode` because it runs before any module loads.
 */
export const noFlashScript: string = `(function(){try{var p=localStorage.getItem('${COLOUR_STORAGE_KEY}');var d=p==='dark'||(p!=='light'&&matchMedia('${DARK_QUERY}').matches);var r=document.documentElement;r.classList.toggle('dark',d);r.dataset.theme=d?'dark':'light';r.style.colorScheme=d?'dark':'light'}catch(e){}})()`
