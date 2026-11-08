/**
 * A stylesheet for the render package's HTML components when they are placed in a figure. The components are styled with
 * utility classes and the `--wb-*` colour tokens of the page that hosts them; a figure has no page, so this gives the
 * few classes the contingency table uses and the tokens, in the light palette.
 */
export const TABLE_STYLE = `
.wb { --wb-fg:#1f2937; --wb-muted:#6b7280; --wb-grid:#d1d5db; --wb-surface:#f3f4f6; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; color:#1f2937 }
.wb table { border-collapse: separate }
.wb button { background: none; border: 0; padding: 0; margin: 0; font: inherit; color: inherit; cursor: default; width: 100% }
.wb .flex{display:flex}.wb .flex-col{flex-direction:column}.wb .gap-0\\.5{gap:2px}.wb .items-baseline{align-items:baseline}
.wb .justify-between{justify-content:space-between}.wb .gap-x-1\\.5{column-gap:6px}.wb .rounded{border-radius:4px}
.wb .rounded-md{border-radius:6px}.wb .rounded-full{border-radius:9999px}.wb .px-0\\.5{padding-left:2px;padding-right:2px}
.wb .px-1{padding-left:4px;padding-right:4px}.wb .px-2{padding-left:8px;padding-right:8px}.wb .py-1{padding-top:4px;padding-bottom:4px}
.wb .pb-0\\.5{padding-bottom:2px}.wb .text-left{text-align:left}.wb .text-center{text-align:center}.wb .align-bottom{vertical-align:bottom}
.wb .align-middle{vertical-align:middle}.wb .align-top{vertical-align:top}.wb .whitespace-nowrap{white-space:nowrap}
.wb .font-normal{font-weight:400}.wb .font-medium{font-weight:500}.wb .font-mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.wb .tabular-nums{font-variant-numeric:tabular-nums}.wb .uppercase{text-transform:uppercase}.wb .tracking-wide{letter-spacing:.025em}
.wb .text-xs{font-size:12px}.wb .text-sm{font-size:14px}.wb .text-\\[10px\\]{font-size:10px}.wb .text-\\[11px\\]{font-size:11px}
.wb .text-\\(--wb-muted\\){color:var(--wb-muted)}.wb .text-\\(--wb-fg\\){color:var(--wb-fg)}
.wb .border{border:1px solid}.wb .border-b{border-bottom:1px solid}.wb .border-\\(--wb-grid\\){border-color:var(--wb-grid)}
.wb .bg-\\(--wb-surface\\){background:var(--wb-surface)}.wb .table-fixed{table-layout:fixed}.wb .w-full{width:100%}
.wb .w-\\[18\\%\\]{width:18%}.wb .w-\\[13\\%\\]{width:13%}.wb .w-\\[24\\%\\]{width:24%}.wb .max-w-3xl{max-width:48rem}
.wb .border-spacing-1{border-spacing:4px}.wb .inline-block{display:inline-block}.wb .size-2{width:8px;height:8px}.wb .mr-1{margin-right:4px}
.wb .line-clamp-2{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}.wb .overflow-auto{overflow:auto}
`
