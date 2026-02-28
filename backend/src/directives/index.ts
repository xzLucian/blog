import type { App } from 'vue'
import { setupHighlightDirective } from './business/highlight'
import { setupRippleDirective } from './business/ripple'
export function setupGlobDirectives(app: App) {
  setupHighlightDirective(app) // 高亮指令
  setupRippleDirective(app) // 水波纹指令
}
