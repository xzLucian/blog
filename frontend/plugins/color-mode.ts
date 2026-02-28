const STORAGE_KEY = 'blog:color-mode'

export default defineNuxtPlugin(() => {
  const isDark = useState<boolean>('isDark', () => false)

  const apply = (next: boolean) => {
    isDark.value = next
    if (!import.meta.client) return
    document.documentElement.classList.toggle('dark', next)
    localStorage.setItem(STORAGE_KEY, next ? 'dark' : 'light')
  }

  if (import.meta.client) {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved === 'dark') apply(true)
    else if (saved === 'light') apply(false)
    else apply(window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false)
  }

  return {
    provide: {
      colorMode: {
        isDark,
        toggle: () => apply(!isDark.value),
        set: apply,
      },
    },
  }
})

