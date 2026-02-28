import { ref } from 'vue'

export function useCeremony() {
  const currentFestivalData = ref<{ date?: string; scrollText?: string } | null>(null)
  const openFestival = () => {}
  const cleanup = () => {}
  return { currentFestivalData, openFestival, cleanup }
}
