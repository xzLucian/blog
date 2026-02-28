import { computed } from 'vue'
import fastEnterConfig from '@/config/modules/fastEnter'

export function useFastEnter() {
  const enabledApplications = computed(() =>
    (fastEnterConfig.applications || []).filter((a) => a.enabled)
  )
  const enabledQuickLinks = computed(() =>
    (fastEnterConfig.quickLinks || []).filter((l) => l.enabled)
  )
  return { enabledApplications, enabledQuickLinks }
}
