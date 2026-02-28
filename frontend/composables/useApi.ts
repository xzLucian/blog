const API_BASE = 'http://localhost:3000'

export const useApi = <T>(url: string, opts?: Record<string, any>) => {
  return useAsyncData<T>(url, async () => {
    const res = await $fetch<{ code: number; data: T; msg: string }>(API_BASE + url, opts)
    return res.data
  })
}
