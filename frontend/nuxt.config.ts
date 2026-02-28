import { defineNuxtConfig } from 'nuxt/config'

export default defineNuxtConfig({
  devtools: { enabled: true },
  modules: ['@pinia/nuxt', '@vueuse/nuxt'],
  plugins: ['~/plugins/color-mode'],
  app: {
    head: {
      title: 'ByteSize',
      titleTemplate: '%s - ByteSize',
      meta: [
        { name: 'viewport', content: 'width=device-width, initial-scale=1' },
        { name: 'description', content: 'Personal blog' },
      ],
      link: [
        { rel: 'icon', type: 'image/png', href: '/favicon.png' },
        { rel: 'apple-touch-icon', href: '/favicon.png' },
      ],
    },
  },
  css: ['~/assets/styles/main.scss'],
  nitro: {
    devProxy: {
      '/api': { target: 'http://localhost:3000/api', changeOrigin: true },
      '/uploads': { target: 'http://localhost:3000/uploads', changeOrigin: true },
    },
    routeRules: {
      '/api/**': { proxy: 'http://localhost:3000/api/**' },
      '/uploads/**': { proxy: 'http://localhost:3000/uploads/**' },
    },
  },
  typescript: {
    strict: true,
    tsConfig: {
      compilerOptions: {
        resolveJsonModule: true,
      },
    },
  },
})
