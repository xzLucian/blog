import { defineConfig } from 'vitepress'

// export interface Footer{
//   message?:string
//   copyright?:string
// }

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Byte Size",
  description: "my blog website",
  head: [
    ['link', { rel: 'icon', href: '/web-title.png' }],
  ],
  lastUpdated: true,
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    logo: { light: '/light-logo.png', dark: '/dark-logo.png' },
    siteTitle: false,
    nav: [
      { text: '首页', link: '/' },
      { text: '网站导航', link: '/navigation' },
      {
        text: '笔记', items: [
          {
            text: 'LangChain v1.0', link: '/langchain'
          }
        ]
      }
    ],
    search: {
      provider: 'local'
    },
    sidebar: {
      '/langchain/': [
        {
          text: 'LangChain',
          // collapsed: true,
          items: [
            { text: 'Get started', link: '/langchain/' },
            {
              text: 'Core components', items: [
                { text: 'Models', link: '/langchain/models' },
                { text: 'Agents', link: '/langchain/agents' },
                { text: 'Messages', link: '/langchain/messages' },
                { text: 'Tools', link: '/langchain/tools' },
                { text: 'Middleware', link: '/langchain/middleware' },
                { text: 'Streaming', link: '/langchain/streaming' },
                { text: 'Short-term memory', link: '/langchain/short-term-memory' },
                { text: 'Structured output', link: '/langchain/structured-output' },
              ]
            }
          ]
        }
      ],
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/xzLucian' }
    ],
    editLink: {
      pattern: 'https://github.com/xzLucian/blog/tree/main/docs/:path'
    },
    footer: {
      message: 'MIT Licensed | Copyright © 2025-2026 <a class="vitepress" target="_blank" href="http://bytesize.asia/">ByteSize</a>',
      copyright: 'Powered by <a class="vitepress" target="_blank" href="//vitepress.vuejs.org/">VitePress - 1.6.4</a>'
    }
  }
})
