import { defineConfig } from 'vitepress'

export interface Footer{
  message?:string
  copyright?:string
}

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Cesar Code",
  description: "my blog website",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    logo: "/logo.png",
    siteTitle: false,
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Notes', link: '/langchain' }
    ],
    search:{
      provider:'local'
    },
    // sidebar: [
    //   {

    //     text: 'Notes',
    //     collapsed: true,

    //     items: [
    //       { text: 'LangChain', link: '/LangChain' },
    //       { text: 'Runtime API Examples', link: '/api-examples' }
    //     ]
    //   }
    // ],
    sidebar: {
      '/langchain/': [
        {
          text: 'LangChain',
          collapsed: true,
          items: [
            { text: 'Introduction', link: '/langchain/' },
            { text: 'Model I/O', link: '/langchain/ModelIO' },
            { text: 'Chains', link: '/langchain/Chains' },
            { text: 'Memory', link: '/langchain/Memory' },
            { text: 'Tools', link: '/langchain/Tools' },
            { text: 'Agents', link: '/langchain/Agents' },
            { text: 'Retrieval', link: '/langchain/Retrieval' },
          ]
        }
      ],
    },
    footer:{
      message:"Release under the MIT License.",
      copyright:"Copyright ©️ 2025-present Evan You"
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/vuejs/vitepress' }
    ]
  }
})
