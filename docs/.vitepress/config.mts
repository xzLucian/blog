import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Byte Size",
  description: "my blog website",
  head: [
    ['link', { rel: 'icon', href: '/web-title.png' }],
    ['script', {
      src: 'https://esm.sh/giscus',
      type: 'module'
    }], ['script', {}, `
      var _hmt = _hmt || [];
      (function() {
        var hm = document.createElement("script");
        hm.src = "https://hm.baidu.com/hm.js?d589414480af5d457fc3fb0e02b5c230";
        var s = document.getElementsByTagName("script")[0];
        s.parentNode.insertBefore(hm, s);
        /* 下面的代码非常重要！！！ 文档：https://tongji.baidu.com/web/help/article?id=324&type=0 */
        _hmt.push(['_requirePlugin', 'UrlChangeTracker', {
          shouldTrackUrlChange: function (newPath, oldPath) {
            console.log('newPath=', newPath, ';oldPath=', oldPath) /* 控制台可查看日志，发布时请删除此行代码 */
            return newPath && oldPath;
          }}
        ]);
      })();
      `]
  ],
  lastUpdated: true,
  // 使用干净链接（无 .html 后缀）
  cleanUrls: true,
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    logo: { light: '/light-logo.png', dark: '/dark-logo.png' },
    siteTitle: false,
    nav: [
      { text: '首页', link: '/' },
      { text: '导航', link: '/navigation' },
      {
        text: '笔记', items: [
          {
            text: 'LangChain v1.0', link: '/langchain'
          }
        ]
      },{
        text: '手册', items: [
          {
            text: 'ArchLinux', link: '/manual/archlinux'
          }
        ]
      }
    ],
    search: {
      provider: 'algolia',
      options: {
        appId: 'BENK8RRKMG',
        apiKey: '09c18b4c05ef81b94c0c03f713c9b9c0',
        indexName: 'ByteSize',
        askAi: {
          assistantId: '87f3jFNLH9lU'
        }
      }
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
      message: 'Copyright © 2025-至今 <a class="vitepress" target="_blank" href="https://beian.miit.gov.cn/#/Integrated/index">赣ICP备2025075792号-1</a>',
      copyright: 'Created by <a class="vitepress" target="_blank" href="http://bytesize.asia/">ByteSize</a>'
    }
  }
})
