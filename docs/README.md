# vitepress-blog

<p align="center"> 基于 <b>VitePress</b> 的个人博客 </p>

---

## 博客地址

- <http://bytesize.asia/>
- <https://github.com/xzLucian/blog>

## 功能

- 默认中文
- 自带导航模块
- 支持 [giscus 评论](https://giscus.app/zh-CN)
- 支持日夜颜色模式自适应切换
- 支持algolia搜索和Ask AI问答
- 支持 [tailwindcss](https://github.com/tailwindlabs/tailwindcss)

## 本地启动

1. 下载并安装 [Node.js](https://nodejs.org/zh-cn/download) 推荐 20 版本及更高版本
2. 安装依赖 `npm install`
3. 启动本地服务 `npm run docs:dev`
4. 访问 `http://localhost:5173` 查看效果

### 开启 giscus 评论

1. 需在 `docs/.vitepress/config.ts` 中配置 `themeConfig.comment`

```ts
export default defineConfig({
  themeConfig: {
    /**
     * giscus 评论配置
     *  请根据 https://giscus.app/zh-CN 生成内容填写
     *  CDN方式引入
     */
    head: [
    ['script', {
      src: 'https://esm.sh/giscus',
      type: 'module'
    }]
  ],
  },
})
```
2. 在`Layout.vue`中添加`giscus-widget`组件
```vue
<template #doc-after>
  <div style="display: block; height: 40px;"></div>
  <giscus-widget id="comments" repo="xzLucian/blog-comment" repoid="R_kgDOQQSxVw" category="Announcements"
    categoryid="DIC_kwDOQQSxV84CxfmU" mapping="title" term="Welcome to giscus!" reactionsenabled="1"
    emitmetadata="0" inputposition="top" theme="light" lang="en" loading="lazy" />
</template>
```
#### algolia搜索
需在 `docs/.vitepress/config.ts` 中配置 `themeConfig.search`

```ts
export default defineConfig({
  themeConfig: {
    search: {
      provider: 'algolia',
      options: {
        // appId
        appId: ' ',
        // apiKey
        apiKey: ' ',
        // indexName
        indexName: ' ',
        // 配置Ask Ai
        askAi: {
          // assistantId
          assistantId: ' '
        }
      }
    },
  },
})
```