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

需在 `docs/.vitepress/config.ts` 中配置 `themeConfig.comment`

```ts
export default defineConfig({
  themeConfig: {
    /**
     * giscus 评论配置
     *  请根据 https://giscus.app/zh-CN 生成内容填写
     */
    comment: {
      /** github 仓库地址 */
      repo: '',
      /** giscus 仓库 ID */
      repoId: '',
      /** Discussion 分类 */
      category: '',
      /** giscus 分类 ID */
      categoryId: '',
    },
  },
})
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