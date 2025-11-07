<template>
  <!-- 默认主题 -->
  <Layout>
    <template #doc-top>
      <Breadcrumb />
    </template>
    <template #aside-outline-before>
      <ArticleShare />
    </template>
    <template #layout-bottom>
      <ScrollToTop />
    </template>
    <template #doc-after>
      <div style="display: block; height: 40px;"></div>
      <giscus-widget id="comments" repo="xzLucian/blog-comment" repoid="R_kgDOQQSxVw" category="Announcements"
        categoryid="DIC_kwDOQQSxV84CxfmU" mapping="title" term="Welcome to giscus!" reactionsenabled="1"
        emitmetadata="0" inputposition="top" theme="light" lang="en" loading="lazy" />
    </template>
  </Layout>
</template>

<script setup lang="ts" name="Layout">
import DefaultTheme from 'vitepress/theme'
const { Layout } = DefaultTheme

</script>