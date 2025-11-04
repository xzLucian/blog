import Layout from "./Layout.vue";
// 使用vitepress系统默认主题
import DefaultTheme from "vitepress/theme";
import { h } from "vue"
import NavLinks from "./components/NavLinks.vue";
import { useData,EnhanceAppContext } from "vitepress";

export default {
    extends: DefaultTheme,
    // 入口组件
    Layout: () => {
        const props: Record<string, any> = {}
        
        return h(Layout, props)
    },
    enhanceApp({ app, router }: EnhanceAppContext) {
        app.component('NavLinks', NavLinks)
    }
}