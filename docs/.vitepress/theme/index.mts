/// <reference types="vite/client" />
import Layout from "./Layout.vue";
// 使用vitepress系统默认主题
import DefaultTheme from "vitepress/theme";
import { h } from "vue"
import NavLinks from "./components/NavLinks.vue";
import Breadcrumb from "./components/Breadcrumb.vue";
import ArticleShare from "./components/ArticleShare.vue";
import ScrollToTop from "./components/ScrollToTop.vue";
import { useData,EnhanceAppContext } from "vitepress";
export default {
    extends: DefaultTheme,
    // 入口组件
    Layout: () => {
        const props: Record<string, any> = {}
        
        return h(Layout, props)
    },
    enhanceApp({ app, router }: EnhanceAppContext) {
        app.component('NavLinks', NavLinks),
        app.component('Breadcrumb', Breadcrumb);
        app.component('ArticleShare', ArticleShare);
        app.component('ScrollToTop', ScrollToTop)
        
        // router.onBeforeRouteChange = (to)=>{
        //     if(import.meta.env.MODE === 'production'){
        //         // 百度统计代码
        //         try {
        //             if (typeof window === 'undefined' && !!to){
        //             // @ts-ignore
        //             _hmt.push(['_trackPageview', to]);
        //             }
        //         } catch (err) {
        //             console.error('Baidu Tongji Error:', err);
        //         }
        //     }
        // }
    }
}