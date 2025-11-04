---
pageClass: m-nav-layout
outline: [2, 3, 4]
---

<script setup>
import { NAV_DATA } from './data'
</script>
<style src="./index.scss"></style>


# 网站导航

<NavLinks v-for="{title, items} in NAV_DATA" :title="title" :items="items"/>

<br />

