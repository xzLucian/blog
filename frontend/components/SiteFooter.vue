<template>
  <footer class="site-footer">
    <div class="site-footer__inner">© 2025-{{ new Date().getFullYear() }} 赣ICP备2025075792号-1</div>
  </footer>
</template>

<style scoped lang="scss">
.site-footer {
  width: 100%;
  margin: 0 auto;
  padding: 0 1.25rem 2.5rem;
  font-size: 0.75rem;
  color: rgb(var(--c-muted));
  display: flex;
  justify-content: center;
}

.site-footer__inner {
  width: 100%;
  max-width: 68rem;
  padding-top: 1.5rem;
  border-top: 1px solid rgb(var(--c-border));
  text-align: center;
}
</style>
