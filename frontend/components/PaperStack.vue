<template>
  <div ref="el" class="paper-stage" @mousemove="onMove" @mouseleave="onLeave">
    <div class="paper-stage__inner">
      <div class="paper paper--shadow" />

      <div class="paper paper--pink">
        <div class="paper__inner paper-code paper-code--light">
          <div class="paper-row">
            <span class="paper-dot paper-dot--light" />
            <span class="paper-label">SQL</span>
          </div>
          <div>
            <span class="paper-strong">SELECT</span> * <span class="paper-strong">FROM</span> world.logic
            <br />
            <span class="paper-strong">WHERE</span> type = "creation"
            <br />
            <span class="paper-strong">AND</span> passion = 100%
            <br />
            <span class="paper-strong">LIMIT</span> 1;
          </div>
        </div>
      </div>

      <div class="paper paper--teal">
        <div class="paper__inner paper-code paper-code--dark">
          <div class="paper-row">
            <span class="paper-dot paper-dot--dark" />
            <span class="paper-label">HTML</span>
          </div>
          <div>
            &lt;div class="<span class="paper-strong">hello</span>"&gt;<br />
            &nbsp;&nbsp;Hello world<br />
            &lt;/div&gt;
          </div>
        </div>
      </div>

      <div class="paper paper--green">
        <div class="paper__inner paper-code paper-code--dark">
          <div class="paper-row">
            <span class="paper-dot paper-dot--dark" />
            <span class="paper-label">JSON</span>
          </div>
          <div>
            {<br />
            &nbsp;&nbsp;"role": "Backend",<br />
            &nbsp;&nbsp;"status": "Building..."<br />
            }<br />
          </div>
        </div>
      </div>

      <div class="paper paper--window">
        <div class="paper__bar">
          <div class="paper-row">
            <span class="paper-dot paper-dot--dark" />
            <span class="paper-dot paper-dot--dark" />
          </div>
          <div class="paper-title">MAIN.JAVA</div>
        </div>
        <div class="paper__inner paper-code paper-code--window">
          <div class="paper-dim">/* Backend Architecture */</div>
          <div>
            <span class="ink-purple paper-strong">public class</span>
            <span class="paper-strong"> Value </span>
            {<br />
            &nbsp;&nbsp;<span class="paper-dim">@Override</span><br />
            &nbsp;&nbsp;<span class="ink-blue paper-strong">public String</span> create() {<br />
            &nbsp;&nbsp;&nbsp;&nbsp;<span class="paper-dim">// Logic...</span><br />
            &nbsp;&nbsp;&nbsp;&nbsp;<span class="ink-orange paper-strong">return</span>
            <span class="ink-orange">"future"</span>;<br />
            &nbsp;&nbsp;}<br />
            }<br />
          </div>

          <div class="paper__stamp">
            <div class="paper-stamp">
              Asa.
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
const el = ref<HTMLElement | null>(null)

const onMove = (e: MouseEvent) => {
  if (!el.value) return
  const rect = el.value.getBoundingClientRect()
  const x = (e.clientX - rect.left) / rect.width - 0.5
  const y = (e.clientY - rect.top) / rect.height - 0.5
  el.value.style.setProperty('--mx', String(x))
  el.value.style.setProperty('--my', String(y))
}

const onLeave = () => {
  if (!el.value) return
  el.value.style.setProperty('--mx', '0')
  el.value.style.setProperty('--my', '0')
}
</script>

<style scoped lang="scss">
.paper-stage {
  position: relative;
  isolation: isolate;
  height: 26rem;
  overflow: visible;
}

.paper-stage__inner {
  position: relative;
  margin: 0 auto;
  transform: translateX(-1.5rem);
  height: 100%;
  width: 100%;
  max-width: 36rem;
}

.paper-code {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  line-height: 1rem;
}

.paper-code--light {
  color: rgba(255, 255, 255, 0.9);
}

.paper-code--dark {
  color: rgba(0, 0, 0, 0.8);
}

.paper-code--window {
  font-size: 0.85rem;
  line-height: 1.5rem;
  color: rgba(0, 0, 0, 0.8);
}

.paper-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}

.paper-dot {
  width: 0.5rem;
  height: 0.5rem;
  border-radius: 999px;
  border: 1px solid rgba(0, 0, 0, 0.6);
}

.paper-dot--light {
  border-color: rgba(255, 255, 255, 0.7);
}

.paper-dot--dark {
  border-color: rgba(0, 0, 0, 0.7);
}

.paper-label {
  font-weight: 600;
  letter-spacing: 0.18em;
}

.paper-strong {
  font-weight: 600;
}

.paper-title {
  font-size: 0.7rem;
  font-weight: 800;
  letter-spacing: 0.18em;
  color: rgba(0, 0, 0, 0.8);
}

.paper-dim {
  color: rgba(0, 0, 0, 0.45);
}

.ink-purple {
  color: #a855f7;
}

.ink-blue {
  color: #2563eb;
}

.ink-orange {
  color: #ea580c;
}

.paper-stamp {
  transform: rotate(-10deg);
  border: 2px solid rgba(0, 0, 0, 0.8);
  padding: 0.25rem 0.75rem;
  font-family: cursive;
  font-size: 1.1rem;
  color: rgba(0, 0, 0, 0.8);
}
.paper {
  position: absolute;
  left: 50%;
  top: 50%;
  --tx: 0px;
  --ty: 0px;
  --r: 0deg;
  --z: 1;
  transform: translate(
      calc(-50% + var(--tx) + (var(--mx, 0) * 12px)),
      calc(-50% + var(--ty) + (var(--my, 0) * 12px))
    )
    rotate(var(--r));
  border: 2px solid rgba(0, 0, 0, 0.9);
  box-shadow: 0 18px 28px rgba(0, 0, 0, 0.18);
  transform-origin: center;
  z-index: var(--z);
  transition:
    transform 220ms ease,
    box-shadow 220ms ease;
}

.paper:hover {
  z-index: 60;
  transform: translate(
      calc(-50% + var(--tx) + (var(--mx, 0) * 12px)),
      calc(-50% + var(--ty) + (var(--my, 0) * 12px))
    )
    rotate(0deg)
    scale(1.03);
}

.paper--shadow:hover {
  z-index: 0;
  transform: translate(
      calc(-50% + var(--tx) + (var(--mx, 0) * 12px)),
      calc(-50% + var(--ty) + (var(--my, 0) * 12px))
    )
    rotate(var(--r));
}

.paper__inner {
  padding: 18px 18px;
}

.paper--shadow {
  --tx: 50px;
  --ty: 26px;
  --r: -8deg;
  --z: 0;
  width: 92%;
  height: 72%;
  background: #000;
  border: none;
  box-shadow: none;
  filter: blur(0.2px);
}

.paper--pink {
  --tx: -110px;
  --ty: -90px;
  --r: -9deg;
  --z: 2;
  width: 72%;
  height: 44%;
  background: #ff7ac8;
}

.paper--teal {
  --tx: -120px;
  --ty: 70px;
  --r: 7deg;
  --z: 3;
  width: 70%;
  height: 40%;
  background: #45d2b7;
}

.paper--green {
  --tx: 95px;
  --ty: 110px;
  --r: 4deg;
  --z: 1;
  width: 82%;
  height: 40%;
  background: #c9e74d;
}

.paper--window {
  --tx: 85px;
  --ty: 0px;
  --r: 6deg;
  --z: 4;
  width: 84%;
  height: 54%;
  background: #fff;
}

.paper--window .paper__inner {
  padding: 20px 22px;
}

.paper__bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 14px;
  border-bottom: 2px solid rgba(0, 0, 0, 0.9);
}

.paper__stamp {
  position: absolute;
  right: 22px;
  bottom: 18px;
}

:global(html.dark) .paper-stage .paper {
  border-color: rgba(255, 255, 255, 0.72);
  box-shadow: 0 22px 34px rgba(0, 0, 0, 0.45);
}

:global(html.dark) .paper-stage .paper--window {
  background: rgba(255, 255, 255, 0.92);
}

@media (prefers-reduced-motion: reduce) {
  .paper {
    transition: none;
  }
}

@media (min-width: 768px) {
  .paper-stage {
    height: 30rem;
  }

  .paper-stage__inner {
    transform: translateX(-2.5rem);
  }
}
</style>
