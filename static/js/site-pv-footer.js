(function () {
  function ensureSourceNode() {
    if (document.getElementById('busuanzi_value_site_pv')) return;

    var hidden = document.createElement('div');
    hidden.style.display = 'none';

    var container = document.createElement('span');
    container.id = 'busuanzi_container_site_pv';

    var value = document.createElement('span');
    value.id = 'busuanzi_value_site_pv';

    container.appendChild(value);
    hidden.appendChild(container);
    document.body.appendChild(hidden);
  }

  function ensureVisibleNode() {
    var footer = document.querySelector('body > footer') || document.querySelector('footer:last-of-type');
    if (!footer) return null;

    var existing = document.getElementById('site-pv-footer');
    if (existing) return existing;

    var wrapper = document.createElement('div');
    wrapper.id = 'site-pv-footer';
    wrapper.className = 'site-pv-footer';
    wrapper.style.display = 'none';

    var label = document.createElement('span');
    label.textContent = '全站浏览量 '; 

    var value = document.createElement('span');
    value.id = 'site-pv-footer-value';
    value.textContent = '0';

    wrapper.appendChild(label);
    wrapper.appendChild(value);
    footer.appendChild(wrapper);

    return wrapper;
  }

  function loadBusuanzi() {
    if (document.querySelector('script[data-busuanzi="true"]')) return;
    var script = document.createElement('script');
    script.src = 'https://busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js';
    script.async = true;
    script.setAttribute('data-busuanzi', 'true');
    document.head.appendChild(script);
  }

  function syncValue() {
    var source = document.getElementById('busuanzi_value_site_pv');
    var target = document.getElementById('site-pv-footer-value');
    var wrapper = document.getElementById('site-pv-footer');
    if (source && target && wrapper && source.textContent.trim()) {
      target.textContent = source.textContent.trim();
      wrapper.style.display = 'block';
    }
  }

  document.addEventListener('DOMContentLoaded', function () {
    ensureSourceNode();
    ensureVisibleNode();
    loadBusuanzi();

    var timer = setInterval(syncValue, 500);
    setTimeout(function () {
      clearInterval(timer);
      syncValue();
    }, 15000);
  });
})();
