(function () {
  function createPvBadge(containerId, valueId, labelText) {
    const wrapper = document.createElement('div');
    wrapper.id = containerId;
    wrapper.className = 'pv-badge';
    wrapper.style.display = 'none';

    const icon = document.createElement('span');
    icon.className = 'pv-icon';
    icon.textContent = '👀';

    const label = document.createElement('span');
    label.textContent = labelText;

    const value = document.createElement('span');
    value.id = valueId;
    value.textContent = '0';

    wrapper.appendChild(icon);
    wrapper.appendChild(label);
    wrapper.appendChild(value);
    return wrapper;
  }

  function addSitePvToBottom() {
    if (document.getElementById('busuanzi_container_site_pv_bottom')) {
      return;
    }

    const targetContainer = document.querySelector('footer') || document.body;
    const siteBadge = createPvBadge(
      'busuanzi_container_site_pv_bottom',
      'busuanzi_value_site_pv_bottom',
      '全站浏览量 '
    );
    siteBadge.classList.add('pv-site-bottom');
    targetContainer.appendChild(siteBadge);
  }

  function addPostPagePv() {
    if (!document.querySelector('article')) {
      return;
    }

    const metaContainer = document.querySelector('article header .text-xs');
    if (!metaContainer || document.getElementById('busuanzi_container_page_pv_post')) {
      return;
    }

    const badge = createPvBadge(
      'busuanzi_container_page_pv_post',
      'busuanzi_value_page_pv_post',
      '本文浏览量 '
    );
    metaContainer.appendChild(badge);
  }

  function syncBusuanziValues() {
    const map = [
      ['busuanzi_value_site_pv', 'busuanzi_value_site_pv_bottom', 'busuanzi_container_site_pv_bottom'],
      ['busuanzi_value_page_pv', 'busuanzi_value_page_pv_post', 'busuanzi_container_page_pv_post']
    ];

    map.forEach(function (item) {
      const source = document.getElementById(item[0]);
      const target = document.getElementById(item[1]);
      const wrapper = document.getElementById(item[2]);

      if (source && target && wrapper && source.textContent.trim()) {
        target.textContent = source.textContent.trim();
        wrapper.style.display = 'block';
      }
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    addSitePvToBottom();
    addPostPagePv();

    const poller = setInterval(syncBusuanziValues, 500);
    setTimeout(function () {
      clearInterval(poller);
      syncBusuanziValues();
    }, 10000);
  });
})();
