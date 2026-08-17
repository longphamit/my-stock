/* === GLOBAL STATE === */
        /* ====================================================
           GLOBAL STATE
        ==================================================== */
        let vnallData = [];
        let vn100Data = [];
        let watchlist = [];
        let predictions = {};
        let activeTicker = '';
        let activeSidebarTab = 'gold';
        let lastStockTab = 'watchlist';
        let activeGoldSubTab = 'prices';
        let activeChartTab = 'price';
        let stockChart = null;
        let currentSortMode = 'alphabetical';
        let watchlistSortMode = 'alpha';
        let activeModel = 'random_forest';
        let activePredictionData = null;
        let showBollingerBands = false;

        // Gold Prices State
        let goldData = null;
        let goldPredictionHistory = [];
        let isFetchingGold = false;
        let tradingViewWidgetLoaded = false;
        let activeGoldModel = null;   // null = auto-select highest R² on first load
        let goldChart = null;
        let dxyChart = null;
        let oilChart = null;
        let djiChart = null;
        let goldGldChart = null;
        let activeGoldChartTimeframe = '30d';
        let lastGoldPredData = null;
        let lastGoldModelData = null;

        // Vietlott State
        let activeVietlottGame = 'power655';
        let isFetchingVietlott = false;
        let isFetchingVietlottTime = 0;
        let vietlottData = {};
        let activeVietlottSubTab = 'stats';
        let activeVietlottModel = 'ensemble';

        // Stock Market Lazy Loading State
        let marketDataLoaded = false;
        let isListLoading = false;

        /* ====================================================
           LIVE CLOCK
        ==================================================== */

        function updateClock() {
            const now = new Date();
            
            // Vietnam Time (Asia/Ho_Chi_Minh)
            let vnTimeStr = "";
            try {
                const vnFormatter = new Intl.DateTimeFormat("en-US", {
                    timeZone: "Asia/Ho_Chi_Minh",
                    hour12: false,
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit"
                });
                vnTimeStr = vnFormatter.format(now);
            } catch (err) {
                const h = String(now.getHours()).padStart(2, '0');
                const m = String(now.getMinutes()).padStart(2, '0');
                const s = String(now.getSeconds()).padStart(2, '0');
                vnTimeStr = `${h}:${m}:${s}`;
            }
            const vnEl = document.getElementById('live-clock-vn');
            if (vnEl) vnEl.textContent = vnTimeStr;

            // US Time (America/New_York)
            let usTimeStr = "";
            try {
                const usFormatter = new Intl.DateTimeFormat("en-US", {
                    timeZone: "America/New_York",
                    hour12: false,
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit"
                });
                usTimeStr = usFormatter.format(now);
            } catch (err) {
                const utc = now.getTime() + (now.getTimezoneOffset() * 60000);
                const nyDate = new Date(utc + (3600000 * -4));
                const h = String(nyDate.getHours()).padStart(2, '0');
                const m = String(nyDate.getMinutes()).padStart(2, '0');
                const s = String(nyDate.getSeconds()).padStart(2, '0');
                usTimeStr = `${h}:${m}:${s}`;
            }
            const usEl = document.getElementById('live-clock-us');
            if (usEl) usEl.textContent = usTimeStr;

            // Update New York time and Gold Market status
            if (activeSidebarTab === 'gold' || activeSidebarTab === 'calendar') {
                try {
                    const options = {
                        timeZone: "America/New_York",
                        hour12: false,
                        weekday: "short",
                        hour: "2-digit",
                        minute: "2-digit",
                        second: "2-digit"
                    };
                    const formatter = new Intl.DateTimeFormat("en-US", options);
                    const parts = formatter.formatToParts(now);
                    const item = {};
                    parts.forEach(p => { item[p.type] = p.value; });
                    
                    const nyHour = parseInt(item.hour);
                    const nyMinute = parseInt(item.minute);
                    const nyWeekday = item.weekday; // 'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'
                    
                    const usTimeEl = document.getElementById('market-us-time');
                    if (usTimeEl) {
                        usTimeEl.textContent = `Giờ Mỹ: ${nyWeekday} ${item.hour}:${item.minute}:${item.second}`;
                    }
                    
                    // Gold market status logic (OTC trades 24/5)
                    // Opens Sunday 18:00 EST, Closes Friday 17:00 EST. Daily break 17:00-18:00 EST Mon-Thu.
                    let isOpen = true;
                    if (nyWeekday === 'Sat') {
                        isOpen = false;
                    } else if (nyWeekday === 'Fri' && nyHour >= 17) {
                        isOpen = false;
                    } else if (nyWeekday === 'Sun' && nyHour < 18) {
                        isOpen = false;
                    } else if (nyHour === 17) {
                        // Daily 1-hour maintenance break
                        isOpen = false;
                    }
                    
                    const statusBadge = document.getElementById('market-status-badge');
                    if (statusBadge) {
                        if (isOpen) {
                            statusBadge.textContent = 'MỞ CỬA';
                            statusBadge.style.background = 'rgba(16, 185, 129, 0.12)';
                            statusBadge.style.color = '#34d399';
                            statusBadge.style.borderColor = 'rgba(16, 185, 129, 0.2)';
                        } else {
                            statusBadge.textContent = 'ĐÓNG CỬA';
                            statusBadge.style.background = 'rgba(239, 68, 68, 0.12)';
                            statusBadge.style.color = '#f87171';
                            statusBadge.style.borderColor = 'rgba(239, 68, 68, 0.2)';
                        }
                    }
                } catch (e) {
                    console.error("Error updating US clock:", e);
                }
            }
        }
        updateClock();
        setInterval(updateClock, 1000);

        function setStatus(online) {
            const dot = document.getElementById('status-dot');
            const label = document.getElementById('status-label');
            if (online) {
                dot.classList.remove('offline');
                label.textContent = 'Trực tuyến · HOSE';
            } else {
                dot.classList.add('offline');
                label.textContent = 'Mất kết nối';
            }
        }



        async function loadMarketData() {
            if (marketDataLoaded || isListLoading) return;
            isListLoading = true;

            const refreshBtn = document.getElementById('refresh-btn');
            if (refreshBtn) refreshBtn.querySelector('i').style.animation = 'spin 0.6s linear infinite';
            showToast("Đang kết nối tải bảng giá...", "info");

            // Show skeleton rows first
            renderSkeletonRows('watchlist-table-body', 5, 6);
            renderSkeletonRows('market-table-body', 5, 4);

            try {
                const [vnallRes, vn100Res] = await Promise.all([
                    fetch('/api/vnall'),
                    fetch('/api/vn100')
                ]);

                if (!vnallRes.ok) throw new Error("Lỗi tải bảng giá VNAllShare.");
                if (!vn100Res.ok)  throw new Error("Lỗi tải bảng giá VN100.");

                vnallData = await vnallRes.json();
                vn100Data  = await vn100Res.json();

                vnallData.sort((a, b) => a.ticker.localeCompare(b.ticker));
                vn100Data.sort((a, b)  => a.ticker.localeCompare(b.ticker));

                // Load watchlist from DB
                try {
                    const wlRes = await fetch('/api/watchlist');
                    if (wlRes.ok) watchlist = await wlRes.json();
                } catch {}

                document.getElementById('vnall-count').textContent = vnallData.length;
                document.getElementById('vn100-count').textContent = vn100Data.length;
                document.getElementById('watchlist-count').textContent = watchlist.length;

                marketDataLoaded = true;

                renderMarketTable();
                renderWatchlistTable();

                if (watchlist.length > 0 && !activeTicker) {
                    selectTicker(watchlist[0]);
                } else if (activeTicker) {
                    selectTicker(activeTicker);
                }

                setStatus(true);
                showToast("Đã tải dữ liệu bảng giá thành công!", "success");
                runBatchPredictionForWatchlist();

            } catch (err) {
                setStatus(false);
                showToast(err.message, "error");
            } finally {
                isListLoading = false;
                if (refreshBtn) refreshBtn.querySelector('i').style.animation = '';
            }
        }

        async function handleRefreshClick() {
            if (activeSidebarTab === 'gold' || activeSidebarTab === 'calendar') {
                const refreshBtn = document.getElementById('refresh-btn');
                if (refreshBtn) refreshBtn.querySelector('i').style.animation = 'spin 0.6s linear infinite';
                showToast("Đang cập nhật giá vàng trực tuyến...", "info");
                
                try {
                    await fetchGoldData();
                    renderGoldSidebar();
                    renderGoldDashboard();
                    showToast("Đã cập nhật giá vàng mới nhất!", "success");
                } catch (err) {
                    showToast("Mất kết nối cập nhật giá vàng", "error");
                } finally {
                    if (refreshBtn) refreshBtn.querySelector('i').style.animation = '';
                }
            } else {
                marketDataLoaded = false;
                await loadMarketData();
            }
        }

        /* ====================================================
           SKELETON LOADER
        ==================================================== */
        function renderSkeletonRows(tbodyId, rows, cols) {
            const tbody = document.getElementById(tbodyId);
            if (!tbody) return;
            const widths = ['40px','60px','50px','55px','70px','28px'];
            tbody.innerHTML = Array.from({ length: rows }, () => `
                <tr class="skeleton-row">
                    ${Array.from({ length: cols }, (_, i) => `
                        <td><span class="skeleton skeleton-cell" style="width:${widths[i] || '50px'}"></span></td>
                    `).join('')}
                </tr>
            `).join('');
        }

        /* ====================================================
           SIDEBAR TAB SWITCHING
        ==================================================== */
        function ensureSidebarExpanded() {
            const mainContainer = document.querySelector('.main-container');
            if (mainContainer && mainContainer.classList.contains('sidebar-collapsed')) {
                toggleSidebar();
            }
        }


        function formatGoldPrice(price) {
            if (price > 1000000) {
                const kVal = Math.round(price / 1000);
                return kVal.toLocaleString('vi-VN');
            }
            return price.toLocaleString('vi-VN');
        }

        function formatUSD(val) {
            if (val === undefined || val === null || isNaN(val)) return '';
            return val.toLocaleString('en-US', {minimumFractionDigits: 1, maximumFractionDigits: 2});
        }


        /* ====================================================
           HEADER SEARCH
        ==================================================== */
        function handleHeaderSearch(event) {
            event.preventDefault();
            const input  = document.getElementById('search-input');
            const ticker = input.value.trim().toUpperCase();
            if (!ticker) return;

            const inWL    = watchlist.includes(ticker);
            const inVn100 = vn100Data.some(s => s.ticker === ticker);
            const inVnall = vnallData.some(s => s.ticker === ticker);

            if (inWL || inVn100 || inVnall) {
                let tab = activeSidebarTab;
                if (activeSidebarTab === 'watchlist' && !inWL) {
                    tab = inVn100 ? 'vn100' : 'vnall';
                } else if (activeSidebarTab === 'vn100' && !inVn100) {
                    tab = inWL ? 'watchlist' : 'vnall';
                } else if (activeSidebarTab === 'vnall' && !inVnall) {
                    tab = inWL ? 'watchlist' : (inVn100 ? 'vn100' : 'vnall');
                }
                if (activeSidebarTab !== tab) switchSidebarTab(tab);
                selectTicker(ticker);
                input.value = '';
                setTimeout(() => {
                    document.querySelector('.row-active')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }, 80);
            } else {
                showToast(`Không tìm thấy mã ${ticker} trong hệ thống.`, "error");
            }
        }

        /* ====================================================
           TOAST
        ==================================================== */
        let toastTimer = null;

        function showToast(message, type = "info") {
            const toast   = document.getElementById('toast-notification');
            const icon    = document.getElementById('toast-icon');
            const msgSpan = document.getElementById('toast-message');

            if (toastTimer) { clearTimeout(toastTimer); toast.classList.remove('show'); }

            toast.className = 'toast';
            if (type === 'success') { toast.classList.add('toast-success'); icon.className = 'fa-solid fa-circle-check'; }
            else if (type === 'error') { toast.classList.add('toast-error'); icon.className = 'fa-solid fa-circle-exclamation'; }
            else { icon.className = 'fa-solid fa-circle-info'; }

            msgSpan.textContent = message;

            requestAnimationFrame(() => {
                toast.classList.add('show');
            });

            toastTimer = setTimeout(() => { toast.classList.remove('show'); }, 3500);
        }

        /* ====================================================
           FORMAT UTILITIES
        ==================================================== */
        function formatPrice(val) {
            if (val === null || val === undefined || isNaN(val)) return '-';
            return parseFloat(val).toLocaleString('vi-VN') + 'đ';
        }

        function formatPriceShort(val) {
            if (val >= 1000000) return (val/1000000).toFixed(1) + 'M';
            if (val >= 1000)    return (val/1000).toFixed(0) + 'K';
            return val.toLocaleString('vi-VN');
        }

        function formatVolume(val) {
            if (val === null || val === undefined) return '-';
            if (val >= 1000000) return (val/1000000).toFixed(2) + 'M';
            if (val >= 1000)    return (val/1000).toFixed(1) + 'K';
            return val.toLocaleString('vi-VN');
        }

        /* ====================================================
           COLLAPSIBLE SIDEBAR & SMOOTH SCROLL FUNCTIONS
        ==================================================== */
        function toggleSidebar() {
            const mainContainer = document.querySelector('.main-container');
            if (!mainContainer) return;
            const isCollapsed = mainContainer.classList.toggle('sidebar-collapsed');
            
            // Save to localStorage
            localStorage.setItem('sidebarCollapsed', isCollapsed ? 'true' : 'false');
            
            // Toggle expand button visibility
            const expandBtn = document.getElementById('expand-sidebar-btn');
            if (expandBtn) {
                expandBtn.style.display = isCollapsed ? 'flex' : 'none';
            }
        }

        function switchGoldSubTab(subTabName) {
            activeGoldSubTab = subTabName;
            
            // Toggle active classes on gold sub-tab buttons
            document.getElementById('tab-gold-prices')?.classList.toggle('active', subTabName === 'prices');
            document.getElementById('tab-gold-forecast')?.classList.toggle('active', subTabName === 'forecast');

            // Toggle visibility of the two sub-views
            const pricesView = document.getElementById('gold-subview-prices');
            const forecastView = document.getElementById('gold-subview-forecast');
            
            if (pricesView) pricesView.style.display = subTabName === 'prices' ? 'block' : 'none';
            if (forecastView) forecastView.style.display = subTabName === 'forecast' ? 'block' : 'none';

            // Ensure chart widget is loaded when prices tab becomes active
            if (subTabName === 'prices' && typeof loadTradingViewGoldWidget === 'function') {
                loadTradingViewGoldWidget();
            }
        }

        function selectCalendarDashboard() {
            scrollToGoldSection('gold-section-calendar');
        }

        function scrollToGoldSection(sectionId) {
            let el = document.getElementById(sectionId);
            
            // Map sectionId to correct page if not on the current page
            const pricesPageSections = ['gold-section-prices', 'gold-section-chart', 'gold-section-brands', 'gold-section-macro', 'gold-section-calendar'];
            const predPageSections = ['gold-mlp-architecture-section', 'gold-forecast-section', 'swing-calculator-section'];
            
            if (!el) {
                if (pricesPageSections.includes(sectionId)) {
                    window.location.href = `/gold#${sectionId}`;
                    return;
                } else if (predPageSections.includes(sectionId)) {
                    window.location.href = `/gold/prediction#${sectionId}`;
                    return;
                }
            }

            if (el) {
                // Determine which gold sub-tab contains this section
                const pricesSections = ['gold-section-prices', 'gold-section-chart', 'gold-section-brands'];
                if (pricesSections.includes(sectionId)) {
                    switchGoldSubTab('prices');
                } else {
                    switchGoldSubTab('forecast');
                }
                
                // Scroll smoothly
                el.scrollIntoView({ behavior: 'smooth', block: 'start' });
                
                // Highlight target section
                el.classList.add('section-highlighted');
                setTimeout(() => {
                    el.classList.remove('section-highlighted');
                }, 1200);
            }
        }

        function scrollToStockSection(sectionId) {
            const el = document.getElementById(sectionId);
            if (el) {
                // Ensure detail panel is showing
                document.getElementById('detail-empty').style.display = 'none';
                document.getElementById('detail-content').style.display = 'flex';
                
                // Scroll smoothly
                el.scrollIntoView({ behavior: 'smooth', block: 'start' });
                
                // Highlight target section
                el.classList.add('stock-section-highlighted');
                setTimeout(() => {
                    el.classList.remove('stock-section-highlighted');
                }, 1200);
            }
        }

        // Initialize sidebar state from localStorage on load
        document.addEventListener('DOMContentLoaded', () => {
            const sidebarCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';
            if (sidebarCollapsed) {
                const mainContainer = document.querySelector('.main-container');
                if (mainContainer) mainContainer.classList.add('sidebar-collapsed');
                
                const expandBtn = document.getElementById('expand-sidebar-btn');
                if (expandBtn) expandBtn.style.display = 'flex';
            }

            // Scroll to hash section if present in URL
            if (window.location.hash) {
                const targetId = window.location.hash.substring(1);
                setTimeout(() => {
                    scrollToGoldSection(targetId);
                }, 400);
            }
        });

