document.addEventListener('DOMContentLoaded', () => {
    const widthInput = document.getElementById('width');
    const heightInput = document.getElementById('height');
    const epochsInput = document.getElementById('epochs');
    const trainBtn = document.getElementById('train-btn');
    const randomizeBtn = document.getElementById('randomize-btn');
    
    const gridContainer = document.getElementById('grid-container');
    const replayGridContainer = document.getElementById('replay-grid-container');
    const routeSvg = document.getElementById('route-svg');
    const routeSelectors = document.getElementById('route-selectors');
    
    const trainStatusEl = document.getElementById('training-status');
    const replayStatusEl = document.getElementById('replay-status');
    const replayBtn = document.getElementById('replay-btn');
    
    const modeSelect = document.getElementById('mode-select');
    const basicModeContent = document.getElementById('basic-mode-content');
    const compareModeContent = document.getElementById('compare-mode-content');
    const replayGridContainerDb = document.getElementById('replay-grid-container-db');
    const replayGridContainerDu = document.getElementById('replay-grid-container-du');
    const trainStatusDb = document.getElementById('training-status-db');
    const trainStatusDu = document.getElementById('training-status-du');
    const finalRouteSvgDb = document.getElementById('final-route-svg-db');
    const finalRouteSvgDu = document.getElementById('final-route-svg-du');
    
    let lossChart = null;
    let lossChartDb = null;
    let lossChartDu = null;
    let finalGame = null;
    let finalGameDb = null;
    let finalGameDu = null;
    let capturedRoutes = [];
    
    const colors = ['#3b82f6', '#f43f5e', '#8b5cf6', '#10b981', '#f59e0b'];

    function makeDraggable(container) {
        const cells = container.querySelectorAll('.cell');
        let draggedCell = null;
        
        cells.forEach(cell => {
            cell.setAttribute('draggable', 'true');
            cell.style.cursor = 'grab';
            
            cell.addEventListener('dragstart', function(e) {
                if (!this.classList.contains('player') && 
                    !this.classList.contains('goal') && 
                    !this.classList.contains('pit') && 
                    !this.classList.contains('wall')) {
                    e.preventDefault();
                    return;
                }
                draggedCell = this;
                e.dataTransfer.setData('text/plain', this.id);
                setTimeout(() => this.style.opacity = '0.5', 0);
            });
            
            cell.addEventListener('dragend', function() {
                setTimeout(() => this.style.opacity = '1', 0);
                draggedCell = null;
            });
            
            cell.addEventListener('dragover', function(e) {
                e.preventDefault();
            });
            
            cell.addEventListener('drop', function(e) {
                e.preventDefault();
                if (this === draggedCell) return;
                
                const tempClasses = [...this.classList];
                const tempText = this.innerText;
                
                this.className = draggedCell.className;
                this.innerText = draggedCell.innerText;
                
                draggedCell.className = tempClasses.join(' ');
                draggedCell.innerText = tempText;
                
                routeSvg.innerHTML = '';
            });
        });
    }

    function initGrid(container, w, h, isDraggable = false) {
        container.style.gridTemplateColumns = `repeat(${w}, 1fr)`;
        container.innerHTML = '';
        
        for (let r = 0; r < h; r++) {
            for (let c = 0; c < w; c++) {
                const cell = document.createElement('div');
                cell.className = 'cell';
                cell.id = `${container.id}-cell-${r}-${c}`;
                
                if (r === 0 && c === 0) {
                    cell.classList.add('goal');
                    cell.innerText = '+';
                } else if (r === 0 && c === 1) {
                    cell.classList.add('pit');
                    cell.innerText = '-';
                } else if (r === 1 && c === 1) {
                    cell.classList.add('wall');
                    cell.innerText = 'W';
                }
                
                container.appendChild(cell);
            }
        }
        
        if (isDraggable) makeDraggable(container);
    }

    function randomizePositions() {
        const w = parseInt(widthInput.value);
        const h = parseInt(heightInput.value);
        const totalCells = w * h;
        
        const indices = [];
        while(indices.length < 4) {
            const r = Math.floor(Math.random() * totalCells);
            if(indices.indexOf(r) === -1) indices.push(r);
        }
        
        const cells = gridContainer.querySelectorAll('.cell');
        cells.forEach(c => {
            c.className = 'cell';
            c.innerText = '';
        });
        
        const types = [
            { class: 'player', text: 'P' },
            { class: 'goal', text: '+' },
            { class: 'pit', text: '-' },
            { class: 'wall', text: 'W' }
        ];
        
        for(let i=0; i<4; i++) {
            const cell = cells[indices[i]];
            cell.classList.add(types[i].class);
            cell.innerText = types[i].text;
        }
        routeSvg.innerHTML = '';
    }

    function updatePlayerPos(container, r, c) {
        container.querySelectorAll('.player').forEach(el => {
            el.classList.remove('player');
            if(el.innerText === 'P') el.innerText = '';
        });
        
        const cell = document.getElementById(`${container.id}-cell-${r}-${c}`);
        if (cell) {
            cell.classList.add('player');
            if(cell.innerText === '') cell.innerText = 'P';
        }
    }

    function initChart() {
        const ctx = document.getElementById('lossChart').getContext('2d');
        if (lossChart) lossChart.destroy();
        
        lossChart = new Chart(ctx, {
            type: 'line',
            data: { labels: [], datasets: [{
                label: 'Training Loss (Moving Avg)',
                data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 2, pointRadius: 0, fill: true, tension: 0.4
            }]},
            options: { responsive: true, maintainAspectRatio: false,
                scales: { x: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
                          y: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } } },
                plugins: { legend: { labels: { color: '#f8fafc' } } },
                animation: false
            }
        });
        
        const ctxDb = document.getElementById('lossChartDb').getContext('2d');
        if (lossChartDb) lossChartDb.destroy();
        lossChartDb = new Chart(ctxDb, {
            type: 'line',
            data: { labels: [], datasets: [{
                label: 'Double DQN Loss',
                data: [], borderColor: '#f43f5e', backgroundColor: 'rgba(244, 63, 94, 0.1)',
                borderWidth: 2, pointRadius: 0, fill: true, tension: 0.4
            }]},
            options: { responsive: true, maintainAspectRatio: false,
                scales: { x: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
                          y: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } } },
                plugins: { legend: { labels: { color: '#f8fafc' } } },
                animation: false
            }
        });

        const ctxDu = document.getElementById('lossChartDu').getContext('2d');
        if (lossChartDu) lossChartDu.destroy();
        lossChartDu = new Chart(ctxDu, {
            type: 'line',
            data: { labels: [], datasets: [{
                label: 'Dueling DQN Loss',
                data: [], borderColor: '#10b981', backgroundColor: 'rgba(16, 185, 129, 0.1)',
                borderWidth: 2, pointRadius: 0, fill: true, tension: 0.4
            }]},
            options: { responsive: true, maintainAspectRatio: false,
                scales: { x: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
                          y: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } } },
                plugins: { legend: { labels: { color: '#f8fafc' } } },
                animation: false
            }
        });
    }

    function updateChart(epoch, loss) {
        lossChart.data.labels.push(epoch);
        lossChart.data.datasets[0].data.push(loss);
        lossChart.update();
    }
    
    function updateChartCompare(epoch, lossDb, lossDu) {
        lossChartDb.data.labels.push(epoch);
        lossChartDb.data.datasets[0].data.push(lossDb);
        lossChartDb.update();
        
        lossChartDu.data.labels.push(epoch);
        lossChartDu.data.datasets[0].data.push(lossDu);
        lossChartDu.update();
    }
    
    function drawRoutes() {
        routeSvg.innerHTML = '';
        
        const checkedBoxes = Array.from(routeSelectors.querySelectorAll('input[type="checkbox"]:checked'));
        if (checkedBoxes.length === 0) return;
        
        const cellSize = 60;
        const gapSize = 4;
        
        checkedBoxes.forEach(box => {
            const index = parseInt(box.value);
            const route = capturedRoutes[index].route;
            const color = colors[index % colors.length];
            
            if (route.length < 2) return;
            
            let d = '';
            for(let i=0; i<route.length; i++) {
                const r = route[i].pos[0];
                const c = route[i].pos[1];
                const x = c * (cellSize + gapSize) + cellSize/2 + 8;
                const y = r * (cellSize + gapSize) + cellSize/2 + 8;
                
                const ox = x + (Math.random() * 8 - 4);
                const oy = y + (Math.random() * 8 - 4);
                
                if (i === 0) d += `M ${ox} ${oy} `;
                else d += `L ${ox} ${oy} `;
            }
            
            const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
            path.setAttribute("d", d);
            path.setAttribute("fill", "none");
            path.setAttribute("stroke", color);
            path.setAttribute("stroke-width", "4");
            path.setAttribute("stroke-linejoin", "round");
            path.setAttribute("stroke-linecap", "round");
            path.setAttribute("opacity", "0.8");
            
            if (route.length > 0) {
                const last = route[route.length-1];
                const lx = last.pos[1] * (cellSize + gapSize) + cellSize/2 + 8;
                const ly = last.pos[0] * (cellSize + gapSize) + cellSize/2 + 8;
                const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                circle.setAttribute("cx", lx);
                circle.setAttribute("cy", ly);
                circle.setAttribute("r", "5");
                circle.setAttribute("fill", color);
                routeSvg.appendChild(circle);
            }
            
            routeSvg.appendChild(path);
        });
    }

    function buildRouteSelectors() {
        routeSelectors.innerHTML = '';
        if (capturedRoutes.length === 0) {
            routeSelectors.innerHTML = '<p class="placeholder-text">No routes captured.</p>';
            return;
        }
        
        capturedRoutes.forEach((cr, i) => {
            const label = document.createElement('label');
            const color = colors[i % colors.length];
            label.style.color = color;
            label.style.fontWeight = 'bold';
            
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.value = i;
            checkbox.addEventListener('change', drawRoutes);
            
            label.appendChild(checkbox);
            label.appendChild(document.createTextNode(` Epoch ${cr.epoch}`));
            routeSelectors.appendChild(label);
        });
    }

    function applyInitPos(initPos, container) {
        if (!initPos) return;
        const cells = container.querySelectorAll('.cell');
        cells.forEach(c => {
            c.className = 'cell';
            c.innerText = '';
        });
        const w = parseInt(widthInput.value);
        
        function setCell(type, pos, symbol) {
            const idx = pos[0] * w + pos[1];
            if (idx >= 0 && idx < cells.length) {
                cells[idx].classList.add(type);
                cells[idx].innerText = symbol;
            }
        }
        setCell('player', initPos.player, 'P');
        setCell('goal', initPos.goal, '+');
        setCell('pit', initPos.pit, '-');
        setCell('wall', initPos.wall, 'W');
    }

    async function playFinalGame() {
        if (!finalGame) return;
        
        replayBtn.disabled = true;
        replayStatusEl.innerText = 'Playing...';
        replayStatusEl.className = 'status running badge';
        
        // Sync starting position with the top grid
        const w = parseInt(widthInput.value);
        const h = parseInt(heightInput.value);
        
        // Find player start in final game
        if (finalGame.length > 0) {
            const start = finalGame[0].pos;
            updatePlayerPos(replayGridContainer, start[0], start[1]);
            await new Promise(r => setTimeout(r, 500));
            
            for (let step of finalGame) {
                updatePlayerPos(replayGridContainer, step.pos[0], step.pos[1]);
                await new Promise(r => setTimeout(r, 200));
            }
        }
        
        replayBtn.disabled = false;
        replayStatusEl.innerText = 'Finished';
        replayStatusEl.className = 'status success badge';
    }

    function drawStaticRoute(route, svgElement, color) {
        svgElement.innerHTML = '';
        if (!route || route.length < 2) return;
        
        const cellSize = 60;
        const gapSize = 4;
        let d = '';
        
        for(let i=0; i<route.length; i++) {
            const r = route[i].pos[0];
            const c = route[i].pos[1];
            const x = c * (cellSize + gapSize) + cellSize/2 + 8;
            const y = r * (cellSize + gapSize) + cellSize/2 + 8;
            
            if (i === 0) d += `M ${x} ${y} `;
            else d += `L ${x} ${y} `;
        }
        
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", d);
        path.setAttribute("fill", "none");
        path.setAttribute("stroke", color);
        path.setAttribute("stroke-width", "4");
        path.setAttribute("stroke-linejoin", "round");
        path.setAttribute("stroke-linecap", "round");
        path.setAttribute("opacity", "0.9");
        
        const last = route[route.length-1];
        const lx = last.pos[1] * (cellSize + gapSize) + cellSize/2 + 8;
        const ly = last.pos[0] * (cellSize + gapSize) + cellSize/2 + 8;
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", lx);
        circle.setAttribute("cy", ly);
        circle.setAttribute("r", "5");
        circle.setAttribute("fill", color);
        
        svgElement.appendChild(path);
        svgElement.appendChild(circle);
    }

    function showFinalRoutesCompare() {
        syncReplayGrid();
        drawStaticRoute(finalGameDb, finalRouteSvgDb, '#f59e0b');
        drawStaticRoute(finalGameDu, finalRouteSvgDu, '#f59e0b');
    }

    function setupGrids() {
        const w = parseInt(widthInput.value);
        const h = parseInt(heightInput.value);
        initGrid(gridContainer, w, h, true);
        initGrid(replayGridContainer, w, h, false);
        initGrid(replayGridContainerDb, w, h, false);
        initGrid(replayGridContainerDu, w, h, false);
        updatePlayerPos(gridContainer, 0, w - 1);
        
        // Copy layout from top grid to replay grid
        syncReplayGrid();
        
        routeSvg.innerHTML = '';
        if (finalRouteSvgDb) finalRouteSvgDb.innerHTML = '';
        if (finalRouteSvgDu) finalRouteSvgDu.innerHTML = '';
        routeSelectors.innerHTML = '<p class="placeholder-text">Routes will appear here after training.</p>';
    }
    
    function syncReplayGrid() {
        const cells = gridContainer.querySelectorAll('.cell');
        const replayCells = replayGridContainer.querySelectorAll('.cell');
        const replayCellsDb = replayGridContainerDb.querySelectorAll('.cell');
        const replayCellsDu = replayGridContainerDu.querySelectorAll('.cell');
        
        cells.forEach((cell, i) => {
            replayCells[i].className = cell.className;
            replayCells[i].innerText = cell.innerText;
            replayCellsDb[i].className = cell.className;
            replayCellsDb[i].innerText = cell.innerText;
            replayCellsDu[i].className = cell.className;
            replayCellsDu[i].innerText = cell.innerText;
        });
    }

    // Init
    setupGrids();
    initChart();
    
    modeSelect.addEventListener('change', (e) => {
        const isLightning = e.target.value === 'lightning_random';
        if (e.target.value === 'compare') {
            basicModeContent.style.display = 'none';
            compareModeContent.style.display = 'flex';
            if (lossChartDb) lossChartDb.resize();
            if (lossChartDu) lossChartDu.resize();
        } else {
            basicModeContent.style.display = 'block';
            compareModeContent.style.display = 'none';
            if (lossChart && !isLightning) lossChart.resize();
        }
        
        const lightningInfoPanel = document.getElementById('lightning-info-panel');
        if (lightningInfoPanel) {
            lightningInfoPanel.style.display = isLightning ? 'block' : 'none';
        }
        const lightningControls = document.getElementById('lightning-controls');
        if (lightningControls) {
            lightningControls.style.display = 'none'; // no training config in random mode
        }

        // Hide/show training-only panels
        const sampledRoutesPanel = document.getElementById('sampled-routes-panel');
        if (sampledRoutesPanel) sampledRoutesPanel.style.display = isLightning ? 'none' : 'block';
        const replayPanelWrapper = document.getElementById('replay-panel-wrapper');
        if (replayPanelWrapper) replayPanelWrapper.style.display = isLightning ? 'none' : 'block';
        const mainContent = document.querySelector('.main-content');
        if (mainContent) mainContent.style.gridTemplateColumns = isLightning ? '1fr' : '1fr 1fr';

        // Hide Start Training button and epochs in random mode
        const trainBtn_ = document.getElementById('train-btn');
        if (trainBtn_) trainBtn_.style.display = isLightning ? 'none' : '';
        const epochsGroup = document.getElementById('epochs-group');
        if (epochsGroup) epochsGroup.style.display = isLightning ? 'none' : '';

        // Show validation grid immediately in lightning_random mode
        const interactiveValidationPanel = document.getElementById('interactive-validation-panel');
        if (interactiveValidationPanel) {
            interactiveValidationPanel.style.display = isLightning ? 'block' : 'none';
        }
        if (isLightning) {
            const w = parseInt(widthInput.value);
            const h = parseInt(heightInput.value);
            if (typeof initValidationGrid === 'function') initValidationGrid(w, h);
            const verifyBtnEl = document.getElementById('verify-btn');
            if (verifyBtnEl) verifyBtnEl.disabled = false;
        }

        if (!isLightning) initChart();
        setupGrids();
        trainStatusEl.innerText = 'Awaiting Training...';
        trainStatusEl.className = 'status badge';
        replayStatusEl.innerText = 'Awaiting Training...';
        replayStatusEl.className = 'status badge';
        trainStatusDb.innerText = 'Awaiting Training...';
        trainStatusDb.className = 'status badge';
        trainStatusDu.innerText = 'Awaiting Training...';
        trainStatusDu.className = 'status badge';
        if (!isLightning) {
            trainBtn.innerText = 'Start Training';
            trainBtn.disabled = false;
        }
        replayBtn.disabled = true;
        const lrStatus = document.getElementById('lr-status');
        if (lrStatus) lrStatus.style.display = 'none';
    });

    function onGridSizeChange() {
        setupGrids();
        if (modeSelect.value === 'lightning_random') {
            const w = parseInt(widthInput.value);
            const h = parseInt(heightInput.value);
            if (typeof initValidationGrid === 'function') initValidationGrid(w, h);
            document.getElementById('validation-route-svg').innerHTML = '';
        }
    }
    widthInput.addEventListener('change', onGridSizeChange);
    heightInput.addEventListener('change', onGridSizeChange);

    randomizeBtn.addEventListener('click', () => {
        randomizePositions();
        syncReplayGrid();
    });
    replayBtn.addEventListener('click', playFinalGame);

    trainBtn.addEventListener('click', () => {
        const w = parseInt(widthInput.value);
        const h = parseInt(heightInput.value);
        const epochs = parseInt(epochsInput.value);

        let playerPos = '', goalPos = '', pitPos = '', wallPos = '';
        for (let r = 0; r < h; r++) {
            for (let c = 0; c < w; c++) {
                const cell = document.getElementById(`${gridContainer.id}-cell-${r}-${c}`);
                if (cell.classList.contains('player')) playerPos = `${r},${c}`;
                if (cell.classList.contains('goal')) goalPos = `${r},${c}`;
                if (cell.classList.contains('pit')) pitPos = `${r},${c}`;
                if (cell.classList.contains('wall')) wallPos = `${r},${c}`;
            }
        }
        
        syncReplayGrid();
        initChart();

        trainBtn.disabled = true;
        randomizeBtn.disabled = true;
        trainBtn.innerText = 'Training...';
        trainStatusEl.innerText = 'Training in progress...';
        trainStatusEl.className = 'status running badge';
        replayBtn.disabled = true;
        replayStatusEl.innerText = 'Awaiting Training...';
        replayStatusEl.className = 'status badge';
        
        const mode = modeSelect.value;
        if (mode === 'compare') {
            trainStatusDb.innerText = 'Training...';
            trainStatusDb.className = 'status running badge';
            trainStatusDu.innerText = 'Training...';
            trainStatusDu.className = 'status running badge';
        }
        
        finalGame = null;
        finalGameDb = null;
        finalGameDu = null;
        capturedRoutes = [];

        if (finalRouteSvgDb) finalRouteSvgDb.innerHTML = '';
        if (finalRouteSvgDu) finalRouteSvgDu.innerHTML = '';

        const gradClipNode = document.getElementById('grad-clip');
        const gradClip = gradClipNode ? gradClipNode.value : '1.0';
        
        const lrStatus = document.getElementById('lr-status');
        if (lrStatus && mode === 'lightning_random') {
            lrStatus.style.display = 'inline-block';
            lrStatus.innerText = 'LR: --';
        }

        const source = new EventSource(`/api/stream_train?mode=${mode}&width=${w}&height=${h}&epochs=${epochs}&player=${playerPos}&goal=${goalPos}&pit=${pitPos}&wall=${wallPos}&grad_clip=${gradClip}`);
        
        source.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            if (data.type === 'progress') {
                if (mode === 'compare') {
                    updateChartCompare(data.epoch, data.loss_double, data.loss_dueling);
                } else {
                    updateChart(data.epoch, data.loss);
                    if (data.lr !== undefined && document.getElementById('lr-status')) {
                        document.getElementById('lr-status').innerText = `LR: ${data.lr.toExponential(2)}`;
                    }
                }
            } else if (data.type === 'complete') {
                source.close();
                
                trainBtn.disabled = false;
                randomizeBtn.disabled = false;
                trainBtn.innerText = 'Start Training';
                trainStatusEl.innerText = 'Training Complete';
                trainStatusEl.className = 'status success badge';
                
                if (mode === 'compare') {
                    finalGameDb = data.final_game_double;
                    finalGameDu = data.final_game_dueling;
                    trainStatusDb.innerText = 'Complete';
                    trainStatusDb.className = 'status success badge';
                    trainStatusDu.innerText = 'Complete';
                    trainStatusDu.className = 'status success badge';
                    
                    showFinalRoutesCompare();
                } else {
                    capturedRoutes = data.routes;
                    finalGame = data.final_game;
                    
                    if (mode === 'lightning_random') {
                        document.getElementById('interactive-validation-panel').style.display = 'block';
                        const verifyBtn = document.getElementById('verify-btn');
                        if (verifyBtn) verifyBtn.disabled = false;
                        if (typeof initValidationGrid === 'function') initValidationGrid(w, h);
                        
                        setTimeout(() => {
                            document.getElementById('interactive-validation-panel').scrollIntoView({behavior: 'smooth'});
                        }, 100);
                    } else {
                        replayBtn.disabled = false;
                        replayStatusEl.innerText = 'Ready';
                        buildRouteSelectors();
                        playFinalGame();
                    }
                }
            }
        };
        
        source.onerror = function() {
            source.close();
            console.log("SSE connection closed.");
        };
    });
    
    // Validation Grid Logic
    let draggedElement = null;
    function initValidationGrid(w, h) {
        const container = document.getElementById('validation-grid-container');
        if (!container) return;
        
        container.innerHTML = '';
        container.style.gridTemplateColumns = `repeat(${w}, 60px)`;
        document.getElementById('validation-route-svg').innerHTML = '';
        
        const numObstacles = Math.min(w, h) - 1;
        const numPits = Math.max(1, Math.floor(Math.random() * numObstacles));
        let numWalls = numObstacles - numPits;
        if (numWalls < 1 && numObstacles > 1) { numWalls = 1; }
        else if (numObstacles <= 1) { numWalls = 1; } // fallback
        
        let pieces = ['player', 'goal'];
        for(let i=0; i<numPits; i++) pieces.push('pit');
        for(let i=0; i<numWalls; i++) pieces.push('wall');
        while(pieces.length < w * h) pieces.push('');
        
        pieces.sort(() => Math.random() - 0.5);
        
        for (let r = 0; r < h; r++) {
            for (let c = 0; c < w; c++) {
                const cell = document.createElement('div');
                cell.className = 'cell';
                cell.id = `val-cell-${r}-${c}`;
                
                const piece = pieces.shift();
                if (piece) {
                    cell.classList.add(piece);
                    if (piece === 'player') cell.innerText = 'P';
                    if (piece === 'goal') cell.innerText = '+';
                    if (piece === 'pit') cell.innerText = '-';
                    if (piece === 'wall') cell.innerText = 'W';
                    cell.draggable = true;
                } else {
                    cell.draggable = false;
                }
                
                cell.addEventListener('dragstart', function(e) {
                    draggedElement = this;
                    e.dataTransfer.effectAllowed = 'move';
                });
                
                cell.addEventListener('dragover', function(e) {
                    e.preventDefault();
                    e.dataTransfer.dropEffect = 'move';
                });
                
                cell.addEventListener('drop', function(e) {
                    e.preventDefault();
                    if (this !== draggedElement) {
                        const tempClass = draggedElement.className;
                        const tempText = draggedElement.innerText;
                        const tempDraggable = draggedElement.draggable;
                        
                        draggedElement.className = this.className;
                        draggedElement.innerText = this.innerText;
                        draggedElement.draggable = this.draggable;
                        
                        this.className = tempClass;
                        this.innerText = tempText;
                        this.draggable = tempDraggable;
                        
                        document.getElementById('validation-route-svg').innerHTML = '';
                        document.getElementById('validation-status').innerText = 'Ready to Verify';
                        document.getElementById('validation-status').className = 'status badge';
                    }
                });
                
                container.appendChild(cell);
            }
        }
    }
    
    const verifyBtn = document.getElementById('verify-btn');
    if (verifyBtn) {
        verifyBtn.addEventListener('click', async () => {
            const w = parseInt(widthInput.value);
            const h = parseInt(heightInput.value);
            let custom_positions = {};
            
            for (let r = 0; r < h; r++) {
                for (let c = 0; c < w; c++) {
                    const cell = document.getElementById(`val-cell-${r}-${c}`);
                    if (!cell) continue;
                    if (cell.classList.contains('player')) custom_positions['Player'] = [r, c];
                    if (cell.classList.contains('goal'))   custom_positions['Goal']   = [r, c];
                    if (cell.classList.contains('pit'))  {
                        if (!custom_positions['Pit']) custom_positions['Pit'] = [];
                        custom_positions['Pit'].push([r, c]);
                    }
                    if (cell.classList.contains('wall')) {
                        if (!custom_positions['Wall']) custom_positions['Wall'] = [];
                        custom_positions['Wall'].push([r, c]);
                    }
                }
            }
            
            const validationStatus = document.getElementById('validation-status');
            validationStatus.innerText = 'Verifying...';
            validationStatus.className = 'status running badge';
            verifyBtn.disabled = true;
            
            try {
                const response = await fetch('/api/verify_random', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({width: w, height: h, custom_positions: custom_positions})
                });
                const data = await response.json();
                
                if (data.status === 'success') {
                    drawStaticRoute(data.route, document.getElementById('validation-route-svg'), '#10b981');
                    validationStatus.innerText = 'Verification Complete';
                    validationStatus.className = 'status success badge';
                } else {
                    validationStatus.innerText = 'Error';
                    validationStatus.className = 'status badge';
                    console.error(data.message);
                }
            } catch (err) {
                console.error(err);
            } finally {
                verifyBtn.disabled = false;
            }
        });
    }

});
