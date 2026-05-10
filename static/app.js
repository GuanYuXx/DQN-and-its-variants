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
    
    let lossChart = null;
    let finalGame = null;
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
    }

    function updateChart(epoch, loss) {
        lossChart.data.labels.push(epoch);
        lossChart.data.datasets[0].data.push(loss);
        lossChart.update();
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

    function setupGrids() {
        const w = parseInt(widthInput.value);
        const h = parseInt(heightInput.value);
        initGrid(gridContainer, w, h, true);
        initGrid(replayGridContainer, w, h, false);
        updatePlayerPos(gridContainer, 0, w - 1);
        
        // Copy layout from top grid to replay grid
        syncReplayGrid();
        
        routeSvg.innerHTML = '';
        routeSelectors.innerHTML = '<p class="placeholder-text">Routes will appear here after training.</p>';
    }
    
    function syncReplayGrid() {
        const cells = gridContainer.querySelectorAll('.cell');
        const replayCells = replayGridContainer.querySelectorAll('.cell');
        
        cells.forEach((cell, i) => {
            replayCells[i].className = cell.className;
            replayCells[i].innerText = cell.innerText;
        });
    }

    // Init
    setupGrids();
    initChart();

    widthInput.addEventListener('change', setupGrids);
    heightInput.addEventListener('change', setupGrids);
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
        
        finalGame = null;
        capturedRoutes = [];

        const source = new EventSource(`/api/stream_train?width=${w}&height=${h}&epochs=${epochs}&player=${playerPos}&goal=${goalPos}&pit=${pitPos}&wall=${wallPos}`);
        
        source.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            if (data.type === 'progress') {
                updateChart(data.epoch, data.loss);
            } else if (data.type === 'complete') {
                source.close();
                capturedRoutes = data.routes;
                finalGame = data.final_game;
                
                trainBtn.disabled = false;
                randomizeBtn.disabled = false;
                trainBtn.innerText = 'Start Training';
                trainStatusEl.innerText = 'Training Complete';
                trainStatusEl.className = 'status success badge';
                
                replayBtn.disabled = false;
                replayStatusEl.innerText = 'Ready';
                
                buildRouteSelectors();
                
                playFinalGame();
            }
        };
        
        source.onerror = function() {
            source.close();
            console.log("SSE connection closed.");
        };
    });
});
