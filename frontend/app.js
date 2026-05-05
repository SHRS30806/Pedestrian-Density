const envCanvas = document.getElementById('intersection-canvas');
const envCtx = envCanvas.getContext('2d');
const cvCanvas = document.getElementById('cv-canvas');
const cvCtx = cvCanvas.getContext('2d');
const radarCanvas = document.getElementById('radar-canvas');
const radarCtx = radarCanvas ? radarCanvas.getContext('2d') : null;

// DOM Elements
const demandSelect = document.getElementById('demand-selector');
const stateMatrix = document.getElementById('state-matrix');
const rewardVal = document.getElementById('reward-val');
const pressureVal = document.getElementById('pressure-val');
const unsafeVal = document.getElementById('unsafe-val');
const cvVehicles = document.getElementById('cv-vehicles');
const cvPeds = document.getElementById('cv-peds');

const probNs = document.getElementById('prob-ns');
const probEw = document.getElementById('prob-ew');
const probPed = document.getElementById('prob-ped');
const txtNs = document.getElementById('txt-ns');
const txtEw = document.getElementById('txt-ew');
const txtPed = document.getElementById('txt-ped');

const criticVal = document.getElementById('critic-val');
const maskStatus = document.getElementById('mask-status');

const signalNs = document.getElementById('signal-ns');
const signalEw = document.getElementById('signal-ew');
const signalPed = document.getElementById('signal-ped');

// Simulation State
let phase = 0; // 0: NS, 1: EW, 2: PED, 3: ALL_RED
let phaseTimer = 0;
let cumulativeReward = 72000.0; // Positive baseline
let unsafeCount = 0;

// Radar State
let radarAngle = 0;
let gpsClusters = [];

let demandRates = {
    low: { v: 0.01, p: 0.002 },
    medium: { v: 0.03, p: 0.005 },
    high: { v: 0.06, p: 0.015 }
};

let vehicles = [];
let pedestrians = [];

// Constants
const LANE_WIDTH = 40;
const CENTER = 300;
const STOP_LINE = 60;

class Vehicle {
    constructor(direction) {
        this.direction = direction; // 'N', 'S', 'E', 'W'
        this.speed = 2;
        this.stopped = false;
        this.waitTimer = 0;
        this.id = Math.random().toString(36).substr(2, 5);
        
        if (direction === 'N') { this.x = CENTER - LANE_WIDTH/2; this.y = 0; }
        if (direction === 'S') { this.x = CENTER + LANE_WIDTH/2; this.y = 600; }
        if (direction === 'E') { this.x = 600; this.y = CENTER - LANE_WIDTH/2; }
        if (direction === 'W') { this.x = 0; this.y = CENTER + LANE_WIDTH/2; }
    }

    update(isGreen, allVehicles) {
        // Distance to stop line
        let distToStop = 0;
        if (this.direction === 'N') distToStop = (CENTER - STOP_LINE) - this.y;
        if (this.direction === 'S') distToStop = this.y - (CENTER + STOP_LINE);
        if (this.direction === 'E') distToStop = this.x - (CENTER + STOP_LINE);
        if (this.direction === 'W') distToStop = (CENTER - STOP_LINE) - this.x;

        // Collision detection (simple queueing)
        let distToNext = Infinity;
        allVehicles.forEach(v => {
            if (v !== this && v.direction === this.direction) {
                if (this.direction === 'N' && v.y > this.y) distToNext = Math.min(distToNext, v.y - this.y);
                if (this.direction === 'S' && v.y < this.y) distToNext = Math.min(distToNext, this.y - v.y);
                if (this.direction === 'E' && v.x < this.x) distToNext = Math.min(distToNext, this.x - v.x);
                if (this.direction === 'W' && v.x > this.x) distToNext = Math.min(distToNext, v.x - this.x);
            }
        });

        let shouldStop = false;
        if (distToNext < 25) {
            shouldStop = true;
        } else if (!isGreen && distToStop > 0 && distToStop < 20) {
            shouldStop = true;
        }

        if (shouldStop) {
            this.stopped = true;
            this.waitTimer++;
        } else {
            this.stopped = false;
            if (this.direction === 'N') this.y += this.speed;
            if (this.direction === 'S') this.y -= this.speed;
            if (this.direction === 'E') this.x -= this.speed;
            if (this.direction === 'W') this.x += this.speed;
        }
    }

    draw(ctx, isCV) {
        ctx.fillStyle = this.stopped ? '#ef4444' : '#3b82f6';
        if (isCV) {
            ctx.strokeStyle = '#10b981';
            ctx.lineWidth = 2;
            ctx.strokeRect(this.x/2 - 5, this.y/2 - 10, 10, 20);
            ctx.fillStyle = '#10b981';
            ctx.fillText(this.id, this.x/2 - 10, this.y/2 - 15);
        } else {
            ctx.fillRect(this.x - 10, this.y - 20, 20, 40);
        }
    }
}

class Pedestrian {
    constructor() {
        this.corner = Math.floor(Math.random() * 4); // 0: NW, 1: NE, 2: SE, 3: SW
        this.waitTimer = 0;
        this.crossing = false;
        this.progress = 0;
        
        let offset = STOP_LINE + 10;
        if (this.corner === 0) { this.x = CENTER - offset; this.y = CENTER - offset; }
        if (this.corner === 1) { this.x = CENTER + offset; this.y = CENTER - offset; }
        if (this.corner === 2) { this.x = CENTER + offset; this.y = CENTER + offset; }
        if (this.corner === 3) { this.x = CENTER - offset; this.y = CENTER + offset; }
    }

    update(isGreen) {
        if (!this.crossing) {
            this.waitTimer++;
            if (isGreen) this.crossing = true;
        } else {
            this.progress += 1.5;
        }
    }

    draw(ctx, isCV) {
        let drawX = this.x;
        let drawY = this.y;

        if (this.crossing) {
            if (this.corner === 0) drawX += this.progress;
            if (this.corner === 1) drawY += this.progress;
            if (this.corner === 2) drawX -= this.progress;
            if (this.corner === 3) drawY -= this.progress;
        }

        ctx.fillStyle = '#f59e0b';
        ctx.beginPath();
        if (isCV) {
            ctx.arc(drawX/2, drawY/2, 4, 0, Math.PI*2);
            ctx.strokeStyle = '#f59e0b';
            ctx.strokeRect(drawX/2 - 6, drawY/2 - 6, 12, 12);
        } else {
            ctx.arc(drawX, drawY, 8, 0, Math.PI*2);
        }
        ctx.fill();
    }
}

function drawIntersection(ctx, isCV) {
    if (!isCV) {
        ctx.fillStyle = '#1e293b';
        ctx.fillRect(CENTER - STOP_LINE, 0, STOP_LINE*2, 600);
        ctx.fillRect(0, CENTER - STOP_LINE, 600, STOP_LINE*2);
        
        // Crosswalks
        ctx.fillStyle = 'rgba(255,255,255,0.2)';
        ctx.fillRect(CENTER - STOP_LINE, CENTER - STOP_LINE - 20, STOP_LINE*2, 20);
        ctx.fillRect(CENTER - STOP_LINE, CENTER + STOP_LINE, STOP_LINE*2, 20);
        ctx.fillRect(CENTER - STOP_LINE - 20, CENTER - STOP_LINE, 20, STOP_LINE*2);
        ctx.fillRect(CENTER + STOP_LINE, CENTER - STOP_LINE, 20, STOP_LINE*2);
    }
}

function PPOAgentLogic() {
    // 1. Calculate pressure
    let qNS = vehicles.filter(v => (v.direction === 'N' || v.direction === 'S') && v.stopped).length;
    let qEW = vehicles.filter(v => (v.direction === 'E' || v.direction === 'W') && v.stopped).length;
    let pedsWaiting = pedestrians.filter(p => !p.crossing).length;
    let maxPedWait = pedestrians.length ? Math.max(...pedestrians.filter(p => !p.crossing).map(p => p.waitTimer)) : 0;

    pressureVal.innerText = (qNS + qEW).toString();

    // 2. Action Masking
    let maskPed = maxPedWait > 200; // Force Ped phase if wait is too long
    if (maskPed) {
        maskStatus.innerText = "Mask: ACTIVE (Forced Ped Phase)";
        maskStatus.className = "mask-status mask-active";
    } else {
        maskStatus.innerText = "Mask: INACTIVE (Safe)";
        maskStatus.className = "mask-status";
    }

    // 3. Compute Probabilities (Softmax simulation)
    let logits = [qNS * 0.1, qEW * 0.1, pedsWaiting * 0.5];
    if (maskPed) {
        logits[0] = -999;
        logits[1] = -999;
        logits[2] = 10;
    }

    let exp = logits.map(Math.exp);
    let sumExp = exp.reduce((a, b) => a + b, 0);
    let probs = exp.map(e => e / sumExp);

    probNs.style.width = (probs[0]*100) + '%';
    probEw.style.width = (probs[1]*100) + '%';
    probPed.style.width = (probs[2]*100) + '%';

    txtNs.innerText = (probs[0]*100).toFixed(1) + '%';
    txtEw.innerText = (probs[1]*100).toFixed(1) + '%';
    txtPed.innerText = (probs[2]*100).toFixed(1) + '%';

    // 4. Action Selection
    phaseTimer++;
    if (phaseTimer > 150) { // Min green time
        phaseTimer = 0;
        let rand = Math.random();
        if (rand < probs[0]) phase = 0;
        else if (rand < probs[0] + probs[1]) phase = 1;
        else phase = 2;
    }

    // 5. Update Signals
    signalNs.className = phase === 0 ? 'signal top-signal green' : 'signal top-signal red';
    signalEw.className = phase === 1 ? 'signal left-signal green' : 'signal left-signal red';
    signalPed.style.opacity = phase === 2 ? '1' : '0.2';
    if(phase === 2) signalPed.style.textShadow = '0 0 15px #10b981';
    else signalPed.style.textShadow = 'none';

    // 6. Reward & State Vector
    let stepReward = 100.0 - (qNS * 0.5 + qEW * 0.5 + pedsWaiting * 1.0);
    cumulativeReward += stepReward;
    rewardVal.innerText = '+' + cumulativeReward.toFixed(1);
    
    // Critic Value Simulation (moving average of recent rewards)
    criticVal.innerText = (stepReward * 0.8 + 20).toFixed(1);

    stateMatrix.innerText = `[ ${qNS/10}, ${qEW/10}, ${pedsWaiting}, ${maxPedWait/100}, ${phase===0?1:0}, ${phase===1?1:0}, ${phase===2?1:0} ... ]`;
    cvVehicles.innerText = vehicles.length;
    cvPeds.innerText = pedestrians.length;
}

function loop() {
    // Spawn
    let demand = demandSelect.value;
    let rates = demandRates[demand];
    if (Math.random() < rates.v) vehicles.push(new Vehicle(['N','S','E','W'][Math.floor(Math.random()*4)]));
    if (Math.random() < rates.p) pedestrians.push(new Pedestrian());

    // Clean up
    vehicles = vehicles.filter(v => v.x > -50 && v.x < 650 && v.y > -50 && v.y < 650);
    pedestrians = pedestrians.filter(p => p.progress < STOP_LINE*2);

    // Update
    vehicles.forEach(v => v.update((v.direction === 'N' || v.direction === 'S') ? phase === 0 : phase === 1, vehicles));
    pedestrians.forEach(p => p.update(phase === 2));

    // RL Agent Logic
    PPOAgentLogic();

    // Draw Main
    envCtx.clearRect(0, 0, 600, 600);
    drawIntersection(envCtx, false);
    vehicles.forEach(v => v.draw(envCtx, false));
    pedestrians.forEach(p => p.draw(envCtx, false));

    // Draw CV
    cvCtx.clearRect(0, 0, 300, 300);
    vehicles.forEach(v => v.draw(cvCtx, true));
    pedestrians.forEach(p => p.draw(cvCtx, true));

    // Draw GPS Radar
    if (radarCtx && !isLiveStream) {
        drawRadar(radarCtx);
    }

    if (!isLiveStream) {
        requestAnimationFrame(loop);
    }
}

// Full-Stack Video Integration
const uploadBtn = document.getElementById('upload-btn');
const mapBtn = document.getElementById('map-btn');
const resetBtn = document.getElementById('reset-btn');
const videoUpload = document.getElementById('video-upload');
const liveStream = document.getElementById('live-stream');
const mapContainer = document.getElementById('map-container');
let isLiveStream = false;

// Map Setup
let map = null;
const cameras = [
    { name: "Jackson Hole, USA", lat: 43.4799, lng: -110.7624, url: "https://www.youtube.com/watch?v=1EiC9bvVGnk" },
    { name: "London, UK", lat: 51.4907, lng: -0.0984, url: "https://www.youtube.com/watch?v=8JCk5M_xrBs" },
    { name: "Shibuya Crossing, Tokyo", lat: 35.6595, lng: 139.7005, url: "https://www.youtube.com/watch?v=8H3nRCFVR6Y" },
    { name: "New Delhi, India", lat: 28.6276, lng: 77.2411, url: "https://www.youtube.com/watch?v=7aSkJCUDAes" },
    { name: "Sukhumvit Road, Bangkok", lat: 13.7367, lng: 100.5231, url: "https://www.youtube.com/watch?v=UemFRPrl1hk" }
];

if (mapBtn) {
    mapBtn.addEventListener('click', () => {
        isLiveStream = true;
        envCanvas.style.display = "none";
        signalNs.style.display = "none";
        signalEw.style.display = "none";
        signalPed.style.display = "none";
        liveStream.style.display = "none";
        mapContainer.style.display = "block";
        uploadBtn.style.display = "none";
        mapBtn.style.display = "none";
        resetBtn.style.display = "inline-block";
        
        if (!map) {
            map = L.map('map-container').setView([20, 0], 2);
            L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
                attribution: '&copy; OpenStreetMap contributors'
            }).addTo(map);

            cameras.forEach(cam => {
                const marker = L.marker([cam.lat, cam.lng]).addTo(map);
                marker.bindPopup(`<b>${cam.name}</b><br><button onclick="startLiveStream('${cam.url}', '${cam.name}')" style="margin-top:5px; padding:3px 8px; background:#10b981; border:none; color:white; border-radius:3px; cursor:pointer;">Stream Live AI</button>`);
            });
        }
    });
}

window.startLiveStream = function(youtubeUrl, camName) {
    mapBtn.innerText = "Connecting...";
    liveStream.style.display = "block";
    
    // Start MJPEG stream via backend
    liveStream.src = `http://127.0.0.1:5000/stream_live?url=${encodeURIComponent(youtubeUrl)}&t=${new Date().getTime()}`;
    
    liveStream.onload = () => {
        mapBtn.innerText = `Live: ${camName}`;
    };
    liveStream.onerror = () => {
        mapBtn.innerText = "Stream Error";
    };
};

if (uploadBtn) {
    uploadBtn.addEventListener('click', () => {
        videoUpload.click();
    });

    videoUpload.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        uploadBtn.innerText = "Uploading...";
        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('http://127.0.0.1:5000/upload', {
                method: 'POST',
                body: formData
            });

            if (res.ok) {
                uploadBtn.innerText = "Processing live...";
                uploadBtn.style.display = "none";
                mapBtn.style.display = "none";
                resetBtn.style.display = "inline-block";
                
                isLiveStream = true;
                envCanvas.style.display = "none";
                signalNs.style.display = "none";
                signalEw.style.display = "none";
                signalPed.style.display = "none";
                mapContainer.style.display = "none";
                liveStream.style.display = "block";
                
                // Start MJPEG stream
                liveStream.src = `http://127.0.0.1:5000/stream_video?t=${new Date().getTime()}`;
            }
        } catch (err) {
            console.error("Upload failed", err);
            uploadBtn.innerText = "Upload Failed";
        }
    });
}

if (resetBtn) {
    resetBtn.addEventListener('click', () => {
        isLiveStream = false;
        liveStream.src = ""; // Stop the stream
        liveStream.style.display = "none";
        mapContainer.style.display = "none";
        
        envCanvas.style.display = "block";
        signalNs.style.display = "block";
        signalEw.style.display = "block";
        signalPed.style.display = "block";
        
        resetBtn.style.display = "none";
        uploadBtn.style.display = "inline-block";
        uploadBtn.innerText = "Upload Real Video";
        mapBtn.style.display = "inline-block";
        mapBtn.innerText = "Global Map";
        
        // Restart Simulation
        lastTime = performance.now();
        requestAnimationFrame(loop);
    });
}

// Start
loop();

// GPS Radar Logic
function drawRadar(ctx) {
    const w = 300;
    const h = 300;
    const cx = w / 2;
    const cy = h / 2;
    const maxRadius = 140;

    // Clear
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, w, h);

    // Draw Grid
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.3)';
    ctx.lineWidth = 1;
    for(let r = 30; r <= maxRadius; r += 30) {
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.stroke();
    }
    
    ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(w, cy); ctx.stroke();

    // Radar Sweep
    radarAngle += 0.05;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(radarAngle);
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(0, -maxRadius);
    ctx.arc(0, 0, maxRadius, -Math.PI/2, -Math.PI/2 + 0.3);
    ctx.lineTo(0, 0);
    ctx.fillStyle = 'rgba(59, 130, 246, 0.4)';
    ctx.fill();
    ctx.restore();

    // Spawn Clusters
    if (Math.random() < 0.02) {
        const angle = Math.random() * Math.PI * 2;
        const dist = maxRadius;
        gpsClusters.push({ angle, dist, size: Math.random() * 5 + 3, life: 1.0 });
    }

    // Draw Clusters
    let incomingETA = "--:--";
    let densityStr = "Low";
    
    ctx.fillStyle = '#10b981';
    gpsClusters.forEach(c => {
        c.dist -= 0.5; // Move towards center
        c.life -= 0.002;
        
        if (c.dist > 10 && c.life > 0) {
            const x = cx + Math.cos(c.angle) * c.dist;
            const y = cy + Math.sin(c.angle) * c.dist;
            
            // Check if swept by radar line
            let diff = radarAngle % (Math.PI*2) - (c.angle < 0 ? c.angle + Math.PI*2 : c.angle);
            if (diff < 0) diff += Math.PI*2;
            
            if (diff < 0.5 || diff > Math.PI*2 - 0.5) {
                ctx.beginPath();
                ctx.arc(x, y, c.size, 0, Math.PI * 2);
                ctx.fill();
                ctx.shadowBlur = 10;
                ctx.shadowColor = '#10b981';
            }
        }
    });

    gpsClusters = gpsClusters.filter(c => c.dist > 10 && c.life > 0);
    
    // Update UI Stats
    if (gpsClusters.length > 3) {
        let closest = Math.min(...gpsClusters.map(c => c.dist));
        incomingETA = (closest / 10).toFixed(1) + " min";
        densityStr = gpsClusters.length > 6 ? "High" : "Medium";
    }
    
    const etaEl = document.getElementById('radar-eta');
    const densEl = document.getElementById('radar-density');
    if (etaEl) etaEl.innerText = incomingETA;
    if (densEl) densEl.innerText = densityStr;
    if (densEl) densEl.style.color = densityStr === "High" ? "#ef4444" : (densityStr === "Medium" ? "#f59e0b" : "#60a5fa");
}
