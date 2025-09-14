/**
 * Behavioral Analyzer Dashboard JavaScript
 * 
 * Handles real-time updates, WebSocket communication, and UI interactions
 */

class BehavioralDashboard {
    constructor() {
        this.socket = null;
        this.charts = {};
        this.dataHistory = {
            blinkRate: [],
            emotionScores: {},
            objectCounts: {},
            timestamps: []
        };
        this.sessionStartTime = Date.now();
        this.lastUpdateTime = Date.now();
        
        this.init();
    }
    
    init() {
        this.initSocket();
        this.initCharts();
        this.initControls();
        this.startSessionTimer();
        this.requestInitialData();
    }
    
    initSocket() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.updateConnectionStatus(true);
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.updateConnectionStatus(false);
        });
        
        this.socket.on('data_update', (data) => {
            this.updateDashboard(data);
        });
        
        this.socket.on('video_frame', (data) => {
            this.updateVideoFeed(data.frame);
        });
        
        this.socket.on('status', (data) => {
            console.log('Status:', data.message);
        });
    }
    
    initCharts() {
        // Emotion Chart
        const emotionCtx = document.getElementById('emotionChart').getContext('2d');
        this.charts.emotion = new Chart(emotionCtx, {
            type: 'doughnut',
            data: {
                labels: ['Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 'Neutral'],
                datasets: [{
                    data: [0, 0, 0, 0, 0, 100],
                    backgroundColor: [
                        '#28a745', '#6f42c1', '#dc3545', 
                        '#fd7e14', '#ffc107', '#6c757d'
                    ],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
        
        // Blink Rate Chart
        const blinkCtx = document.getElementById('blinkChart').getContext('2d');
        this.charts.blink = new Chart(blinkCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Blink Rate (bpm)',
                    data: [],
                    borderColor: '#007bff',
                    backgroundColor: 'rgba(0, 123, 255, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 30
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
        
        // Object Detection Chart
        const objectCtx = document.getElementById('objectChart').getContext('2d');
        this.charts.objects = new Chart(objectCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Object Count',
                    data: [],
                    backgroundColor: '#ffc107',
                    borderColor: '#ffc107',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
    
    initControls() {
        // Debug Mode Toggle
        document.getElementById('debug-mode').addEventListener('change', (e) => {
            this.socket.emit('update_controls', { debug_mode: e.target.checked });
        });
        
        // Show Landmarks Toggle
        document.getElementById('show-landmarks').addEventListener('change', (e) => {
            this.socket.emit('update_controls', { show_landmarks: e.target.checked });
        });
        
        // Show Objects Toggle
        document.getElementById('show-objects').addEventListener('change', (e) => {
            this.socket.emit('update_controls', { show_objects: e.target.checked });
        });
        
        // Video Stream Toggle
        document.getElementById('video-stream').addEventListener('change', (e) => {
            this.socket.emit('toggle_video_stream', { enabled: e.target.checked });
        });
    }
    
    startSessionTimer() {
        setInterval(() => {
            const elapsed = Date.now() - this.sessionStartTime;
            const hours = Math.floor(elapsed / 3600000);
            const minutes = Math.floor((elapsed % 3600000) / 60000);
            const seconds = Math.floor((elapsed % 60000) / 1000);
            
            document.getElementById('session-time').textContent = 
                `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }, 1000);
    }
    
    requestInitialData() {
        this.socket.emit('request_data');
    }
    
    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connection-status');
        if (connected) {
            statusElement.className = 'badge bg-success me-2';
            statusElement.innerHTML = '<i class="fas fa-circle"></i> Connected';
        } else {
            statusElement.className = 'badge bg-danger me-2';
            statusElement.innerHTML = '<i class="fas fa-circle"></i> Disconnected';
        }
    }
    
    updateDashboard(data) {
        this.lastUpdateTime = Date.now();
        
        // Update video analysis data
        if (data.video) {
            this.updateVideoAnalysis(data.video);
            this.updatePersonTracking(data.video);
        }
        
        // Update audio analysis data
        if (data.audio) {
            this.updateAudioAnalysis(data.audio);
        }
        
        // Update object detection data
        if (data.objects) {
            this.updateObjectDetection(data.objects);
        }
        
        // Update session statistics
        if (data.session_stats) {
            this.updateSessionStats(data.session_stats);
        }
        
        // Update charts
        this.updateCharts(data);
    }
    
    updateVideoAnalysis(videoData) {
        // Update emotion
        const emotionElement = document.getElementById('current-emotion');
        emotionElement.textContent = videoData.emotion || 'Unknown';
        emotionElement.className = `text-primary emotion-${videoData.emotion?.toLowerCase() || 'neutral'}`;
        
        // Update attention
        const attentionElement = document.getElementById('attention-state');
        attentionElement.textContent = videoData.attention_state || 'Unknown';
        attentionElement.className = `text-warning attention-${videoData.attention_state?.toLowerCase().replace(' ', '-') || 'unknown'}`;
        
        // Update posture
        const postureElement = document.getElementById('posture-state');
        postureElement.textContent = videoData.posture_state || 'Unknown';
        postureElement.className = `text-info posture-${videoData.posture_state?.toLowerCase() || 'unknown'}`;
        
        // Update blink analysis
        document.getElementById('blink-rate').textContent = Math.round(videoData.blink_rate || 0);
        document.getElementById('total-blinks').textContent = videoData.total_blinks || 0;
        
        // Update fatigue level
        const fatigueLevel = videoData.fatigue_level || 'Normal';
        document.getElementById('fatigue-level').textContent = fatigueLevel;
        
        const fatigueBar = document.getElementById('fatigue-bar');
        const fatiguePercentage = this.getFatiguePercentage(fatigueLevel);
        fatigueBar.style.width = `${fatiguePercentage}%`;
        fatigueBar.className = `progress-bar ${this.getFatigueColor(fatigueLevel)}`;
        
        // Update FPS
        document.getElementById('video-fps').textContent = Math.round(videoData.fps || 0);
        
        // Store data for charts
        this.dataHistory.blinkRate.push(videoData.blink_rate || 0);
        this.dataHistory.timestamps.push(new Date().toLocaleTimeString());
        
        // Keep only last 20 data points
        if (this.dataHistory.blinkRate.length > 20) {
            this.dataHistory.blinkRate.shift();
            this.dataHistory.timestamps.shift();
        }
    }
    
    updateAudioAnalysis(audioData) {
        // Update transcription
        const transcriptionElement = document.getElementById('transcription');
        if (audioData.transcription && audioData.transcription.trim()) {
            transcriptionElement.innerHTML = `<strong>${audioData.transcription}</strong>`;
        } else {
            transcriptionElement.innerHTML = '<em class="text-muted">No speech detected</em>';
        }
        
        // Update audio emotion
        document.getElementById('audio-emotion').textContent = audioData.emotion || 'neutral';
        
        // Update sentiment
        const sentimentScore = audioData.sentiment || 0;
        document.getElementById('sentiment-score').textContent = sentimentScore.toFixed(2);
        document.getElementById('sentiment-score').className = 
            `text-${sentimentScore > 0 ? 'success' : sentimentScore < 0 ? 'danger' : 'secondary'}`;
        
        // Update word count
        if (audioData.session_stats && audioData.session_stats.total_words) {
            document.getElementById('total-words').textContent = audioData.session_stats.total_words;
        }
    }
    
    updateObjectDetection(objectData) {
        const objectCount = objectData.detections ? objectData.detections.length : 0;
        document.getElementById('object-count').textContent = objectCount;
        
        // Update object list
        const objectListElement = document.getElementById('object-list');
        if (objectData.detections && objectData.detections.length > 0) {
            const objectCounts = {};
            objectData.detections.forEach(detection => {
                const className = detection.class_name;
                objectCounts[className] = (objectCounts[className] || 0) + 1;
            });
            
            objectListElement.innerHTML = Object.entries(objectCounts)
                .map(([name, count]) => `
                    <div class="object-item">
                        <span class="object-name">${name}</span>
                        <span class="object-confidence">${count}</span>
                    </div>
                `).join('');
        } else {
            objectListElement.innerHTML = '<em class="text-muted">No objects detected</em>';
        }
        
        // Update object counts for chart
        if (objectData.object_counts) {
            this.dataHistory.objectCounts = objectData.object_counts;
        }
    }
    
    updatePersonTracking(videoData) {
        const personTracking = videoData.person_tracking || {};
        const mainPerson = videoData.main_person;
        
        // Update person count
        document.getElementById('person-count').textContent = personTracking.total_persons || 0;
        
        // Update main person status
        const mainPersonStatus = document.getElementById('main-person-status');
        if (personTracking.has_main_person) {
            mainPersonStatus.textContent = '✓';
            mainPersonStatus.className = 'text-success';
        } else {
            mainPersonStatus.textContent = '✗';
            mainPersonStatus.className = 'text-danger';
        }
        
        // Update main person details
        if (mainPerson) {
            document.getElementById('main-person-id').textContent = mainPerson.id ? mainPerson.id.substring(0, 8) + '...' : 'None';
            document.getElementById('main-person-confidence').textContent = (personTracking.main_person_confidence || 0).toFixed(2);
            document.getElementById('main-person-size').textContent = (mainPerson.size_ratio || 0).toFixed(3);
        } else {
            document.getElementById('main-person-id').textContent = 'None';
            document.getElementById('main-person-confidence').textContent = '0.00';
            document.getElementById('main-person-size').textContent = '0.000';
        }
        
        // Update person list
        const personListElement = document.getElementById('person-list');
        if (personTracking.person_detections && personTracking.person_detections.length > 0) {
            personListElement.innerHTML = personTracking.person_detections
                .map((person, index) => `
                    <div class="object-item">
                        <span class="object-name">Person ${index + 1}</span>
                        <span class="object-confidence">${(person.confidence * 100).toFixed(0)}%</span>
                    </div>
                `).join('');
        } else {
            personListElement.innerHTML = '<em class="text-muted">No people detected</em>';
        }
    }
    
    updateSessionStats(sessionStats) {
        // Update session duration
        const duration = sessionStats.session_duration || 0;
        const hours = Math.floor(duration / 3600);
        const minutes = Math.floor((duration % 3600) / 60);
        const seconds = Math.floor(duration % 60);
        
        document.getElementById('session-duration').textContent = 
            `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }
    
    updateCharts(data) {
        // Update blink rate chart
        if (this.dataHistory.blinkRate.length > 0) {
            this.charts.blink.data.labels = this.dataHistory.timestamps;
            this.charts.blink.data.datasets[0].data = this.dataHistory.blinkRate;
            this.charts.blink.update('none');
        }
        
        // Update object detection chart
        if (Object.keys(this.dataHistory.objectCounts).length > 0) {
            const objectEntries = Object.entries(this.dataHistory.objectCounts);
            this.charts.objects.data.labels = objectEntries.map(([name]) => name);
            this.charts.objects.data.datasets[0].data = objectEntries.map(([, count]) => count);
            this.charts.objects.update('none');
        }
        
        // Update emotion chart
        if (data.video && data.video.emotion_scores) {
            const emotionScores = data.video.emotion_scores;
            const emotionData = [
                emotionScores.happy || 0,
                emotionScores.sad || 0,
                emotionScores.angry || 0,
                emotionScores.fear || 0,
                emotionScores.surprise || 0,
                emotionScores.neutral || 0
            ];
            
            this.charts.emotion.data.datasets[0].data = emotionData;
            this.charts.emotion.update('none');
        }
    }
    
    updateVideoFeed(frameData) {
        const videoElement = document.getElementById('video-feed');
        const placeholderElement = document.getElementById('video-placeholder');
        
        if (frameData) {
            videoElement.src = `data:image/jpeg;base64,${frameData}`;
            videoElement.style.display = 'block';
            placeholderElement.style.display = 'none';
        } else {
            videoElement.style.display = 'none';
            placeholderElement.style.display = 'flex';
        }
    }
    
    getFatiguePercentage(fatigueLevel) {
        switch (fatigueLevel.toLowerCase()) {
            case 'normal': return 20;
            case 'mild': return 40;
            case 'moderate': return 60;
            case 'severe': return 80;
            default: return 20;
        }
    }
    
    getFatigueColor(fatigueLevel) {
        switch (fatigueLevel.toLowerCase()) {
            case 'normal': return 'bg-success';
            case 'mild': return 'bg-warning';
            case 'moderate': return 'bg-warning';
            case 'severe': return 'bg-danger';
            default: return 'bg-success';
        }
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new BehavioralDashboard();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Page is hidden, reduce update frequency
        console.log('Page hidden, reducing updates');
    } else {
        // Page is visible, resume normal updates
        console.log('Page visible, resuming updates');
        if (window.dashboard) {
            window.dashboard.requestInitialData();
        }
    }
});

// Handle window resize
window.addEventListener('resize', () => {
    if (window.dashboard && window.dashboard.charts) {
        Object.values(window.dashboard.charts).forEach(chart => {
            chart.resize();
        });
    }
});
