// chart.js - Visualization functions for emotion analysis

// Color palette for emotions
const emotionColors = {
    'Angry': 'rgba(255, 99, 132, 0.8)',
    'Disgust': 'rgba(75, 192, 192, 0.8)',
    'Fear': 'rgba(153, 102, 255, 0.8)',
    'Happy': 'rgba(255, 206, 86, 0.8)',
    'Sad': 'rgba(54, 162, 235, 0.8)',
    'Surprise': 'rgba(255, 159, 64, 0.8)',
    'Neutral': 'rgba(201, 203, 207, 0.8)',
    'Unknown': 'rgba(100, 100, 100, 0.8)'
};

// Create emotion distribution pie chart
function createEmotionPieChart(elementId, emotionLabels, emotionValues) {
    const ctx = document.getElementById(elementId)?.getContext('2d');
    if (!ctx) {
        console.error(`Canvas element ${elementId} not found.`);
        return null;
    }
    
    // Ensure data is valid
    const validLabels = emotionLabels.length ? emotionLabels : ['No Data'];
    const validValues = emotionValues.length ? emotionValues : [100];
    
    // Generate colors array based on emotion labels
    const colors = validLabels.map(label => emotionColors[label] || 'rgba(100, 100, 100, 0.8)');
    
    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: validLabels,
            datasets: [{
                data: validValues,
                backgroundColor: colors,
                borderColor: colors.map(color => color.replace('0.8)', '1)')),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'right',
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.formattedValue || '0';
                            return `${label}: ${value}%`;
                        }
                    }
                },
                title: {
                    display: true,
                    text: 'Emotion Distribution'
                }
            },
            cutout: '50%'
        }
    });
}

// Create scene context bar chart
function createSceneContextChart(elementId, contentLabels, contentValues) {
    const ctx = document.getElementById(elementId)?.getContext('2d');
    if (!ctx) {
        console.error(`Canvas element ${elementId} not found.`);
        return null;
    }
    
    // Ensure data is valid
    const validLabels = contentLabels.length ? contentLabels : ['No Data'];
    const validValues = contentValues.length ? contentValues : [0];
    
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: validLabels,
            datasets: [{
                label: 'Context Confidence',
                data: validValues,
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.formattedValue || '0';
                            return `${value}% confidence`;
                        }
                    }
                },
                title: {
                    display: true,
                    text: 'Scene Context Analysis'
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Confidence (%)'
                    }
                }
            }
        }
    });
}

// Create emotional timeline chart
function createEmotionTimelineChart(elementId, timeLabels, emotionTimelineData) {
    const ctx = document.getElementById(elementId)?.getContext('2d');
    if (!ctx) {
        console.error(`Canvas element ${elementId} not found.`);
        return null;
    }
    
    // Ensure data is valid
    const validData = emotionTimelineData.length ? emotionTimelineData : [{ timestamp: 0, emotion: 'Unknown', intensity: 0 }];
    
    // Extract unique emotions
    const uniqueEmotions = [...new Set(validData.map(item => item.emotion || 'Unknown'))];
    
    // Create a dataset for each emotion
    const datasets = uniqueEmotions.map(emotion => {
        // Filter data points for this emotion
        const points = validData.filter(item => (item.emotion || 'Unknown') === emotion);
        
        return {
            label: emotion,
            data: points.map(item => ({
                x: item.timestamp || 0,
                y: item.intensity || 0
            })),
            backgroundColor: emotionColors[emotion] || 'rgba(100, 100, 100, 0.8)',
            borderColor: (emotionColors[emotion] || 'rgba(100, 100, 100, 0.8)').replace('0.8', '1'),
            borderWidth: 2,
            pointRadius: 5,
            tension: 0.3,
            fill: false
        };
    });

    return new Chart(ctx, {
        type: 'line',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Emotional Timeline'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const emotion = context.dataset.label || 'Unknown';
                            const time = parseFloat(context.parsed.x || 0).toFixed(2);
                            const intensity = parseFloat(context.parsed.y || 0).toFixed(2);
                            return `${emotion} @ ${time}s: ${intensity * 100}% intensity`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'Time (seconds)'
                    }
                },
                y: {
                    min: 0,
                    max: 1,
                    title: {
                        display: true,
                        text: 'Intensity'
                    }
                }
            }
        }
    });
}

// Create frame slider for frame-by-frame analysis
function initializeFrameSlider(frameDetails) {
    if (!frameDetails || !Array.isArray(frameDetails) || frameDetails.length === 0) {
        console.warn('Invalid or empty frameDetails data. Slider not initialized.');
        return;
    }
    
    const slider = document.getElementById('frameSlider');
    const frameImage = document.getElementById('frameImage');
    const timeDisplay = document.getElementById('timeDisplay');
    const primaryEmotionDisplay = document.getElementById('primaryEmotion');
    const faceCountDisplay = document.getElementById('faceCount');
    
    if (!slider || !frameImage || !timeDisplay || !primaryEmotionDisplay || !faceCountDisplay) {
        console.error('Required DOM elements for frame slider not found.');
        return;
    }
    
    // Set slider range
    slider.min = 0;
    slider.max = frameDetails.length - 1;
    slider.value = 0;
    
    // Initial frame display
    updateFrameDisplay(0);
    
    // Slider event handler
    slider.addEventListener('input', function() {
        const frameIndex = parseInt(this.value);
        updateFrameDisplay(frameIndex);
    });
    
    function updateFrameDisplay(index) {
        if (index < 0 || index >= frameDetails.length) {
            console.warn(`Invalid frame index: ${index}`);
            return;
        }
        
        const frame = frameDetails[index];
        frameImage.src = frame.image_url || '';
        timeDisplay.textContent = frame.time || '0.00s';
        primaryEmotionDisplay.textContent = frame.primary_emotion || 'Unknown';
        faceCountDisplay.textContent = frame.face_count || 0;
        
        // Update emotion bars if they exist
        const emotions = frame.emotions || {};
        Object.keys(emotions).forEach(emotion => {
            const bar = document.getElementById(`emotion-${emotion.toLowerCase().replace(/\s/g, '-')}`);
            if (bar) {
                const value = (emotions[emotion] || 0) * 100;
                bar.style.width = `${value}%`;
                bar.textContent = `${value.toFixed(1)}%`;
            }
        });
        
        // Update scene context bars if they exist
        const content = frame.content || {};
        Object.keys(content).forEach(context => {
            const bar = document.getElementById(`context-${context.toLowerCase().replace(/\s/g, '-')}`);
            if (bar) {
                const value = (content[context] || 0) * 100;
                bar.style.width = `${value}%`;
                bar.textContent = `${value.toFixed(1)}%`;
            }
        });
    }
}