// main.js - Core functionality for the emotion analysis app

document.addEventListener('DOMContentLoaded', function() {
    // Video upload preview
    const videoInput = document.getElementById('video');
    const videoPreview = document.getElementById('videoPreview');
    const videoContainer = document.getElementById('videoPreviewContainer');
    
    if (videoInput && videoPreview && videoContainer) {
        videoInput.addEventListener('change', function() {
            if (this.files && this.files[0]) {
                const file = this.files[0];
                
                // Check if it's a video file
                if (!file.type.match('video.*')) {
                    alert('Please select a valid video file');
                    return;
                }
                
                // Display video preview
                videoContainer.classList.remove('d-none');
                videoPreview.src = URL.createObjectURL(file);
                videoPreview.play();
            }
        });
    }
    
    // Upload progress tracking
    const uploadForm = document.getElementById('uploadForm');
    const progressBar = document.getElementById('uploadProgress');
    const progressContainer = document.getElementById('progressContainer');
    const uploadButton = document.getElementById('uploadButton');
    
    if (uploadForm && progressBar && progressContainer && uploadButton) {
        uploadForm.addEventListener('submit', function(e) {
            // Check if a file is selected
            if (!videoInput.files || videoInput.files.length === 0) {
                return; // Let the normal form validation handle it
            }
            
            e.preventDefault();
            
            // Show progress bar
            progressContainer.classList.remove('d-none');
            uploadButton.disabled = true;
            uploadButton.textContent = 'Uploading...';
            
            // Create FormData
            const formData = new FormData(uploadForm);
            
            // Create and configure XMLHttpRequest
            const xhr = new XMLHttpRequest();
            xhr.open('POST', uploadForm.action, true);
            
            // Upload progress event
            xhr.upload.addEventListener('progress', function(e) {
                if (e.lengthComputable) {
                    const percentComplete = Math.round((e.loaded / e.total) * 100);
                    progressBar.style.width = percentComplete + '%';
                    progressBar.textContent = percentComplete + '%';
                    
                    if (percentComplete === 100) {
                        progressBar.textContent = 'Processing video... Please wait';
                    }
                }
            });
            
            // Load completion event
            xhr.addEventListener('load', function() {
                if (xhr.status === 200) {
                    // Set the full response as the document HTML
                    document.open();
                    document.write(xhr.responseText);
                    document.close();
                } else {
                    alert('Upload failed. Please try again.');
                    uploadButton.disabled = false;
                    uploadButton.textContent = 'Upload & Analyze';
                    progressContainer.classList.add('d-none');
                }
            });
            
            // Error event
            xhr.addEventListener('error', function() {
                alert('Upload failed. Please try again.');
                uploadButton.disabled = false;
                uploadButton.textContent = 'Upload & Analyze';
                progressContainer.classList.add('d-none');
            });
            
            // Send the form data
            xhr.send(formData);
        });
    }
    
    // Initialize charts if we're on the results page
    if (typeof Chart !== 'undefined') {
        initializeResultsCharts();
    } else {
        console.error('Chart.js not loaded. Skipping chart initialization.');
    }
});

// Initialize all charts on the results page
function initializeResultsCharts() {
    // Check if we're on the results page by looking for chart containers
    const emotionChartElement = document.getElementById('emotionPieChart');
    const sceneChartElement = document.getElementById('sceneContextChart');
    const timelineChartElement = document.getElementById('emotionTimeline');
    const frameDetailsElement = document.getElementById('frameDetails');
    
    if (!emotionChartElement) {
        console.warn('Not on results page or chart elements missing.');
        return;
    }
    
    // Parse the data from the data attributes or script tags
    let emotionLabels = [];
    let emotionValues = [];
    let contentLabels = [];
    let contentValues = [];
    let timelineLabels = [];
    let emotionTimeline = [];
    let frameDetails = [];
    
    try {
        // Get emotion distribution data
        const emotionLabelsData = document.getElementById('emotionLabelsData');
        const emotionValuesData = document.getElementById('emotionValuesData');
        if (emotionLabelsData && emotionValuesData) {
            emotionLabels = JSON.parse(emotionLabelsData.textContent || '[]');
            emotionValues = JSON.parse(emotionValuesData.textContent || '[]');
        }
        
        // Get scene context data if available
        const contentLabelsData = document.getElementById('contentLabelsData');
        const contentValuesData = document.getElementById('contentValuesData');
        if (contentLabelsData && contentValuesData) {
            contentLabels = JSON.parse(contentLabelsData.textContent || '[]');
            contentValues = JSON.parse(contentValuesData.textContent || '[]');
        }
        
        // Get timeline data
        const timelineLabelsData = document.getElementById('timelineLabelsData');
        const emotionTimelineData = document.getElementById('emotionTimelineData');
        if (timelineLabelsData && emotionTimelineData) {
            timelineLabels = JSON.parse(timelineLabelsData.textContent || '[]');
            emotionTimeline = JSON.parse(emotionTimelineData.textContent || '[]');
        }
        
        // Get frame details if available
        const frameDetailsData = document.getElementById('frameDetailsData');
        if (frameDetailsData) {
            frameDetails = JSON.parse(frameDetailsData.textContent || '[]');
        }
    } catch (e) {
        console.error('Error parsing chart data:', e);
        // Fallback to empty data to prevent breaking the UI
        emotionLabels = ['No Data'];
        emotionValues = [100];
        contentLabels = ['No Data'];
        contentValues = [0];
        timelineLabels = [0];
        emotionTimeline = [{ timestamp: 0, emotion: 'Unknown', intensity: 0 }];
        frameDetails = [];
    }
    
    // Create emotion distribution chart
    if (emotionChartElement && emotionLabels.length && emotionValues.length) {
        createEmotionPieChart('emotionPieChart', emotionLabels, emotionValues);
    } else {
        console.warn('Emotion chart data missing or invalid.');
    }
    
    // Create scene context chart if data exists
    if (sceneChartElement && contentLabels.length && contentValues.length) {
        createSceneContextChart('sceneContextChart', contentLabels, contentValues);
    } else {
        console.warn('Scene context chart data missing or invalid.');
    }
    
    // Create timeline chart
    if (timelineChartElement && emotionTimeline.length) {
        createEmotionTimelineChart('emotionTimeline', timelineLabels, emotionTimeline);
    } else {
        console.warn('Timeline chart data missing or invalid.');
    }
    
    // Initialize frame slider if frame details exist
    if (frameDetailsElement && frameDetails.length) {
        initializeFrameSlider(frameDetails);
    } else {
        console.warn('Frame details data missing or invalid.');
    }
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}