# upload_ui.py
# upload_ui.py

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse
import os

router = APIRouter()

@router.get("/upload", response_class=HTMLResponse)
async def upload_form(page: str = Query(None, description="Suggested Notion page")):
    """Serve the upload form with optional page suggestion"""
    
    # Read the HTML file content
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Upload to Juno Knowledge Base</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            padding: 40px;
            max-width: 500px;
            width: 100%;
            backdrop-filter: blur(10px);
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .header h1 {
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .header p {
            color: #666;
            font-size: 1.1em;
        }

        .form-group {
            margin-bottom: 25px;
        }

        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
            font-size: 1.1em;
        }

        .file-input-wrapper {
            position: relative;
            display: inline-block;
            width: 100%;
        }

        .file-input {
            width: 100%;
            padding: 15px;
            border: 2px dashed #667eea;
            border-radius: 12px;
            background: #f8f9ff;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
            color: #667eea;
            font-size: 1.1em;
        }

        .file-input:hover {
            border-color: #764ba2;
            background: #f0f2ff;
            transform: translateY(-2px);
        }

        .file-input input[type="file"] {
            position: absolute;
            opacity: 0;
            width: 100%;
            height: 100%;
            cursor: pointer;
        }

        .select-wrapper {
            position: relative;
        }

        select {
            width: 100%;
            padding: 15px;
            border: 2px solid #e1e8ed;
            border-radius: 12px;
            background: white;
            font-size: 1.1em;
            color: #333;
            appearance: none;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        select:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .select-wrapper::after {
            content: '▼';
            position: absolute;
            right: 15px;
            top: 50%;
            transform: translateY(-50%);
            color: #667eea;
            pointer-events: none;
        }

        .upload-btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1.2em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }

        .upload-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }

        .upload-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 12px;
            font-weight: 600;
            text-align: center;
            display: none;
        }

        .result.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }

        .result.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }

        .progress {
            width: 100%;
            height: 6px;
            background: #e1e8ed;
            border-radius: 3px;
            overflow: hidden;
            margin-top: 15px;
            display: none;
        }

        .progress-bar {
            height: 100%;
            background: linear-gradient(135deg, #667eea, #764ba2);
            width: 0%;
            transition: width 0.3s ease;
        }

        .file-info {
            margin-top: 10px;
            padding: 10px;
            background: #f8f9ff;
            border-radius: 8px;
            font-size: 0.9em;
            color: #666;
            display: none;
        }

        @media (max-width: 600px) {
            .container {
                padding: 30px 20px;
            }
            
            .header h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Juno</h1>
            <p>Upload files to the knowledge base</p>
        </div>

        <form id="uploadForm">
            <div class="form-group">
                <label for="fileInput">📁 Select File</label>
                <div class="file-input-wrapper">
                    <div class="file-input">
                        <input type="file" id="fileInput" name="file" required>
                        <span>Click to browse or drag & drop</span>
                    </div>
                </div>
                <div class="file-info" id="fileInfo"></div>
            </div>

            <div class="form-group">
                <label for="pageSelect">📄 Notion Page</label>
                <div class="select-wrapper">
                    <select id="pageSelect" name="notion_page_name" required>
                        <option value="">Choose destination...</option>
                        <option value="Team Interviews">Team Interviews</option>
                        <option value="Brand Guidelines">Brand Guidelines</option>
                        <option value="Team Directory">Team Directory</option>
                        <option value="Mission, Vision, Values">Mission, Vision, Values</option>
                        <option value="Strategic Goals">Strategic Goals</option>
                        <option value="Competition Handbook & Rules">Competition Handbook & Rules</option>
                        <option value="SOPs">SOPs</option>
                        <option value="General Knowledge">General Knowledge</option>
                    </select>
                </div>
            </div>

            <button type="submit" class="upload-btn" id="uploadBtn">
                Upload to Juno
            </button>

            <div class="progress" id="progressContainer">
                <div class="progress-bar" id="progressBar"></div>
            </div>
        </form>

        <div class="result" id="result"></div>
    </div>

    <script>
        const uploadForm = document.getElementById('uploadForm');
        const fileInput = document.getElementById('fileInput');
        const fileInfo = document.getElementById('fileInfo');
        const uploadBtn = document.getElementById('uploadBtn');
        const result = document.getElementById('result');
        const progressContainer = document.getElementById('progressContainer');
        const progressBar = document.getElementById('progressBar');

        // Get API base URL from current location
        const API_BASE = window.location.origin;

        // File input change handler
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const fileSize = (file.size / 1024 / 1024).toFixed(2);
                fileInfo.innerHTML = `
                    <strong>${file.name}</strong><br>
                    Size: ${fileSize} MB<br>
                    Type: ${file.type || 'Unknown'}
                `;
                fileInfo.style.display = 'block';
                
                // Update file input display
                const fileInputSpan = document.querySelector('.file-input span');
                fileInputSpan.textContent = file.name;
            }
        });

        // Drag and drop handlers
        const fileInputWrapper = document.querySelector('.file-input');
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            fileInputWrapper.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            fileInputWrapper.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            fileInputWrapper.addEventListener(eventName, unhighlight, false);
        });

        function highlight(e) {
            fileInputWrapper.style.borderColor = '#764ba2';
            fileInputWrapper.style.background = '#f0f2ff';
        }

        function unhighlight(e) {
            fileInputWrapper.style.borderColor = '#667eea';
            fileInputWrapper.style.background = '#f8f9ff';
        }

        fileInputWrapper.addEventListener('drop', handleDrop, false);

        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length > 0) {
                fileInput.files = files;
                const event = new Event('change', { bubbles: true });
                fileInput.dispatchEvent(event);
            }
        }

        // Form submission
        uploadForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const file = fileInput.files[0];
            
            if (!file) {
                showResult('Please select a file', 'error');
                return;
            }

            // Show progress
            uploadBtn.disabled = true;
            uploadBtn.textContent = 'Uploading...';
            progressContainer.style.display = 'block';
            result.style.display = 'none';
            
            // Animate progress bar
            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += Math.random() * 15;
                if (progress > 90) progress = 90;
                progressBar.style.width = progress + '%';
            }, 200);

            try {
                const response = await fetch(`${API_BASE}/upload_file`, {
                    method: 'POST',
                    body: formData
                });

                clearInterval(progressInterval);
                progressBar.style.width = '100%';

                const responseData = await response.json();
                
                if (response.ok) {
                    showResult(`✅ File "${file.name}" uploaded successfully to ${formData.get('notion_page_name')}!`, 'success');
                    uploadForm.reset();
                    fileInfo.style.display = 'none';
                    document.querySelector('.file-input span').textContent = 'Click to browse or drag & drop';
                } else {
                    throw new Error(responseData.detail || 'Upload failed');
                }
            } catch (error) {
                clearInterval(progressInterval);
                console.error('Upload error:', error);
                showResult(`❌ Upload failed: ${error.message}`, 'error');
            } finally {
                uploadBtn.disabled = false;
                uploadBtn.textContent = 'Upload to Juno';
                progressContainer.style.display = 'none';
                progressBar.style.width = '0%';
            }
        });

        function showResult(message, type) {
            result.textContent = message;
            result.className = `result ${type}`;
            result.style.display = 'block';
            
            // Auto-hide success messages after 5 seconds
            if (type === 'success') {
                setTimeout(() => {
                    result.style.display = 'none';
                }, 5000);
            }
        }

        // Auto-detect page from URL parameters
        const urlParams = new URLSearchParams(window.location.search);
        const suggestedPage = urlParams.get('page');
        if (suggestedPage) {
            const pageSelect = document.getElementById('pageSelect');
            // Try to find matching option
            for (let option of pageSelect.options) {
                if (option.value.toLowerCase().includes(suggestedPage.toLowerCase()) ||
                    suggestedPage.toLowerCase().includes(option.value.toLowerCase())) {
                    option.selected = true;
                    break;
                }
            }
        }
    </script>
</body>
</html>"""
    
    return html_content