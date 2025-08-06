# Industrial Energy Coaching System

A Flask-based web application for monitoring real-time energy consumption of industrial equipment and displaying AI-generated optimization suggestions.

## Directory Structure
- `app.py`: Main Flask application.
- `templates/index.html`: HTML template for the dashboard.
- `static/css/styles.css`: Custom CSS styles.
- `static/js/chart.js`: Placeholder for additional JavaScript.
- `data/energy_data.json`: JSON file for real-time energy data.
- `requirements.txt`: Python dependencies.
- `README.md`: Project instructions.

## Prerequisites
- Python (v3.8+): Install from [python.org](https://www.python.org)
- pip: Ensure it's installed (`python -m ensurepip --upgrade`)

## Setup Instructions
1. **Create the Project Folder**:
   - Create a folder named `energy-coaching-system`.
   - Create subfolders: `templates/`, `static/css/`, `static/js/`, `data/`.
   - Place all files in their respective locations as per the directory structure.

2. **Install Dependencies**:
   - Navigate to the project folder:
     ```bash
     cd energy-coaching-system
     ```
   - Install Python dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Update Real-Time Data**:
   - Modify `data/energy_data.json` manually or via a script to add new energy measurements.
   - Example script to append data (`append_data.py`):
     ```python
     import json
     import time
     from datetime import datetime
     import random

     while True:
         with open('data/energy_data.json', 'r+') as f:
             data = json.load(f)
             new_data = {
                 "equipmentId": f"EQ{random.randint(1, 5)}",
                 "energy": round(random.uniform(20, 100), 2),
                 "timestamp": datetime.now().isoformat()
             }
             data.append(new_data)
             f.seek(0)
             json.dump(data, f, indent=4)
         time.sleep(5)
     ```

4. **Run the Application**:
   - Start the Flask server:
     ```bash
     python app.py
     ```
   - Open `http://localhost:5000` in your browser.

## Expected Output
- A dashboard with:
  - A real-time line chart of energy consumption (updated every 5 seconds from `energy_data.json`).
  - A list of static AI suggestions.

## Updating Real-Time Data
- **Manual**: Edit `data/energy_data.json` to add new entries:
  ```json
  {
      "equipmentId": "EQ1",
      "energy": 55.23,
      "timestamp": "2025-05-18T21:57:00.000Z"
  }
  ```

## Azure Deployment Instructions

### 1. Prepare Your Project
- Ensure your code is in a folder with:
  - `app.py` (main Flask app)
  - `requirements.txt` (all dependencies listed)
  - `static/` and `templates/` folders
  - Any other needed files (e.g., `users_db.py`, data files)

### 2. Create a `requirements.txt`
If not already present, run:
```
pip freeze > requirements.txt
```
This ensures all dependencies are listed.

### 3. Create an Azure App Service
- Go to [Azure Portal](https://portal.azure.com)
- Click **Create a resource** > **Web App**
- Choose:
  - **Publish:** Code
  - **Runtime stack:** Python 3.11 (or your version)
  - **Operating System:** Linux
  - **Region:** Choose your region
  - **App Service Plan:** Choose a plan (B1 is good for testing)

### 4. Deploy Your Code
#### Option A: Using GitHub
- Push your code to a GitHub repository.
- In Azure Portal, go to your Web App > **Deployment Center**
- Choose **GitHub** and connect your repo.
- Azure will auto-deploy on every push.

#### Option B: Using ZIP Deploy
- Zip your project folder (all files, not the folder itself).
- In Azure Portal, go to your Web App > **Deployment Center** > **Zip Deploy**
- Upload your zip file.

#### Option C: Using Azure CLI
```
az webapp up --name <your-app-name> --resource-group <your-resource-group> --runtime "PYTHON|3.11"
```
This command deploys your app from the current directory.

### 5. Configure Startup Command
In Azure Portal > Your Web App > **Configuration** > **General settings**:
- Set **Startup Command** to:
```
gunicorn --bind=0.0.0.0 --timeout 600 app:app
```
(Assumes your Flask app is in `app.py` and the Flask instance is named `app`.)

### 6. Set Environment Variables
If you use secrets (e.g., MongoDB URI, API keys), set them in Azure Portal > **Configuration** > **Application settings**.

### 7. Verify and Test
- Go to your Web App URL (e.g., `https://<your-app-name>.azurewebsites.net`)
- Check logs in Azure Portal > **Log Stream** for errors.

### 8. (Optional) Connect to MongoDB Atlas
- If you use MongoDB Atlas, ensure your connection string is set as an environment variable.
- Whitelist Azure App Service IPs in MongoDB Atlas.

### 9. (Optional) Enable HTTPS
- Azure App Service provides HTTPS by default.

### 10. (Optional) Custom Domain
- In Azure Portal > Your Web App > **Custom domains**, add and verify your domain.

### 11. (Optional) Scale Up
- In Azure Portal > Your Web App > **Scale Up**, choose a higher plan for more resources.