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