# Quickstart Guide 🚀

Don't want to read the academic papers and mathematical specifications? Just want to see the AI control traffic in real-time? 

Follow these 3 simple steps to get the Full-Stack Dashboard running on your local machine.

## Prerequisites
Ensure you have Python 3.8+ installed on your computer.

---

## Step 1: Install Dependencies
Open your terminal (or command prompt), navigate to the `python` folder of this project, and install the required libraries:

```bash
cd python
pip install -r requirements.txt
```
*(If you don't have a requirements file, run: `pip install yt-dlp flask flask-cors ultralytics opencv-python torch numpy`)*

## Step 2: Start the Backend Server
The Flask server is required to run the PyTorch model and the YOLOv8 object detection. It will process the video feeds and stream them to your browser.

Run this command inside the `python` directory:
```bash
python server.py
```
*You should see output indicating the server is running on `http://127.0.0.1:5000`.*

## Step 3: Open the Dashboard
Keep the terminal open (the server needs to stay running).

Open your favorite web browser (Chrome, Edge, Firefox) and navigate to:
👉 **http://127.0.0.1:5000**

### Using the Dashboard:
1. **The JavaScript Simulation:** By default, you will see the mathematical MDP simulation running on the JS canvas.
2. **Live World Map:** Click the **"Global Map"** button. Click any marker (e.g., London or Tokyo) to instantly connect to a live CCTV camera and watch the AI process traffic in real-time!
3. **Upload Real Video:** Click **"Upload Real Video"** to upload any local `.mp4` dashcam or traffic footage to see how the AI handles it.
4. **Predictive GPS Radar:** Watch the simulated UI panel on the bottom left calculate incoming platoons.
