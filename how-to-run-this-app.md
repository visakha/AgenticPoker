# 🚀 How to Run "Agentic Poker" (The Windows Edition)

So you want to play some poker against robots? Excellent choice. Here is your survival guide to getting this thing running on Windows.

## 💻 The Weapon of Choice: PowerShell

Use **PowerShell**.
Why? Because the old Command Prompt (`cmd.exe`) is like playing poker with a blindfold on. PowerShell is modern, colorful, and doesn't yell at you as much.

Open it up (Search for "PowerShell" in your Start menu) and navigate to this folder.

## 🛠️ Step 0: The "Do I Have This Stuff?" Check

You need Python installed. If you don't have it, go get it. I'll wait.
-   **Python** (3.13 or newer) -> `python --version`

## 🏃‍♂️ Step 1: Install the Python Brains 🧠

Install the libraries that make the poker logic work:

```powershell
pip install -r requirements.txt
```

*If you see a bunch of text scrolling by, that's good. It means it's learning.*

## 🎮 Step 2: IT'S GAME TIME! 🃏

Launch the FastAPI application:

```powershell
uvicorn ui.fastapi_app:app --reload --port 8000
```

Then open your browser to:
```
http://localhost:8000
```

**Features**:
- ⚡ **Instant response** - No more waiting for page reloads!
- 🔄 **Auto-play mode** - Watch 20 steps per second
- 🎨 **Modern dark theme** - Easy on the eyes
- 📡 **Real-time updates** - WebSocket magic

## 🤖 Bonus: Running a Simulation

Want to watch the bots fight in the matrix (text only)?

```powershell
python cli.py --games 10
```

## 🆘 Troubleshooting

-   **"uvicorn is not recognized"**: Try `python -m uvicorn ui.fastapi_app:app --reload --port 8000`
-   **Port 8000 already in use**: Change to a different port like `--port 8001`
-   **WebSocket not connecting**: Check browser console (F12) for errors

Enjoy losing your fake money! 💸
