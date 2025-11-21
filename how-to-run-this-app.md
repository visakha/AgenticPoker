# 🚀 How to Run "Agentic Poker" (The Windows Edition)

So you want to play some poker against robots? Excellent choice. Here is your survival guide to getting this thing running on Windows.

## 💻 The Weapon of Choice: PowerShell

Use **PowerShell**.
Why? Because the old Command Prompt (`cmd.exe`) is like playing poker with a blindfold on. PowerShell is modern, colorful, and doesn't yell at you as much.

Open it up (Search for "PowerShell" in your Start menu) and navigate to this folder.

## 🛠️ Step 0: The "Do I Have This Stuff?" Check

You need two things installed. If you don't have them, go get them. I'll wait.
1.  **Python** (3.13 or newer) -> `python --version`
2.  **Node.js** (for the fancy UI) -> `node --version`

## 🏃‍♂️ Step 1: Install the Python Brains 🧠

First, let's install the libraries that make the poker logic work.

```powershell
pip install -r requirements.txt
```

*If you see a bunch of text scrolling by, that's good. It means it's learning.*

## 🎨 Step 2: Build the Pretty Face (The UI) 💅

Now we need to compile the React frontend. This is the part that makes it look like a sci-fi movie instead of a spreadsheet.

```powershell
# Go into the UI folder
cd ui/poker_component

# Install the JavaScript dependencies (this might take a minute)
npm install

# Build the production version
npm run build

# Go back to the root folder
cd ../..
```

*Note: You only need to do this step ONCE (or whenever you change the React code).*

## 🎮 Step 3: IT'S GAME TIME! 🃏

Launch the application:

```powershell
streamlit run ui/app.py
```

A browser window should pop up automatically. If it doesn't, click the "Local URL" link it shows you.

## 🤖 Bonus: Running a Simulation

Want to watch the bots fight in the matrix (text only)?

```powershell
python cli.py --games 10
```

## 🆘 Troubleshooting

-   **"npm is not recognized"**: You didn't install Node.js, did you?
-   **"streamlit is not recognized"**: Try `python -m streamlit run ui/app.py`.
-   **The UI is blank**: Did you run `npm run build`? Be honest.

Enjoy losing your fake money! 💸
