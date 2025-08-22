FPL (Fantasy Premier League) Analyzer

A Fantasy Premier League (FPL) data analysis and visualization tool built with Python. This project helps FPL managers make smarter decisions by analyzing player statistics, team performance, and suggesting optimal strategies for team selection.

🚀 Features

📊 Player & Team Analysis – Compare players across multiple metrics.

🔍 Custom Team Analyzer – Upload your squad and evaluate strengths & weaknesses.

⚡ Formation & Strategy Testing – Experiment with different formations.

📈 Data Visualization – Interactive plots and charts for insights.

🖼️ Image Upload Support – Import screenshots for analysis (in progress).

🛠️ Tech Stack

Python 3.12+

Pandas / NumPy – Data processing

Matplotlib / Plotly – Visualization

Flask – Backend (migration from Streamlit in progress)

OpenCV – OCR & image-based squad recognition

📂 Project Structure
FPL/
│-- app/                # Flask backend code
│-- data/               # Sample datasets
│-- static/             # CSS, JS, Images
│-- templates/          # HTML templates
│-- requirements.txt    # Dependencies
│-- README.md           # Project documentation

⚙️ Installation & Setup

Clone the repo:

git clone https://github.com/gauravs585/FPL.git
cd FPL


Create a virtual environment:

python -m venv .venv
.venv\Scripts\activate   # On Windows
source .venv/bin/activate # On Mac/Linux


Install dependencies:

pip install -r requirements.txt


Run the app:

flask run

📖 Usage

Open your browser at http://127.0.0.1:5000/

Upload your FPL team data or enter player details.

Explore stats, visualizations, and formation suggestions.

🤝 Contributing

Contributions are welcome! Feel free to fork this repo, create a new branch, and submit a PR.

📜 License

This project is licensed under the MIT License.
