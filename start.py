"""
CreditSage Loan Advisory Agent — Alternative Entry Point

Evaluator can run either:
    python run.py   OR   python start.py
Both launch the same Streamlit application.
"""

import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "streamlit", "run", "creditsage_app.py"])
