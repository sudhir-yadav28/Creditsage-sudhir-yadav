"""
CreditSage Loan Advisory Agent — Entry Point

Launch the Streamlit application with a single command:
    python run.py

This script invokes Streamlit programmatically so the evaluator
does not need to know the underlying framework.
"""

import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "streamlit", "run", "creditsage_app.py"])
