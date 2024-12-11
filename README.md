Set-Up Instructions:
1. Ensure that Python 3.7 is installed on your local machine. You can do this by:
	a. Windows/Mac
		i. Visit Python Website & download the latest version
		ii. Run the installer and check the box "Add Python to PATH" during installation, then follow the prompts to install

	b. Linux (In Command Prompt)
		i. python3 --version
		ii. sudo apt update
		iii. sudo apt install python3
2. Download & Unzip Project Zip
	a. Should include
		i. stockDataGathering.py
		ii. combined.json
		iii. requirements.txt
3. Install necessary libraries
	a. pip install -r requirements.txt
4. Configure Directory & Change any existing pathways in stockDataGathering.py if necessary
5. Sign up and obtain an API Key from AlphaVantage
	a. Replace the api_key in the file with the new one you have obtained
6. Verify the Environment with the line
	a. python -c "import requests, bs4, transformers, torch, numpy, matplotlib, sklearn, bs4, collections, json, datetime, pandas"
7. Ensure file addresses are correct
	a. File addresses such as for combined.json in format_tweets() need to be adjusted.
8. Run Python file with the command: python3 stockDataGathering.py