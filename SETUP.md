# SETUP.md

## Overview
1. Clone the repo using git clone
2. Create an environment
3. Activate the environment
4. Install dependencies from requirements.txt
5. Setup API key (GROQ)
6. Run the app

### Environment
Create the environment:
```conda create -n duke-dining python=3.10```
Activate the environment:
```conda activate duke-dining```
Install dependencies: 
```pip install -r requirements.txt```
Set API Key:
Create a .env file in the root directory, then:
```GROQ_API_KEY=your_api_key_here```

### Running the APP
Use ```python app.py``` to run the UI