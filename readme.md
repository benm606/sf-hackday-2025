# setup instructions
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cd voice-ui & npm install & cd ..

# run chat room
cd voice-ui & npm run build & npm start & cd ..

# run agent
python agent/agent.py start

# fill db
python db/mock/extract_problems_script.py

# run insights UI
python insights-ui/ui.py
