# WingMaker
This framework automates the creation of aircraft geometry using a Mac-based AI Controller and a Linux-based CFD Physics Server.

💻 Mac Setup (The Controller)
Environment:
python -m venv mlx_env
source mlx_env/bin/activate

Dependencies:
pip install -r requirements.txt
git clone https://github.com/peterdsharpe/AeroSandbox.git
git clone https://github.com/OpenVSP/OpenVSP.git
curl -L -O https://m-selig.ae.illinois.edu/ads/archives/coord_seligFmt.zip
unzip coord_seligFmt.zip -d UIUC-Airfoil-Database

🐧 Linux Setup (The Physics Server)
Environment:
python -m venv ai_env
source ai_env/bin/activate

Dependencies:
pip install -r requirements.txt
git clone https://github.com/ProjectPhysX/FluidX3D.git
also you should install CUDA drivers from your distro repository

Launch Server:
uvicorn physics_server2:app --host 0.0.0.0 --port 8000

🚀 Workflow
1. Start the Linux Server first.
2. Run python mlx_brain2.py on your Mac.
3. Enter your design goal (e.g., "Long endurance surveillance drone").
4. Watch the VLM Plateau and CFD Refinement stages.
5. Collect the ULTIMATE_CFD_CHAMPION.stl from the Linux machine.
