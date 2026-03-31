# WingMaker
small flying wing model maker
installation steps

on Mac

pip install -r requirements.txt

git clone https://github.com/peterdsharpe/AeroSandbox.git

git clone https://github.com/OpenVSP/OpenVSP.git

curl -L -O https://m-selig.ae.illinois.edu/ads/archives/coord_seligFmt.zip

unzip coord_seligFmt.zip -d UIUC-Airfoil-Database

python build_knowledge.py

on Linux

pip install -r requirements.txt

git clone https://github.com/ProjectPhysX/FluidX3D.git
