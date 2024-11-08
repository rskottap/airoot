#!/bin/bash

pip install pip==23.3.1
pip install -r requirements.txt
pip install gradio-client==0.8.1
pip install gradio==3.48.0
# install cuda fix
python -m pip install ort-nightly-gpu --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-12-nightly/pypi/simple/
sudo apt update
sudo apt install sox

