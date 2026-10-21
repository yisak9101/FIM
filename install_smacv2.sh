#!/bin/bash
set -e  # Exit on any error

# Reinstall SMACv2
pip uninstall -y SMAC
pip install --upgrade pip setuptools wheel
pip install git+https://github.com/oxwhirl/smacv2.git

# Download and unzip SMAC maps
cd "$SC2PATH/Maps/SMAC_Maps/"
rm -f *.SC2Map
wget https://github.com/oxwhirl/smacv2/releases/download/maps/SMAC_Maps.zip
unzip -o SMAC_Maps.zip
rm -f SMAC_Maps.zip
