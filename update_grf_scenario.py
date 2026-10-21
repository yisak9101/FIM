import gfootball
import shutil
import os

gfootball_path = os.path.dirname(gfootball.__file__)
scenario_dir = os.path.join(gfootball_path, 'scenarios')

source_dir = 'data'

for filename in os.listdir(source_dir):
    src_path = os.path.join(source_dir, filename)
    dst_path = os.path.join(scenario_dir, filename)

    if os.path.isfile(src_path):
        shutil.copy2(src_path, dst_path)
