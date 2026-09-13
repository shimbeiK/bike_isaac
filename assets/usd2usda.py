from isaacsim import SimulationApp
# ヘッドレスモード（GUIなし）でIsaac Simをバックグラウンド起動し、パスを通す
simulation_app = SimulationApp({"headless": True})
from pxr import Usd
import sys

input_file = "/home/shin-linux/bike_isaac/assets/bike_v3/bike_v3.usd"
# 拡張子を .usda (AはASCIIの略) にすることで、人間が読めるテキスト形式になります
output_file = "/home/shin-linux/bike_isaac/assets/bike_v3/bike_v3_readable.usda"

print(f"Reading {input_file}...")
stage = Usd.Stage.Open(input_file)
if stage:
    stage.Export(output_file)
    print(f"Success! Saved as {output_file}")
else:
    print("Failed to open USD file.")