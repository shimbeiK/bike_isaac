from pxr import Usd
import sys

input_file = "bike_V3_mjcf.usd"
# 拡張子を .usda (AはASCIIの略) にすることで、人間が読めるテキスト形式になります
output_file = "bike_V3_mjcf_readable.usda"

print(f"Reading {input_file}...")
stage = Usd.Stage.Open(input_file)
if stage:
    stage.Export(output_file)
    print(f"Success! Saved as {output_file}")
else:
    print("Failed to open USD file.")