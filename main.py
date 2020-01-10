import cv2
import argparse
import matplotlib.pyplot as plt
from box_layout import bounding_box

def main():
    parser = argparse.ArgumentParser(description='Synth')
    parser.add_argument('filename', help='path to image file [REQUIRED]')
    args = parser.parse_args()

    bounding_box.layout(args.filename)

if __name__ == "__main__":
    main()