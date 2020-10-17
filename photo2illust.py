import cv2
import numpy as np
import os

def sub_color(src, K):
  z = src.reshape((-1, 3))
  z = np.float32(z)
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

  ret, label, center = cv2.kmeans(z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  center = np.uint8(center)
  res = center[label.flatten()]
  
  return res.reshape((src.shape))

def anime_filter(img, K):
  gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
  edge = cv2.blur(gray, (3, 3))
  edge = cv2.Canny(edge, 50, 150, apertureSize=3)
  edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
  img = sub_color(img, K)

  return cv2.subtract(img, edge)

def main():
  target_path = os.path.join(os.path.abspath(os.curdir), "images")
  img = cv2.imread('./images/shashin.jpg')
  print(os.path.join(target_path, "写真.jpg"))
  anime = anime_filter(img, 10)
  target_path = os.path.join(target_path, "illust.jpg")
  cv2.imwrite(target_path, anime)

if __name__ == '__main__':
  main()
