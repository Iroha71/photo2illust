import cv2
import numpy as np
import sys

"""
画像をドット化する
-----
Usage
-----
1. $ python photo2dot.py [画像名] [ドット化倍率]
  - [画像名]: imageディレクトリに格納した画像名
  - [ドット化倍率]: 0 - 1.0間で指定。値が低いほどドットが粗くなる
"""

"""
画像の減色を行う

Parameters
----------
src: numpy
  入力画像のnumpy配列
K: int
  k-meansクラスタリングを行う際に必要なクラスタ数

Returns
-------
res: numpy
  入力画像を減色加工したnumpy配列
"""
def reduce_color(src, K):
  print(f'減色開始前の次元数: {src.shape}')
  z = src.reshape((-1, 3))
  z = np.float32(z)
  print(f'減色開始時の次元数: {z.shape}')

  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  ret, label, center = cv2.kmeans(z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

  center = np.uint8(center)

  res = center[label.flatten()]

  return res.reshape((src.shape))

"""
画像の縮小→拡大を行い、モザイク処理をかける

Parameters
----------
img: numpy
  入力画像のnumpy配列
alpha: float
  画像の縮小率(小さいほどモザイクが粗くなる)

Returns
-------
img: numpy
  モザイク処理後の画像のnumpy配列
"""
def do_mosaic(img, alpha):
  h, w, ch = img.shape

  img = cv2.resize(img, (int(w * alpha), int(h * alpha)))
  img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

  return img

"""
画像をドット絵にする

Parameters
----------
img: numpy
  ドット絵変換対象
alpha: int
  モザイクをかける際の倍率
K: int
  k-meansで減色する際のクラスタ数
import_image_name: string
  入力画像のファイル名

Returns
-------
img: numpy
  ドット絵加工後の画像
"""
def change_pixel(img, alpha=2, K=4, import_image_name='sample.jpg'):
  img = do_mosaic(img, alpha)
  cv2.imwrite(f'./images/mosaiced@{import_image_name}', img)

  return reduce_color(img, K)

def main(import_image, dot_rate=0.5):
  dot_rate = float(dot_rate)
  img = cv2.imread('./images/' + import_image)
  reduced_image = reduce_color(img, 4)
  cv2.imwrite(f'./images/reduced_color@{import_image}', reduced_image)

  dot_image = change_pixel(img, dot_rate, 4, import_image)
  cv2.imwrite(f'./images/dot@{import_image}', dot_image)

if __name__ == '__main__':
  args = sys.argv
  if len(args) < 3:
    main(args[1])
  else:
    main(args[1], args[2])
