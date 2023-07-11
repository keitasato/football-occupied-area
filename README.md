#タイトル
サッカー試合映像を用いた選手の個別追跡による占有領域の可視化
(Visualization of Occupied Areas by Individually Tracking Soccer Players in Video)

#説明
選択したサッカー試合映像から、選手の占有領域を俯瞰図上で表示する動画を出力する。
![output](./field_video_final.mp4)

#実行環境
python3.8

#ライブラリ
os, sys, argparse, time, pathlib, cv2, torch, numpy, matplotlib, sklearn, subprocess, kivy, csv, random, scipy, PIL, [SST](https://github.com/shijieS/SST)

#デモ実行方法(事前にこのレポジトリと同階層にSSTをクローンしておく必要がある)
weights/sst300_0712_83000.zip を解凍する
当レポジトリのルートディレクトリで以下コマンドを実行する
'''terminal
python menu.py
'''
動画ファイルの選択を行う。
ピッチサイズの入力を行う(単位はメートル)。
「Start Analysis」ボタンを押下。
ガイドに従って対応点を16点クリックする。
![例](./field08.png)
分析が終わると、プログラムと同じディレクトリに「field_video」という名称の占有領域を可視化した動画ファイルが出力される。