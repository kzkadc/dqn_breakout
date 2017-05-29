# dqn_breakout
ChainerとOpenAI Gymで実装したDQNです。
[Nature論文](https://www.nature.com/nature/journal/v518/n7540/full/nature14236.html "Human-level control through deep reinforcement learning")
を参考に，できるだけ忠実に実装しました。

`docker/Dockerfile`からコンテナをビルドすればすぐに実行できます。

## 学習
`python main_breakout.py`を実行

### オプション
- `-g 整数` or `--gpu 整数`：GPUのID
- `--learn_start`：学習を開始するステップ数
- `test_interval`：テストの間隔
- `-i` or `--iteration`：イテレーション回数
- `--mem 整数`：replay memoryの容量

## テスト
`python test_agent.py modelfile`を実行

### オプション
- `modelfile`：NNのモデルファイル（必須）
- `-m フォルダ名` or `--movie フォルダ名`：プレイ動画の保存先
- `-r` or `--random`：モデルによらずランダム行動をする
- `-i フォルダ名` or `--image フォルダ名`：プレイ画面を連続画像として保存
- `-q` or `--qplot`：Q値を時系列としてCSVで出力
