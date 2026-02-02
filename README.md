# research-agent

ロボット研究分野に特化した自動リサーチエージェント。arXiv・RSS・会議ページを巡回し、Claude Code でスコアリング・要約を行い、週次 Markdown レポートを生成する。

## 基本思想

- **1 agent = 1 分野** — 本エージェントは cs.RO (ロボティクス) を対象とする
- **収集と一次選別まで** — 判断と理解は人間が行う
- **ノイズを減らし高シグナルのみを残す**

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Claude Code CLI がインストール済みであること:

```bash
claude --version
```

## 使い方

```bash
python agent.py                    # 通常実行 (デフォルト: opus)
python agent.py -m sonnet          # sonnet で実行
python agent.py -m haiku           # haiku で実行 (高速・低コスト)
python agent.py --dry-run          # レポート生成なしでテスト
python agent.py -c custom.yaml     # カスタム設定ファイルを指定
```

## 処理フロー

1. **収集** — arXiv API (cs.RO)、RSS フィード、指定 URL を巡回
2. **重複排除** — URL + タイトルの fingerprint (SHA-256) で重複を検出
3. **Claude Code スコアリング** — Claude Code CLI で 5 軸評価 (各 2 点、最大 10 点)
   - 先端技術の利用
   - コード公開の有無
   - データセット/ベンチマーク
   - 実機/リアルワールド検証
   - 安全性/堅牢性
4. **レポート生成** — スコア上位論文を Markdown ファイルとして出力

## 設定

`config.yaml` で以下を調整可能:

- 対象カテゴリ、取得件数
- Claude Code モデル (`opus` / `sonnet` / `haiku`)、バッチサイズ
- スコアリング基準と配点
- レポートの出力先、上位件数、最低スコア

## systemd timer で週次自動実行

`systemd/` にサービスとタイマーのユニットファイルを同梱。

```bash
# ユニットファイルをコピー
sudo cp systemd/research-agent.service /etc/systemd/system/
sudo cp systemd/research-agent.timer /etc/systemd/system/

# 必要に応じてパスやユーザーを編集
sudo systemctl edit research-agent.service

# 有効化・開始
sudo systemctl daemon-reload
sudo systemctl enable --now research-agent.timer

# 状態確認
systemctl status research-agent.timer
systemctl list-timers research-agent.timer
journalctl -u research-agent.service
```

## テスト

```bash
pip install pytest
python -m pytest test_agent.py -v
```

## ライセンス

MIT
