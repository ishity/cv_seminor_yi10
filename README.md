# Motion Cutout
画像の物体を選んで切り抜き、モーションを付けて2D動画(webm/mp4)にする静的サイト。
推論はブラウザ内 (Transformers.js + SlimSAM)。ビルド不要。

## GitHub Pages で公開
Settings → Pages → Branch: `main` / `(root)` を選択。公開URLは `https://<user>.github.io/<repo>/`。

## ローカル確認
`python3 -m http.server` → http://localhost:8000
