# Motion Cutout

A static website that lets you pick objects in an image, cut them out, add motion, and export a 2D video (webm/mp4).

Inference runs entirely in the browser (Transformers.js + SlimSAM). No server, API key, or build step is required, and your image is never uploaded.

## Deploy on GitHub Pages

Settings → Pages → Branch: `main` / `(root)`. The site will be available at `https://<user>.github.io/<repo>/`.

## Run locally

```
python3 -m http.server
```

Then open http://localhost:8000.
