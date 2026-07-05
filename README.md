# ✨ Vantage Vision

AI-powered visual search, refined.

Vantage Vision is a Streamlit app that runs YOLOv11 object detection over a folder of images, then lets you search, filter, and browse them by the objects found inside — with bounding boxes, class highlighting, and a customizable grid view.

## How it works

1. **Analyze** — Point Vantage Vision at a folder of images. A YOLOv11 model scans every photo and detects the objects in it, no manual tagging needed.
2. **Store** — Detections are saved to a metadata file you can reload anytime, so you only need to analyze a folder once.
3. **Search** — Pick the classes you care about, choose "at least one" or "all", and instantly filter down to the matching images.
4. **Review** — Browse results in a resizable grid with optional bounding boxes, and highlight only the classes you searched for.

## Features

- 🔍 Object detection search across an entire image folder
- 🧠 Powered by YOLOv11 (via the `ultralytics` package)
- 💾 Save/reload analyzed metadata so you don't have to re-run detection every time
- 🎛️ Filter by "at least one selected class" or "all selected classes", with optional max-count thresholds per class
- 🖼️ Adjustable grid view (2–6 columns) with bounding-box overlays and match highlighting
- 📤 Export search results as JSON
- 🌌 Dark, glassmorphic UI with animated ambient glow

## Requirements

- Python 3.9+
- [Streamlit](https://streamlit.io/)
- [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLOv11)
- Pillow

Install dependencies:
```bash
pip install -r requirements.txt
```

> If you don't have a `requirements.txt` yet, a minimal one looks like:
> ```
> streamlit
> ultralytics
> pillow
> ```

## Configuration

Vantage Vision reads optional defaults from a config file (see `src/config.py`), including:
- `model.yolo` — path/name of the YOLO weights to use by default
- `model.conf_threshold` — confidence threshold for displaying detections
- `app.default_grid` — default grid size for the results view
- `app.show_boxes` — whether bounding boxes are shown by default
- `app.highlight_matches` — whether only matching classes are highlighted by default
- `app.search_mode` — default search mode ("Atleast one of the selected ones" / "All of the selected ones")

If no config is found, sensible defaults are used automatically.

## Running the app

```bash
streamlit run app.py
```

Then in the browser:
1. Choose **"Analyze New Images"**, enter the path to a folder of images and your YOLO weights, and click **Start Inferencing**.
2. Once analysis finishes, use **Discover** to select the classes you want, set an optional threshold, and click **Find Images**.
3. Adjust the **View Options** (show boxes, grid size, highlight matches) to browse your results.
4. Optionally export your results as JSON from the **Export Options** section.

Next time, skip analysis entirely by choosing **"Load Existing Data"** and pointing to the saved metadata file.

## Project structure

```
.
├── app.py                 # Streamlit UI and app logic
├── src/
│   ├── inference.py        # YOLOv11Inference wrapper
│   ├── utils.py             # save / load_data / get_class_freq helpers
│   └── config.py            # load_config helper
└── README.md
```

## License

Add your preferred license here (e.g. MIT).
