## Gel Image Lane Analysis (English)

This project is a small Python tool to automatically analyze **lanes** and **bands** in a gel image (e.g. DNA/protein gel electrophoresis).  
The script `lane_analysis.py`:

- Loads and smooths a **grayscale gel image**.
- Uses vertical projection to automatically detect **lane boundaries**.
- Computes the **intensity profile** for each lane and finds **peaks** corresponding to bands.
- Presents results both in **interactive windows** and as **CSV + PNG outputs**.

---

### Requirements

- Python 3.8+ (around 3.10 recommended)
- The following packages:
  - `opencv-python`
  - `numpy`
  - `matplotlib`

Install (example):

```bash
pip install opencv-python numpy matplotlib
```

---

### Project Structure

- `lane_analysis.py` → Main analysis script
- `gel_image.png` → Gel image to analyze (must be in the same folder)
- `outputs/` → Output folder (created automatically when the script runs)
  - `lane_boundaries.png` → Gel image with lane boundaries drawn
  - `profile_lane_X.png` → Intensity profile plot for each lane (X = lane number)
  - `gel_report.csv` → Numerical summary report for all lanes and bands

---

### How to Run

1. Make sure `lane_analysis.py` and `gel_image.png` are in the same directory.
2. Install the required Python packages.
3. In a terminal, move to the project directory and run:

```bash
python lane_analysis.py
```

---

### Interactive Usage

When the script is running, two windows appear:

- `Jel Goruntusu - Serit Sinirlari`  
  Gel image with lane boundaries marked by red vertical lines.

- `Serit Yogunluk Profili`  
  Intensity profile for the currently selected lane.

Keyboard controls:

- `1`–`9` → Show the profile for the corresponding lane (if it exists).
- `D` / `d` or **right arrow key** → Move to the next lane (wraps around).
- `A` / `a` or **left arrow key** → Move to the previous lane (wraps around).
- `q` → Quit the program.

Each time you select a lane:

- The console prints a textual explanation and a **peak summary table**.
- The profile plot is saved as `outputs/profile_lane_X.png`.
- The numerical data is appended/updated in `outputs/gel_report.csv`.

---

### `gel_report.csv` Contents

Each row corresponds to a **single band (peak)** in a specific lane:

- `lane_index` → Lane number (1-based)
- `peak_row` → Row index of the band in the image (top to bottom)
- `peak_intensity` → Peak intensity value
- `rel_intensity_max_percent` → Percentage relative to the strongest band in that lane
- `rel_intensity_sum_percent` → Percentage relative to the total intensity of all peaks in that lane
- `z_score` → Z-score of the peak intensity within that lane
- `normalized_position` → Normalized vertical position in the gel (0–1)

You can open this file in Excel, R, Python, etc. for further analysis.

---

### Notes and Tips

- Thresholds and kernel sizes can be adjusted in the code to better match different gel qualities.
- If lane detection is poor, pre-processing the image (cropping, contrast adjustment) may help.
- As a next step, it would be straightforward to add a loop to batch-process multiple gel images.


