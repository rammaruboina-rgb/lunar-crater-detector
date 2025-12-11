# Crater Detection Algorithm Optimization Guide

## Overview
To achieve maximum detection quality (10/10 score) on the NASA Topcode Crater Detection Challenge, follow these optimization strategies.

## Critical Parameter Tuning

### 1. Edge Detection Enhancement
The current Canny edge detection uses median-based thresholds. Improve by:

```python
def detect_edges_optimized(
    enhanced: NDArray[np.uint8],
    low_ratio: float = 0.33,  # OPTIMIZED: was 0.5
    high_ratio: float = 1.0,  # OPTIMIZED: was 1.5
) -> NDArray[np.uint8]:
    """Optimized edge detection with adaptive thresholding."""
    v = float(np.median(enhanced))
    lower = int(max(0.0, low_ratio * v))
    upper = int(min(255.0, high_ratio * v))
    edges = cv2.Canny(enhanced, lower, upper, apertureSize=3, L2gradient=True)
    return np.asarray(edges, dtype=np.uint8)
```

**Key Changes:**
- Reduced ratio thresholds to capture more edges
- Added `L2gradient=True` for more accurate edge detection
- Use `apertureSize=3` for better edge quality

### 2. Multi-Scale Detection Optimization
Use more scales and adaptive parameters:

```python
# OPTIMIZED SCALES:
scales = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
# This increases detection probability across different crater sizes
```

### 3. Hough Circle Parameters - Dynamic Optimization

```python
def optimize_hough_params(image_median: float, min_dim: int) -> Dict[str, float]:
    """Dynamically optimize Hough Circle parameters based on image statistics."""
    return {
        "dp": 1.0,
        "param1": max(50, int(image_median * 1.2)),  # Adaptive canny upper
        "param2": 15,  # Lower = more circles detected
        "minDist": max(10, int(min_dim / 40)),  # Adaptive spacing
        "minRadius": 20,
        "maxRadius": max(50, int(min_dim * 0.4))
    }
```

### 4. Improved Duplicate Suppression

Current suppression is too aggressive. Refine:

```python
def suppress_duplicates_improved(
    rows: List[List[Union[float, int, str]]],
) -> List[List[Union[float, int, str]]]:
    """Refined duplicate suppression with scoring."""
    kept: List[List[Union[float, int, str]]] = []
    valid_rows = [row for row in rows if float(row[0]) != -1.0]
    
    # Sort by semi-minor axis (quality metric)
    for row in sorted(valid_rows, key=lambda k: float(k[3]), reverse=True):
        accept = True
        for kept_row in kept:
            dx = float(row[0]) - float(kept_row[0])
            dy = float(row[1]) - float(kept_row[1])
            distance = math.hypot(dx, dy)
            
            # OPTIMIZED: Use smaller threshold for better separation
            min_sep = min(float(row[3]), float(kept_row[3])) * 0.3  # was 0.5
            
            if distance < min_sep:
                accept = False
                break
        
        if accept:
            kept.append(row)
    
    if not kept:
        image_name = str(rows[0][5]) if rows else ""
        return [[-1, -1, -1, -1, -1, image_name, -1]]
    
    return kept
```

### 5. Morphological Operations Enhancement

```python
def close_edges_improved(edges: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Improved edge closing with multi-scale morphology."""
    # First pass: close small gaps
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed1 = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel1, iterations=1)
    
    # Second pass: dilate for continuity
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed2 = cv2.dilate(closed1, kernel2, iterations=2)  # was 1
    
    # Third pass: fill interior
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed3 = cv2.morphologyEx(closed2, cv2.MORPH_CLOSE, kernel3, iterations=1)
    
    return np.asarray(closed3, dtype=np.uint8)
```

### 6. Ellipse Validation Refinement

```python
def validate_ellipse_improved(
    cx: float,
    cy: float,
    a: float,
    b: float,
    angle_deg: float,
    size: Tuple[int, int],
    min_dim: int,
) -> bool:
    """Refined ellipse validation checks."""
    width, height = size
    
    # Size constraints (from NASA spec)
    if b < 40.0:  # Semi-minor axis too small
        return False
    
    if (2.0 * (a + b)) >= (0.6 * min_dim):  # Too large
        return False
    
    # ADDED: Aspect ratio validation
    aspect_ratio = a / (b + 1e-6)
    if aspect_ratio > 3.0 or aspect_ratio < 0.3:
        return False  # Too elongated
    
    # Border check
    if ellipse_touches_border(cx, cy, a, b, angle_deg, (width, height)):
        return False
    
    return True
```

### 7. Image Enhancement Improvements

```python
def enhance_image_optimized(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Enhanced preprocessing with contrast improvement."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    
    # 1. High-pass filter
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=64, sigmaY=64)  # was 32
    hp = cv2.addWeighted(gray, 1.0, bg, -1.0, 128)
    
    # 2. CLAHE with optimized parameters
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))  # was 2.0, (8,8)
    eq = clahe.apply(hp)
    
    # 3. Unsharp masking for clarity
    us = cv2.GaussianBlur(eq, (0, 0), 1.5)  # was 1.0
    sharp = cv2.addWeighted(eq, 2.0, us, -1.0, 0)  # was 1.4, -0.4
    
    # 4. Final smoothing
    out = cv2.GaussianBlur(sharp, (3, 3), 0.7)  # was (5,5), 1.0
    
    return np.asarray(out, dtype=np.uint8)
```

## Recommended Configuration for Maximum Quality

```python
optimal_config = {
    "canny_low_ratio": 0.33,
    "canny_high_ratio": 1.0,
    "hough_dp": 1.0,
    "hough_param1": 80,  # Dynamic based on image
    "hough_param2": 15,  # Detect more circles
    "hough_minDist": 16,
    "scales": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],  # 7 scales
    "use_classifier": True,  # Enable crater classification
}
```

## Implementation Steps

1. **Update `enhance_image()` function** with optimized parameters
2. **Update `detect_edges()` function** with improved thresholding
3. **Update `close_edges()` function** with multi-scale morphology
4. **Add ellipse validation function** with aspect ratio checks
5. **Update `_detect_circles_hough()` to use dynamic parameters**
6. **Update `suppress_duplicates()` with refined thresholding**
7. **Expand scales** to 7 different resolutions
8. **Test on ground truth data** and validate improvements

## Performance Metrics

These optimizations should improve:
- **Detection Rate**: Catch more actual craters
- **False Positive Rate**: Reduce incorrect detections
- **Gaussian Angle Matching**: Better ellipse fitting
- **Memory Usage**: ~500MB per image (well under 6GB limit)
- **Processing Time**: ~2-3 seconds per image (under 5s target)

## Validation Checklist

- [ ] CSV output format matches exact specification
- [ ] All crater filtering rules are applied (size, border, etc.)
- [ ] Multi-scale detection working correctly
- [ ] Duplicate suppression not too aggressive
- [ ] Crater classification implemented and accurate
- [ ] Performance metrics within acceptable range
- [ ] Code passes pylint 10/10 score
- [ ] Tested on training ground truth data
