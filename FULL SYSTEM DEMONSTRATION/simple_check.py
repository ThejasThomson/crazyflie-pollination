import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

#USER CONFIG
IMAGE_PATH   = r"E:\Downloads\new8.jpg"
YOLO_WEIGHTS = "runs/detect/train13/weights/best.pt" 
YOLO_CONF    = 0.80            
YOLO_CLASS_NAME = "strawberry"    

OUT_ANNOTATED = "annotated_search_area4.png"
OUT_CSV       = "flower_world_coords.csv"


def get_aruco_detector():
    
    aruco = cv2.aruco
    try:
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, parameters)
        return detector, dictionary, "new"
    except AttributeError:
        dictionary = aruco.Dictionary_get(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters_create()
        return (dictionary, parameters), dictionary, "old"

def detect_aruco_corners(image_bgr):
  
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    det, dictionary, api = get_aruco_detector()

    if api == "old":
        aruco = cv2.aruco
        corners, ids, _ = aruco.detectMarkers(gray, det[0], parameters=det[1])
    else:
        corners, ids, _ = det.detectMarkers(gray)

    if ids is None or len(ids) < 4:
        raise ValueError("Did not find enough ArUco markers.")

    ids = ids.flatten().tolist()
    id_to_corners = {}
    for i, mid in enumerate(ids):
        id_to_corners[mid] = corners[i].reshape(-1, 2)  # 4x2 float

    need = [0, 1, 2, 3]
    if not all(mid in id_to_corners for mid in need):
        raise ValueError(f"Missing required corner IDs 0,1,2,3. Found: {sorted(id_to_corners.keys())}")

    def center(pts):
        return np.mean(pts, axis=0)

 
    pts_img = np.array([
        center(id_to_corners[0]),  # TL
        center(id_to_corners[1]),  # TR
        center(id_to_corners[2]),  # BR
        center(id_to_corners[3]),  # BL
    ], dtype=np.float32)

    return pts_img, id_to_corners

def build_homography(pts_img, width_m, height_m):
    """
    Build image->world homography.
    World frame:
      origin at rectangle centre (0,0)
      +X to the right, +Y toward the image TOP (so top edge has +Y).
    Corners in world: TL(-W/2,+H/2), TR(W/2,+H/2), BR(W/2,-H/2), BL(-W/2,-H/2)
    """
    world_pts = np.array([
        [-width_m/2,  height_m/2],  # TL
        [ width_m/2,  height_m/2],  # TR
        [ width_m/2, -height_m/2],  # BR
        [-width_m/2, -height_m/2],  # BL
    ], dtype=np.float32)

    H, _ = cv2.findHomography(pts_img, world_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if H is None:
        raise RuntimeError("Homography estimation failed.")
    return H

def image_points_to_world(H, pts_img):
    """Map Nx2 image points to Nx2 world points using homography H."""
    pts = np.hstack([pts_img, np.ones((pts_img.shape[0], 1))])  # Nx3
    wp = (H @ pts.T).T
    wp = wp[:, :2] / wp[:, 2:3]
    return wp  # Nx2


def load_yolo(weights_path):
    return YOLO(weights_path)

def detect_flowers(model, image_bgr, conf=0.4, class_name="strawberry"):
    """
    Return:
      - Nx2 array of IMAGE-space flower centres
      - list of (xyxy, conf, cls_name) for annotation
    """
    results = model.predict(source=image_bgr, conf=conf, verbose=False)
    if len(results) == 0:
        return np.zeros((0, 2), dtype=np.float32), []

    r = results[0]
    boxes = r.boxes
    if boxes is None or boxes.xyxy is None or boxes.xyxy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), []


    names = r.names if hasattr(r, "names") else model.names
    target_cls = None
    if isinstance(names, dict):
        for k, v in names.items():
            if str(v).lower() == class_name.lower():
                target_cls = int(k)
                break

    flower_centres = []
    det_for_draw = []
    for i in range(boxes.xyxy.shape[0]):
        cls = int(boxes.cls[i].item()) if boxes.cls is not None else None
        if (target_cls is None) or (cls == target_cls):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            flower_centres.append([cx, cy])
            conf = float(boxes.conf[i].item()) if boxes.conf is not None else 0.0
            cname = names.get(cls, str(cls)) if isinstance(names, dict) else str(cls)
            det_for_draw.append(((int(x1), int(y1), int(x2), int(y2)), conf, cname))
    return np.array(flower_centres, dtype=np.float32), det_for_draw

#Annotation
def annotate_and_save(image_bgr, id_to_corners, flower_img_pts, flower_world_pts, det_for_draw,
                      out_path="annotated_search_area.png"):
    vis = image_bgr.copy()

 
    for mid, pts in id_to_corners.items():
        pts_i = pts.astype(int)
        cv2.polylines(vis, [pts_i], True, (0, 255, 0), 2)
        c = np.mean(pts_i, axis=0).astype(int)
        cv2.circle(vis, tuple(c), 4, (0, 255, 0), -1)
        cv2.putText(vis, f"ID {mid}", tuple(c + np.array([6, -6])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

   
    for ((x1, y1, x2, y2), conf, cname), (cx, cy), (wx, wy) in zip(det_for_draw, flower_img_pts, flower_world_pts):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.circle(vis, (int(cx), int(cy)), 4, (255, 0, 0), -1)
        label = f"{cname} {conf:.2f} | x={wx:+.3f} m, y={wy:+.3f} m"
        cv2.putText(vis, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 200, 255), 2)

    cv2.imwrite(out_path, vis)
    print(f"[INFO] Saved annotated image: {out_path}")


def main():
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"IMAGE_PATH not found: {IMAGE_PATH}")

    image_bgr = cv2.imread(IMAGE_PATH)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {IMAGE_PATH}")

    # Ask physical dimensions
    try:
        width_m  = float(input("Enter RECTANGLE width in metres (left→right, X): ").strip())
        height_m = float(input("Enter RECTANGLE height in metres (top→bottom, Y): ").strip())
    except Exception:
        raise ValueError("Please enter numeric width/height in metres.")

 
    print("[INFO] Detecting ArUco 0/1/2/3…")
    pts_img, id_to_corners = detect_aruco_corners(image_bgr)


    print("[INFO] Estimating homography (origin at rectangle centre)…")
    H = build_homography(pts_img, width_m, height_m)


    print("[INFO] Loading YOLOv8 model and detecting strawberries…")
    model = load_yolo(YOLO_WEIGHTS)
    flower_img_pts, det_for_draw = detect_flowers(model, image_bgr, conf=YOLO_CONF, class_name=YOLO_CLASS_NAME)

    if flower_img_pts.shape[0] == 0:
        print("[WARN] No flowers detected — nothing to export/annotate.")
        return


    flower_world_pts = image_points_to_world(H, flower_img_pts)  # Nx2
    flower_world_pts_list = flower_world_pts.tolist()

  
    print("\n[INFO] Flower world coordinates (metres, origin at rectangle centre):")
    rows = []
    for i, (x, y) in enumerate(flower_world_pts_list, 1):
        print(f"  Flower {i:02d}: x={x:+.3f}, y={y:+.3f}")
        rows.append({"flower_id": i, "x_m": x, "y_m": y})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"[INFO] Saved coordinates CSV: {OUT_CSV}")

 
    annotate_and_save(image_bgr, id_to_corners, flower_img_pts, flower_world_pts_list,
                      det_for_draw, out_path=OUT_ANNOTATED)

if __name__ == "__main__":
    main()
