import cv2
import os
import time
import numpy as np
from deepface import DeepFace
 
# ── Config ────────────────────────────────────────────────────────────────────
REFERENCE_FOLDER   = "reference_faces"
REFERENCE_IMAGES   = []          # fallback list if folder missing
MAX_ATTEMPTS       = 100
VERIFY_THRESHOLD   = 3
DISTANCE_THRESHOLD = 0.5         # slightly looser to account for glasses/accessories
COUNTDOWN_SECONDS  = 3
 
# DeepFace settings
# - detector: "retinaface" or "mtcnn" are far better than haar for glasses
# - model:    "ArcFace" handles occlusion (glasses, masks) much better than VGG-Face
DETECTOR_BACKEND = "retinaface"  # options: retinaface | mtcnn | opencv | ssd
RECOGNITION_MODEL = "ArcFace"   # options: ArcFace | Facenet512 | VGG-Face | DeepFace
# ─────────────────────────────────────────────────────────────────────────────
 
os.makedirs(REFERENCE_FOLDER, exist_ok=True)
 
# Keep Haar as a lightweight fallback just for drawing the preview box
haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
 
def load_reference_images():
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    paths = [
        os.path.join(REFERENCE_FOLDER, f)
        for f in os.listdir(REFERENCE_FOLDER)
        if f.lower().endswith(exts)
    ]
    return paths if paths else REFERENCE_IMAGES
 
 
def verify_against_all(face_img, ref_images):
    """
    Try to match face_img against every reference image.
    Uses ArcFace model + RetinaFace detector for glasses robustness.
    Returns (True, matched_path) or (False, None).
    """
    for ref_path in ref_images:
        try:
            result = DeepFace.verify(
                img1_path          = face_img,
                img2_path          = ref_path,
                model_name         = RECOGNITION_MODEL,
                detector_backend   = DETECTOR_BACKEND,
                enforce_detection  = False,   # don't crash if detector misses
                align              = True,    # face alignment helps with glasses
            )
            if result["verified"] and result["distance"] < DISTANCE_THRESHOLD:
                return True, ref_path
        except Exception as e:
            print(f"  DeepFace error for '{ref_path}': {e}")
    return False, None
 
 
def detect_faces_haar(frame):
    """Lightweight Haar detection just for live preview rectangles."""
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    return faces
 
 
def draw_centered_text(frame, text, y, font_scale=0.9,
                        color=(255, 255, 255), thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (frame.shape[1] - tw) // 2
    cv2.putText(frame, text, (x+2, y+2), font, font_scale, (0,0,0), thickness+1)
    cv2.putText(frame, text, (x,   y  ), font, font_scale, color,   thickness)
 
 
# ── Mode selection ────────────────────────────────────────────────────────────
 
def choose_mode():
    while True:
        frame = np.zeros((400, 640, 3), dtype="uint8")
        frame[:] = (40, 40, 40)
 
        draw_centered_text(frame, "Face Auth System",       80,  1.2, (100, 220, 100))
        draw_centered_text(frame, "Press  R  to Register", 180,  0.9, (200, 200, 200))
        draw_centered_text(frame, "Press  L  to Login",    240,  0.9, (200, 200, 200))
        draw_centered_text(frame, "Press  Q  to Quit",     310,  0.7, (120, 120, 120))
 
        cv2.imshow("Face Auth", frame)
        key = cv2.waitKey(0) & 0xFF
 
        if   key == ord("r"): return "register"
        elif key == ord("l"): return "login"
        elif key == ord("q"):
            cv2.destroyAllWindows()
            exit()
 
 
# ── Registration ──────────────────────────────────────────────────────────────
 
def capture_snapshot(cap, label, save_path):
    """
    Show a live feed with countdown and save a snapshot.
    Returns True on success, False if cancelled.
    """
    start_time = time.time()
 
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            return False
 
        elapsed   = time.time() - start_time
        remaining = max(0, COUNTDOWN_SECONDS - int(elapsed))
 
        faces        = detect_faces_haar(frame)
        face_detected = len(faces) > 0
        display      = frame.copy()
 
        for (x, y, w, h) in faces:
            color = (0, 255, 0) if remaining == 0 else (0, 200, 255)
            cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
 
        if face_detected:
            msg = f"Hold still... {remaining}" if remaining > 0 else "Captured!"
            draw_centered_text(display, msg, 50, 1.1, (0, 220, 255))
        else:
            draw_centered_text(display, "No face detected — move closer",
                               50, 0.85, (0, 80, 255))
            start_time = time.time()  # reset if face disappears
 
        draw_centered_text(display, f"Registering: {label}",
                           display.shape[0] - 50, 0.7, (180, 180, 180))
        draw_centered_text(display, "C = capture now   |   Q = cancel",
                           display.shape[0] - 20, 0.65, (120, 120, 120))
 
        cv2.imshow("Face Auth", display)
        key = cv2.waitKey(1) & 0xFF
 
        if (face_detected and elapsed >= COUNTDOWN_SECONDS) or \
           (key == ord("c") and face_detected):
            cv2.imwrite(save_path, frame)
            print(f"  Saved: {save_path}")
            return True
 
        if key == ord("q"):
            return False
 
 
def register_face(cap):
    """
    Register a person by capturing 1–3 photos:
      1. Without glasses (base)
      2. With glasses  (optional but strongly recommended)
      3. Different angle (optional)
    Each saved image is named  name_1.jpg, name_2.jpg, etc.
    """
    print("\n[REGISTER] Look at the camera.")
    name = input("Enter a name/label for this face (e.g. 'alice'): ").strip()
    if not name:
        name = f"face_{int(time.time())}"
 
    shots = [
        ("without glasses (or your normal look)", f"{name}_1.jpg"),
        ("with glasses  (skip with Q if N/A)",     f"{name}_glasses.jpg"),
        ("slight left/right angle  (skip with Q)", f"{name}_angle.jpg"),
    ]
 
    saved_any = False
    for description, filename in shots:
        save_path = os.path.join(REFERENCE_FOLDER, filename)
        print(f"\n  Shot: {description}")
        ok = capture_snapshot(cap, f"{name} — {description}", save_path)
        if ok:
            saved_any = True
            # Brief confirmation flash
            confirm = np.zeros((200, 640, 3), dtype="uint8")
            draw_centered_text(confirm, f"Saved: {filename}",
                               100, 0.9, (0, 255, 120))
            cv2.imshow("Face Auth", confirm)
            cv2.waitKey(900)
        else:
            print(f"  Skipped: {filename}")
 
    if saved_any:
        print(f"\n[REGISTER] Done! Registered '{name}' with multiple shots.")
    return saved_any, name
 
 
# ── Login ─────────────────────────────────────────────────────────────────────
 
def login_face(cap):
    reference_images = load_reference_images()
    if not reference_images:
        print("[LOGIN] No reference images found. Please register first.")
        return
 
    print(f"\n[LOGIN] {len(reference_images)} reference image(s) loaded.")
    print(f"        Detector: {DETECTOR_BACKEND}  |  Model: {RECOGNITION_MODEL}")
 
    attempts      = 0
    verify_count  = 0
    matched_image = None
 
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
 
        display         = frame.copy()
        face_recognized = False
 
        # Use the full frame — RetinaFace/MTCNN handle detection internally
        matched, ref_path = verify_against_all(frame, reference_images)
 
        # Also run Haar just for drawing a box on screen
        faces = detect_faces_haar(frame)
        for (x, y, w, h) in faces:
            color = (0, 255, 0) if matched else (0, 0, 255)
            cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
 
        if matched:
            verify_count += 1
            print(f"  Verifying... {verify_count}/{VERIFY_THRESHOLD}  (matched: {ref_path})")
            if verify_count >= VERIFY_THRESHOLD:
                matched_image   = ref_path
                face_recognized = True
        else:
            verify_count = 0
 
        # HUD
        draw_centered_text(display, f"Attempt {attempts}/{MAX_ATTEMPTS}",
                           display.shape[0] - 20, 0.65, (180, 180, 180))
        draw_centered_text(display, "Q = cancel",
                           display.shape[0] - 50, 0.65, (120, 120, 120))
        if verify_count > 0 and not face_recognized:
            draw_centered_text(display, f"Almost...  {verify_count}/{VERIFY_THRESHOLD}",
                               50, 0.9, (0, 220, 255))
 
        cv2.imshow("Face Auth", display)
 
        if face_recognized:
            perform_login(display, matched_image)
            return
 
        attempts += 1
        if attempts >= MAX_ATTEMPTS:
            prompt_password_login()
            return
 
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
 
 
def perform_login(frame, ref_path):
    # Strip _1 / _glasses / _angle suffix to get the clean name
    base = os.path.splitext(os.path.basename(ref_path))[0]
    name = base.replace("_glasses", "").replace("_angle", "").replace("_1", "")
    print(f"\nLogged in as '{name}'  (matched: {ref_path})")
 
    success = frame.copy()
    draw_centered_text(success, f"Welcome, {name}!",
                       success.shape[0] // 2, 1.3, (0, 255, 120))
    cv2.imshow("Face Auth", success)
    cv2.waitKey(2000)
    # TODO: implement actual login logic here
 
 
def prompt_password_login():
    print(f"❌  Face not recognized after {MAX_ATTEMPTS} attempts. "
          "Switching to password login...")
    # TODO: implement password login
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
 
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
 
    try:
        while True:
            mode = choose_mode()
 
            if mode == "register":
                ok, _ = register_face(cap)
                if ok:
                    print("[REGISTER] Proceeding to login...")
                    cv2.waitKey(500)
                    login_face(cap)
                    break
 
            elif mode == "login":
                login_face(cap)
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()