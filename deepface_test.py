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
DISTANCE_THRESHOLD = 0.4
COUNTDOWN_SECONDS  = 3           # countdown before auto-capture in register mode
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(REFERENCE_FOLDER, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
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


def verify_against_all(face_region, ref_images):
    for ref_path in ref_images:
        try:
            result = DeepFace.verify(face_region, ref_path, enforce_detection=False)
            if result["verified"] and result["distance"] < DISTANCE_THRESHOLD:
                return True, ref_path
        except Exception as e:
            print(f"  DeepFace error for '{ref_path}': {e}")
    return False, None


def draw_centered_text(frame, text, y, font_scale=0.9, color=(255, 255, 255), thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (frame.shape[1] - tw) // 2
    # Shadow for readability
    cv2.putText(frame, text, (x+2, y+2), font, font_scale, (0, 0, 0), thickness + 1)
    cv2.putText(frame, text, (x, y),     font, font_scale, color,     thickness)


# ── Mode selection ────────────────────────────────────────────────────────────

def choose_mode():
    """Show a menu window and return 'register' or 'login'."""
    while True:
        frame = np.zeros((400, 640, 3), dtype="uint8")
        frame[:] = (40, 40, 40)

        draw_centered_text(frame, "Face Auth System",       80,  1.2, (100, 220, 100))
        draw_centered_text(frame, "Press  R  to Register", 180,  0.9, (200, 200, 200))
        draw_centered_text(frame, "Press  L  to Login",    240,  0.9, (200, 200, 200))
        draw_centered_text(frame, "Press  Q  to Quit",     310,  0.7, (120, 120, 120))

        cv2.imshow("Face Auth", frame)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("r"):
            return "register"
        elif key == ord("l"):
            return "login"
        elif key == ord("q"):
            cv2.destroyAllWindows()
            exit()


# ── Registration ──────────────────────────────────────────────────────────────

def register_face(cap):
    """
    Capture a photo with a live countdown, detect a face, save it
    to REFERENCE_FOLDER, then return the saved path (or None on failure).
    """
    print("\n[REGISTER] Look at the camera. A photo will be taken automatically.")
    name = input("Enter a name/label for this face (e.g. 'alice'): ").strip()
    if not name:
        name = f"face_{int(time.time())}"

    save_path  = os.path.join(REFERENCE_FOLDER, f"{name}.jpg")
    start_time = time.time()
    captured_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            return None

        elapsed   = time.time() - start_time
        remaining = max(0, COUNTDOWN_SECONDS - int(elapsed))

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        display      = frame.copy()
        face_detected = len(faces) > 0

        for (x, y, w, h) in faces:
            color = (0, 255, 0) if remaining == 0 else (0, 200, 255)
            cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)

        # Countdown / status overlay
        if face_detected:
            if remaining > 0:
                draw_centered_text(display, f"Hold still...  {remaining}",
                                   50, 1.1, (0, 220, 255))
            else:
                draw_centered_text(display, "Captured!",
                                   50, 1.2, (0, 255, 100))
        else:
            draw_centered_text(display, "No face detected — move closer",
                               50, 0.85, (0, 80, 255))
            start_time = time.time()   # reset countdown if face disappears

        draw_centered_text(display, f"Registering: {name}",
                           display.shape[0] - 50, 0.7, (180, 180, 180))
        draw_centered_text(display, "C = capture now   |   Q = cancel",
                           display.shape[0] - 20, 0.65, (120, 120, 120))

        cv2.imshow("Face Auth", display)
        key = cv2.waitKey(1) & 0xFF

        # Auto-capture after countdown (face must be present)
        if face_detected and elapsed >= COUNTDOWN_SECONDS:
            captured_frame = frame.copy()
            break

        # Manual capture
        if key == ord("c") and face_detected:
            captured_frame = frame.copy()
            break

        if key == ord("q"):
            print("[REGISTER] Cancelled.")
            return None

    if captured_frame is None:
        return None

    # Crop the largest detected face and save
    gray2  = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2GRAY)
    faces2 = face_cascade.detectMultiScale(
        gray2, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces2) == 0:
        cv2.imwrite(save_path, captured_frame)   # fallback: full frame
    else:
        x, y, w, h = max(faces2, key=lambda f: f[2] * f[3])   # largest face
        pad = 20
        x1 = max(0, x - pad);  y1 = max(0, y - pad)
        x2 = min(captured_frame.shape[1], x + w + pad)
        y2 = min(captured_frame.shape[0], y + h + pad)
        cv2.imwrite(save_path, captured_frame[y1:y2, x1:x2])

    print(f"[REGISTER] Face saved to '{save_path}'")
    return save_path


# ── Login ─────────────────────────────────────────────────────────────────────

def login_face(cap):
    reference_images = load_reference_images()
    if not reference_images:
        print("[LOGIN] No reference images found. Please register first.")
        return

    print(f"\n[LOGIN] {len(reference_images)} reference image(s) loaded. Look at the camera.")

    attempts     = 0
    verify_count = 0
    matched_image = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        display         = frame.copy()
        face_recognized = False

        for (x, y, w, h) in faces:
            face_region        = frame[y:y+h, x:x+w]
            matched, ref_path  = verify_against_all(face_region, reference_images)

            if matched:
                verify_count += 1
                print(f"  Verifying... {verify_count}/{VERIFY_THRESHOLD}  (matched: {ref_path})")
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)

                if verify_count >= VERIFY_THRESHOLD:
                    matched_image   = ref_path
                    face_recognized = True
                    break
            else:
                verify_count = 0
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), 2)

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
    name = os.path.splitext(os.path.basename(ref_path))[0]
    print(f"\nLogged in as '{name}'  (matched: {ref_path})")

    success = frame.copy()
    draw_centered_text(success, f"Welcome, {name}!",
                       success.shape[0] // 2, 1.3, (0, 255, 120))
    cv2.imshow("Face Auth", success)
    cv2.waitKey(2000)
    # TODO: implement actual login logic here


def prompt_password_login():
    print(f"Face not recognized after {MAX_ATTEMPTS} attempts. "
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
                saved = register_face(cap)
                if saved:
                    print("[REGISTER] Done! Proceeding to login...")
                    cv2.waitKey(800)
                    login_face(cap)
                    break
                # if cancelled, loop back to menu

            elif mode == "login":
                login_face(cap)
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()