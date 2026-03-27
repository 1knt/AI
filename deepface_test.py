import cv2
import os
import time
import hashlib
import json
import threading
import numpy as np
from deepface import DeepFace

# ── Config ────────────────────────────────────────────────────────────────────
REFERENCE_FOLDER      = "reference_faces"
PASSWORDS_FILE        = os.path.join(REFERENCE_FOLDER, "passwords.json")
REFERENCE_IMAGES      = []
MAX_FACE_ATTEMPTS     = 3        # failed face scans before switching to password
MAX_PASSWORD_ATTEMPTS = 3        # wrong passwords before lockout
VERIFY_THRESHOLD      = 3        # consecutive matches needed to confirm identity
DISTANCE_THRESHOLD    = 0.5
COUNTDOWN_SECONDS     = 3

DETECTOR_BACKEND  = "opencv"
RECOGNITION_MODEL = "Facenet"
FRAME_SCALE       = 0.5
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(REFERENCE_FOLDER, exist_ok=True)

print("[Init] Loading model, please wait...")
try:
    DeepFace.build_model(RECOGNITION_MODEL)
    print(f"[Init] '{RECOGNITION_MODEL}' ready.\n")
except Exception as e:
    print(f"[Init] Warning: {e}")

haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ── Password storage ──────────────────────────────────────────────────────────

def _hash(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_passwords():
    if os.path.exists(PASSWORDS_FILE):
        with open(PASSWORDS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_password(name, password):
    db = load_passwords()
    db[name.lower()] = _hash(password)
    with open(PASSWORDS_FILE, "w") as f:
        json.dump(db, f)

def check_password(name, password):
    db = load_passwords()
    stored = db.get(name.lower())
    return stored is not None and stored == _hash(password)

def account_exists(name):
    db = load_passwords()
    return name.lower() in db


# ── Button UI ─────────────────────────────────────────────────────────────────

class Button:
    def __init__(self, label, x, y, w, h,
                 color=(70, 70, 70),
                 hover_color=(100, 180, 100),
                 text_color=(255, 255, 255)):
        self.label       = label
        self.x, self.y   = x, y
        self.w, self.h   = w, h
        self.color       = color
        self.hover_color = hover_color
        self.text_color  = text_color
        self.hovered     = False

    def contains(self, mx, my):
        return self.x <= mx <= self.x + self.w and self.y <= my <= self.y + self.h

    def draw(self, frame):
        bg = self.hover_color if self.hovered else self.color
        cv2.rectangle(frame,
                      (self.x + 3, self.y + 3),
                      (self.x + self.w + 3, self.y + self.h + 3),
                      (20, 20, 20), -1)
        cv2.rectangle(frame,
                      (self.x, self.y),
                      (self.x + self.w, self.y + self.h),
                      bg, -1)
        cv2.rectangle(frame,
                      (self.x, self.y),
                      (self.x + self.w, self.y + self.h),
                      (200, 200, 200), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.8, 2
        (tw, th), _ = cv2.getTextSize(self.label, font, scale, thickness)
        tx = self.x + (self.w - tw) // 2
        ty = self.y + (self.h + th) // 2
        cv2.putText(frame, self.label, (tx, ty), font, scale,
                    self.text_color, thickness, cv2.LINE_AA)


class ButtonMenu:
    def __init__(self, title, buttons, width=640, height=420,
                 bg_color=(35, 35, 35)):
        self.title    = title
        self.buttons  = buttons
        self.width    = width
        self.height   = height
        self.bg_color = bg_color
        self._clicked = None

    def _mouse_cb(self, event, mx, my, flags, param):
        for btn in self.buttons:
            btn.hovered = btn.contains(mx, my)
            if event == cv2.EVENT_LBUTTONDOWN and btn.hovered:
                self._clicked = btn.label

    def run(self):
        self._clicked = None
        cv2.setMouseCallback("Face Auth", self._mouse_cb)

        while self._clicked is None:
            frame = np.zeros((self.height, self.width, 3), dtype="uint8")
            frame[:] = self.bg_color

            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, _), _ = cv2.getTextSize(self.title, font, 1.1, 2)
            cv2.putText(frame, self.title,
                        ((self.width - tw) // 2, 80),
                        font, 1.1, (100, 220, 100), 2, cv2.LINE_AA)
            cv2.line(frame, (80, 105), (self.width - 80, 105), (80, 80, 80), 1)

            for btn in self.buttons:
                btn.draw(frame)

            cv2.imshow("Face Auth", frame)
            key = cv2.waitKey(16) & 0xFF
            if key == ord("q"):
                self._clicked = "Quit"

        cv2.setMouseCallback("Face Auth", lambda *a: None)
        return self._clicked


def make_main_menu():
    cx = 640 // 2
    bw, bh = 260, 55
    buttons = [
        Button("Register", cx - bw // 2, 150, bw, bh,
               color=(50, 100, 50), hover_color=(70, 160, 70)),
        Button("Login",    cx - bw // 2, 240, bw, bh,
               color=(50, 50, 120), hover_color=(80, 80, 180)),
        Button("Quit",     cx - bw // 2, 330, bw, bh,
               color=(100, 40, 40), hover_color=(160, 60, 60)),
    ]
    return ButtonMenu("Face Auth System", buttons)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_reference_images():
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    paths = [
        os.path.join(REFERENCE_FOLDER, f)
        for f in os.listdir(REFERENCE_FOLDER)
        if f.lower().endswith(exts)
    ]
    return paths if paths else REFERENCE_IMAGES


def images_for_account(name):
    """Return only the reference images that belong to this account name."""
    all_images = load_reference_images()
    return [p for p in all_images if label_from_path(p).lower() == name.lower()]


def label_from_path(ref_path):
    base = os.path.splitext(os.path.basename(ref_path))[0]
    return base.replace("_glasses", "").replace("_angle", "").replace("_1", "")


def check_already_registered(face_img):
    ref_images = load_reference_images()
    if not ref_images:
        return False, None
    for ref in ref_images:
        try:
            r = DeepFace.verify(
                img1_path         = face_img,
                img2_path         = ref,
                model_name        = RECOGNITION_MODEL,
                detector_backend  = DETECTOR_BACKEND,
                enforce_detection = False,
                align             = True,
            )
            if r["verified"] and r["distance"] < DISTANCE_THRESHOLD:
                return True, label_from_path(ref)
        except Exception as e:
            print(f"  [Dup-check] error: {e}")
    return False, None


def scale_frame(frame):
    if FRAME_SCALE == 1.0:
        return frame
    w = int(frame.shape[1] * FRAME_SCALE)
    h = int(frame.shape[0] * FRAME_SCALE)
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


def detect_faces_haar(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    return faces if len(faces) > 0 else []


def pick_primary_face(faces, frame_w, frame_h):
    return max(faces, key=lambda f: f[2] * f[3])


def crop_face(frame, face, padding=20):
    x, y, w, h = face
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(frame.shape[1], x + w + padding)
    y2 = min(frame.shape[0], y + h + padding)
    return frame[y1:y2, x1:x2]


def draw_centered_text(frame, text, y, font_scale=0.9,
                        color=(255, 255, 255), thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (frame.shape[1] - tw) // 2
    cv2.putText(frame, text, (x+2, y+2), font, font_scale, (0,0,0), thickness+1)
    cv2.putText(frame, text, (x,   y  ), font, font_scale, color,   thickness)


def show_message_screen(lines, duration_ms=2500):
    screen = np.zeros((300, 640, 3), dtype="uint8")
    screen[:] = (30, 30, 30)
    y_start = 300 // 2 - (len(lines) - 1) * 30
    for i, (text, color) in enumerate(lines):
        draw_centered_text(screen, text, y_start + i * 50, 0.85, color)
    cv2.imshow("Face Auth", screen)
    cv2.waitKey(duration_ms)


def text_input_popup(prompt="Enter label:", max_chars=24, masked=False):
    """
    On-screen text entry popup.
    masked=True shows asterisks (for passwords).
    Returns entered string or None if cancelled.
    """
    cv2.setMouseCallback("Face Auth", lambda *a: None)
    text       = ""
    blink      = True
    last_blink = time.time()

    while True:
        screen = np.zeros((420, 640, 3), dtype="uint8")
        screen[:] = (30, 30, 30)

        px, py, pw, ph = 80, 130, 480, 160
        cv2.rectangle(screen, (px, py), (px+pw, py+ph), (55, 55, 55), -1)
        cv2.rectangle(screen, (px, py), (px+pw, py+ph), (120, 120, 120), 1)

        draw_centered_text(screen, prompt, py - 20, 0.8, (180, 180, 180))

        fx, fy, fw, fh = px + 20, py + 50, pw - 40, 50
        cv2.rectangle(screen, (fx, fy), (fx+fw, fy+fh), (20, 20, 20), -1)
        cv2.rectangle(screen, (fx, fy), (fx+fw, fy+fh), (160, 160, 160), 1)

        if time.time() - last_blink > 0.5:
            blink      = not blink
            last_blink = time.time()

        display_text = ("*" * len(text) if masked else text) + ("|" if blink else " ")
        cv2.putText(screen, display_text,
                    (fx + 10, fy + 33), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, (255, 255, 255), 2, cv2.LINE_AA)

        draw_centered_text(screen, "Enter = confirm     Esc = cancel",
                           py + ph + 40, 0.62, (100, 100, 100))

        cv2.imshow("Face Auth", screen)
        key = cv2.waitKey(30) & 0xFF

        if key == 13:
            return text.strip() if text.strip() else None
        elif key == 27:
            return None
        elif key in (8, 127):
            text = text[:-1]
        elif 32 <= key <= 126 and len(text) < max_chars:
            text += chr(key)


# ── Background verification worker ───────────────────────────────────────────

class FaceVerifier:
    def __init__(self):
        self.matched  = False
        self.ref_path = None
        self.busy     = False
        self._lock    = threading.Lock()

    def start_verify(self, face_crop, ref_images):
        if self.busy:
            return
        self.busy = True
        t = threading.Thread(
            target=self._run, args=(face_crop.copy(), ref_images), daemon=True
        )
        t.start()

    def _run(self, face_crop, ref_images):
        matched, ref_path = False, None
        for ref in ref_images:
            try:
                r = DeepFace.verify(
                    img1_path         = face_crop,
                    img2_path         = ref,
                    model_name        = RECOGNITION_MODEL,
                    detector_backend  = DETECTOR_BACKEND,
                    enforce_detection = False,
                    align             = True,
                )
                if r["verified"] and r["distance"] < DISTANCE_THRESHOLD:
                    matched, ref_path = True, ref
                    break
            except Exception as e:
                print(f"  DeepFace error: {e}")
        with self._lock:
            self.matched  = matched
            self.ref_path = ref_path
            self.busy     = False

    def reset(self):
        with self._lock:
            self.matched  = False
            self.ref_path = None

    def get_result(self):
        with self._lock:
            return self.matched, self.ref_path


# ── Registration ──────────────────────────────────────────────────────────────

def capture_snapshot(cap, label):
    """Returns (True, frame) on success, (False, None) if cancelled."""
    cv2.setMouseCallback("Face Auth", lambda *a: None)
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            return False, None

        elapsed       = time.time() - start_time
        remaining     = max(0, COUNTDOWN_SECONDS - int(elapsed))
        faces         = detect_faces_haar(frame)
        face_detected = len(faces) == 1
        display       = frame.copy()

        for (x, y, w, h) in faces:
            color = (0, 255, 0) if remaining == 0 else (0, 200, 255)
            cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)

        if len(faces) > 1:
            draw_centered_text(display, "Multiple faces — please register alone",
                               50, 0.8, (0, 60, 255))
            start_time = time.time()
        elif face_detected:
            msg = f"Hold still... {remaining}" if remaining > 0 else "Captured!"
            draw_centered_text(display, msg, 50, 1.1, (0, 220, 255))
        else:
            draw_centered_text(display, "No face detected — move closer",
                               50, 0.85, (0, 80, 255))
            start_time = time.time()

        draw_centered_text(display, f"Registering: {label}",
                           display.shape[0] - 50, 0.7, (180, 180, 180))
        draw_centered_text(display, "C = capture now   |   Q = cancel",
                           display.shape[0] - 20, 0.65, (120, 120, 120))
        cv2.imshow("Face Auth", display)
        key = cv2.waitKey(1) & 0xFF

        if (face_detected and elapsed >= COUNTDOWN_SECONDS) or \
           (key == ord("c") and face_detected):
            return True, frame
        if key == ord("q"):
            return False, None


def register_face(cap):
    # ── Step 1: name popup ────────────────────────────────────────────────────
    name = text_input_popup(prompt="Enter your name to register:")
    if not name:
        print("  Registration cancelled (no name entered).")
        return False, ""
    print(f"  Name entered: {name}")

    # ── Step 2: password popup ────────────────────────────────────────────────
    password = text_input_popup(prompt=f"Set a password for '{name}':", masked=True)
    if not password:
        print("  Registration cancelled (no password entered).")
        return False, ""

    confirm_pw = text_input_popup(prompt="Confirm your password:", masked=True)
    if confirm_pw != password:
        show_message_screen([
            ("Passwords do not match.", (0, 80, 255)),
            ("Registration cancelled.",  (180, 180, 180)),
        ], duration_ms=2500)
        return False, ""

    shots = [
        ("without glasses (or your normal look)", f"{name}_1.jpg"),
        ("with glasses  (skip with Q if N/A)",     f"{name}_glasses.jpg"),
        ("slight left/right angle  (skip with Q)", f"{name}_angle.jpg"),
    ]

    # ── Step 3: capture base photo + duplicate check ──────────────────────────
    description, filename = shots[0]
    save_path = os.path.join(REFERENCE_FOLDER, filename)
    print(f"\n  Shot: {description}")

    ok, base_frame = capture_snapshot(cap, f"{name} — {description}")
    if not ok:
        print("  Registration cancelled.")
        return False, name

    checking = np.zeros((300, 640, 3), dtype="uint8")
    checking[:] = (30, 30, 30)
    draw_centered_text(checking, "Checking for existing registration...",
                       150, 0.8, (200, 200, 100))
    cv2.imshow("Face Auth", checking)
    cv2.waitKey(1)

    already_registered, existing_label = check_already_registered(
        scale_frame(base_frame)
    )

    if already_registered:
        if existing_label.lower() != name.lower():
            print(f"\n[REGISTER] Blocked: face already registered as '{existing_label}'.")
            show_message_screen([
                (f"Face already registered as '{existing_label}'.", (0, 80, 255)),
                ("Cannot register under a different label.",         (180, 180, 180)),
            ], duration_ms=3000)
            return False, name
        else:
            print(f"\n[REGISTER] Updating registration for '{existing_label}'...")
            show_message_screen([
                (f"Updating registration for '{existing_label}'.", (0, 220, 255)),
            ], duration_ms=1500)

    # ── Step 4: save base photo + password ────────────────────────────────────
    cv2.imwrite(save_path, base_frame)
    save_password(name, password)
    print(f"  Saved: {filename}")

    confirm = np.zeros((200, 640, 3), dtype="uint8")
    draw_centered_text(confirm, f"Saved: {filename}", 100, 0.9, (0, 255, 120))
    cv2.imshow("Face Auth", confirm)
    cv2.waitKey(900)

    # ── Step 5: optional extra shots ─────────────────────────────────────────
    for description, filename in shots[1:]:
        save_path = os.path.join(REFERENCE_FOLDER, filename)
        print(f"\n  Shot: {description}")
        ok, frame = capture_snapshot(cap, f"{name} — {description}")
        if ok:
            cv2.imwrite(save_path, frame)
            print(f"  Saved: {filename}")
            confirm = np.zeros((200, 640, 3), dtype="uint8")
            draw_centered_text(confirm, f"Saved: {filename}", 100, 0.9, (0, 255, 120))
            cv2.imshow("Face Auth", confirm)
            cv2.waitKey(900)
        else:
            print(f"  Skipped: {filename}")

    print(f"\n[REGISTER] Done! Registered '{name}'.")
    return True, name


# ── Password login fallback ───────────────────────────────────────────────────

def password_login(account_name):
    """
    On-screen password login with MAX_PASSWORD_ATTEMPTS tries.
    Returns True on success, False on lockout.
    """
    for attempt in range(1, MAX_PASSWORD_ATTEMPTS + 1):
        remaining = MAX_PASSWORD_ATTEMPTS - attempt + 1
        pw = text_input_popup(
            prompt=f"Password for '{account_name}'  "
                   f"({remaining} attempt{'s' if remaining != 1 else ''} left):",
            masked=True
        )

        if pw is None:                        # user pressed Esc
            return False

        if check_password(account_name, pw):
            return True

        if attempt < MAX_PASSWORD_ATTEMPTS:
            show_message_screen([
                ("Wrong password.", (0, 80, 255)),
                (f"{MAX_PASSWORD_ATTEMPTS - attempt} attempt(s) remaining.",
                 (180, 180, 180)),
            ], duration_ms=1800)
        else:
            show_message_screen([
                ("Too many wrong attempts.", (0, 0, 200)),
                ("Account locked out.",      (180, 180, 180)),
            ], duration_ms=3000)

    return False


# ── Face login ────────────────────────────────────────────────────────────────

def login_face(cap, account_name, ref_images):
    """
    Clean state-machine face login:
      WAITING  → face not yet visible, show warning
      SCANNING → face found, DeepFace running in background
      RESULT   → check arrived: success → login / fail → count attempt

    Each full scan that comes back as "not this account" counts as one attempt.
    No-face frames never burn an attempt.
    After MAX_FACE_ATTEMPTS misses → password fallback.
    Returns True on successful login, False otherwise.
    """
    cv2.setMouseCallback("Face Auth", lambda *a: None)

    STATE_WAITING  = "waiting"
    STATE_SCANNING = "scanning"
    STATE_RESULT   = "result"

    verifier      = FaceVerifier()
    verify_count  = 0           # consecutive matched frames → confirms identity
    face_attempts = 0           # completed non-matching scans
    state         = STATE_WAITING
    last_matched  = False
    last_ref      = None

    print(f"\n[LOGIN] Scanning face for '{account_name}'...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display        = frame.copy()
        faces          = detect_faces_haar(frame)
        h, w           = frame.shape[:2]
        multiple_faces = len(faces) > 1
        has_face       = len(faces) > 0

        # ── Draw face boxes ───────────────────────────────────────────────────
        if multiple_faces:
            primary = pick_primary_face(faces, w, h)
            px, py, pw2, ph = primary
            for (x, y, fw, fh) in faces:
                is_primary = (x == px and y == py)
                cv2.rectangle(display, (x, y), (x+fw, y+fh),
                              (0, 255, 0) if is_primary else (0, 0, 255), 2)
                if not is_primary:
                    cv2.putText(display, "Not verified", (x, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
        elif has_face:
            primary = pick_primary_face(faces, w, h)
            px, py, pw2, ph = primary
            cv2.rectangle(display, (px, py), (px+pw2, py+ph),
                          (0, 255, 0) if last_matched else (0, 0, 255), 2)

        # ── State machine ─────────────────────────────────────────────────────
        if state == STATE_WAITING:
            # Waiting for a face to appear
            if not has_face:
                verify_count = 0
                draw_centered_text(display, "No face detected — move closer",
                                   50, 0.85, (0, 80, 255))
            elif multiple_faces:
                verify_count = 0
                draw_centered_text(display,
                                   "Multiple faces — only closest will be verified",
                                   50, 0.7, (0, 140, 255))
            else:
                # Single face appeared — kick off a scan
                verifier.reset()
                verifier.start_verify(
                    scale_frame(crop_face(frame, primary)), ref_images
                )
                state = STATE_SCANNING
                draw_centered_text(display, "Scanning...", 50, 0.9, (0, 200, 255))

        elif state == STATE_SCANNING:
            # DeepFace is running — keep displaying live feed
            if not has_face or multiple_faces:
                # Face lost mid-scan — abort and go back to waiting
                verifier.reset()
                verify_count = 0
                state = STATE_WAITING
                if not has_face:
                    draw_centered_text(display, "No face detected — move closer",
                                       50, 0.85, (0, 80, 255))
                else:
                    draw_centered_text(display,
                                       "Multiple faces — only closest will be verified",
                                       50, 0.7, (0, 140, 255))
            elif not verifier.busy:
                # Scan finished — move to result
                last_matched, last_ref = verifier.get_result()
                state = STATE_RESULT
                draw_centered_text(display, "Checking...", 50, 0.9, (0, 200, 255))
            else:
                draw_centered_text(display, "Scanning...", 50, 0.9, (0, 200, 255))

        elif state == STATE_RESULT:
            if last_matched:
                # ── Match: need VERIFY_THRESHOLD consecutive confirmations ──
                verify_count += 1
                draw_centered_text(display,
                                   f"Match!  Confirming {verify_count}/{VERIFY_THRESHOLD}",
                                   50, 0.9, (0, 255, 120))

                if verify_count >= VERIFY_THRESHOLD:
                    perform_login(display, account_name)
                    return True

                # Fire another quick scan to accumulate confirmations
                verifier.reset()
                verifier.start_verify(
                    scale_frame(crop_face(frame, primary)), ref_images
                )
                state = STATE_SCANNING

            else:
                # ── No match: count as one failed attempt ─────────────────
                face_attempts += 1
                verify_count   = 0
                print(f"  Face did not match '{account_name}' — "
                      f"attempt {face_attempts}/{MAX_FACE_ATTEMPTS}")

                if face_attempts >= MAX_FACE_ATTEMPTS:
                    show_message_screen([
                        (f"Face does not match account '{account_name}'.", (0, 80, 255)),
                        ("3 attempts used. Switching to password login.", (180, 180, 180)),
                    ], duration_ms=2800)
                    return password_login(account_name)

                # Show failure message briefly, then go back to waiting
                remaining = MAX_FACE_ATTEMPTS - face_attempts
                show_message_screen([
                    ("Face does not match.", (0, 80, 255)),
                    (f"{remaining} attempt{'s' if remaining != 1 else ''} remaining."
                     "  Please try again.", (180, 180, 180)),
                ], duration_ms=1800)
                state = STATE_WAITING

        # ── HUD ──────────────────────────────────────────────────────────────
        remaining_attempts = MAX_FACE_ATTEMPTS - face_attempts
        draw_centered_text(display,
                           f"Logging in as: {account_name}",
                           display.shape[0] - 50, 0.65, (120, 200, 120))
        draw_centered_text(display,
                           f"Face attempts remaining: {remaining_attempts}  |  Q = cancel",
                           display.shape[0] - 20, 0.62, (160, 160, 160))

        cv2.imshow("Face Auth", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

    return False


# ── Login entry point (asks for account name first) ───────────────────────────

def start_login(cap):
    # ── Step 1: ask for account name ─────────────────────────────────────────
    account_name = text_input_popup(prompt="Enter your account name to log in:")
    if not account_name:
        return

    # ── Step 2: validate account exists ──────────────────────────────────────
    if not account_exists(account_name):
        show_message_screen([
            (f"Account '{account_name}' not found.", (0, 80, 255)),
            ("Please register first.",               (180, 180, 180)),
        ], duration_ms=2500)
        return

    # ── Step 3: load only this account's reference images ────────────────────
    ref_images = images_for_account(account_name)
    if not ref_images:
        show_message_screen([
            (f"No face images found for '{account_name}'.", (0, 80, 255)),
            ("Please register again.",                       (180, 180, 180)),
        ], duration_ms=2500)
        return

    print(f"\n[LOGIN] Account '{account_name}' — "
          f"{len(ref_images)} reference image(s) loaded.")

    # ── Step 4: face scan → password fallback if needed ──────────────────────
    login_face(cap, account_name, ref_images)


# ── Post-login ────────────────────────────────────────────────────────────────

def perform_login(frame, name):
    print(f"\nLogged in as '{name}'")
    success = frame.copy()
    draw_centered_text(success, f"Welcome, {name}!",
                       success.shape[0] // 2, 1.3, (0, 255, 120))
    cv2.imshow("Face Auth", success)
    cv2.waitKey(2000)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow("Face Auth")
    menu = make_main_menu()

    try:
        while True:
            choice = menu.run()

            if choice == "Register":
                ok, _ = register_face(cap)
                if ok:
                    cv2.waitKey(500)
                    start_login(cap)
                    break

            elif choice == "Login":
                start_login(cap)
                break

            elif choice == "Quit":
                cap.release()
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                exit()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


if __name__ == "__main__":
    main()