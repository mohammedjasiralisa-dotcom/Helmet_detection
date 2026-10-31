import re
import pytesseract
import cv2
from PIL import Image
from ultralytics import YOLO

# Load YOLO model for helmet detection
model = YOLO("yolov8n.pt")  # Replace with actual model path
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
orb = cv2.ORB_create()

# Load helmet reference image for ORB matching
helmet_reference = cv2.imread("helmet.png", cv2.IMREAD_GRAYSCALE)
helmet_reference = cv2.resize(helmet_reference, (100, 100))

# Initialize video capture
cap = cv2.VideoCapture(0)

# Detection history for smoothing results
detection_history = []

# Define the expected license number
expected_license_number = " N63Z20240001148"

def detect_license():
    """Extract license number from an image and validate it."""
    global expected_license_number

    # Load the image
    image_path = "licience.png"  # Replace with the correct path
    image = Image.open(image_path)

    # Perform OCR
    extracted_text = pytesseract.image_to_string(image)

    # Print extracted text for debugging
    print("OCR Extracted Text:\n", extracted_text)

    # Use regex to extract the license number
    match = re.search(r'N\d+[A-Z0-9]+', extracted_text)

    if match:
        extracted_license_number = match.group().strip()  # Remove extra spaces
        print("Extracted License Number:", extracted_license_number)

        # Compare extracted number with expected number
        if extracted_license_number == expected_license_number:
            print("✅ License Approved")
            detect_helmet()  # Start helmet detection if license is valid
        else:
            print("❌ License Rejected: Number Mismatch")
    else:
        print("❌ License Number Not Found")


def detect_helmet():
    """Detect helmet using YOLO and ORB feature matching."""
    global detection_history
    ret, frame = cap.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    helmet_detected = False

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_region = frame[y:y + h, x:x + w]

        results = model(face_region)
        for obj in results[0].boxes.data:
            class_id = int(obj[5])
            confidence = float(obj[4])
            if class_id == 0 and confidence > 0.6:
                helmet_detected = True
                break

        face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (100, 100))

        kp1, des1 = orb.detectAndCompute(face_resized, None)
        kp2, des2 = orb.detectAndCompute(helmet_reference, None)

        if des1 is not None and des2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            if len(matches) > 15:
                helmet_detected = True

    detection_history.append(helmet_detected)
    stable_output = sum(detection_history) > (len(detection_history) / 2)

    if stable_output:
        print("Helmet Detected ✅")
    else:
        print("No Helmet ❌")

    # Show live video feed
    cv2.imshow("Helmet Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        return

    detect_helmet()  # Recursively call to keep detecting


if __name__ == "__main__":
    detect_license()  # Start with license detection
