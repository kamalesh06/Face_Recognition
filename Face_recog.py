import os
import pickle
import cv2
from deepface import DeepFace
import numpy as np

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Paths for saving model and data
embeddings_path = './face_embeddings.pkl'
labels_path = './face_labels.pkl'

RADIUS = 5
DISTANCE = 20

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2

    # Draw the bordered corners around the rectangle
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

# Load embeddings and labels
def load_data():
    if os.path.exists(embeddings_path) and os.path.exists(labels_path):
        with open(embeddings_path, 'rb') as f_emb, open(labels_path, 'rb') as f_lbl:
            known_face_encodings = pickle.load(f_emb)
            known_face_names = pickle.load(f_lbl)
        print("Embeddings and labels loaded successfully.")
    else:
        known_face_encodings, known_face_names = [], []
        print("No saved embeddings or labels found. Starting fresh.")
    
    return known_face_encodings, known_face_names

# Save embeddings and labels
def save_data(known_face_encodings, known_face_names):
    with open(embeddings_path, 'wb') as f_emb, open(labels_path, 'wb') as f_lbl:
        pickle.dump(known_face_encodings, f_emb)
        pickle.dump(known_face_names, f_lbl)
    print("Embeddings and labels saved successfully.")

# Compute cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Load known embeddings and labels
known_face_encodings, known_face_names = load_data()

# Initialize video capture
cap = cv2.VideoCapture(0)

# Check if the video capture opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Minimum size for detected faces
min_face_size = (50, 50)

unknown_faces = []
selected_index = None

scale_percent = 50  # Reduce to 50% of original size

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to grayscale for detection
    small_frame = cv2.resize(frame, (0, 0), fx = scale_percent / 100, fy = scale_percent / 100)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=min_face_size)

    unknown_faces.clear()  # Clear the unknown faces list for each frame

    for (x, y, w, h) in faces:
        x *= 100 // scale_percent
        y *= 100 // scale_percent
        w *= 100 // scale_percent
        h *= 100 // scale_percent
        
        # Extract the face for encoding
        face_image = frame[y:y+h, x:x+w]

        # Encode the face using DeepFace
        face_encoding = DeepFace.represent(face_image, enforce_detection=False)[0]['embedding']

        name = "Unknown"
        max_similarity = 0.7  # Similarity threshold

        for known_encoding, known_name in zip(known_face_encodings, known_face_names):
            similarity = cosine_similarity(face_encoding, known_encoding)
            if similarity > max_similarity:
                name = known_name
                break

        # Draw rectangle with color logic for specific face selection
        if name != "Unknown":
            color = (0, 255, 0)  # Green for known faces
        else:
            color = (0, 0, 255)  # Red for unknown faces
            unknown_faces.append((x, y, w, h))  # Store only unknown face coordinates

        # Draw a border around each face
        draw_border(frame, (x, y), (x + w, y + h), color, 3, RADIUS, DISTANCE)

        # Label the face
        if name == "Unknown":
            label_text = f"{name} - ({len(unknown_faces) - 1})"  # Show index only for unknown faces
            cv2.putText(frame, label_text, (x + 50, y + h - 6), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 2)
        else:
            cv2.putText(frame,name, (x + 75, y + h - 6), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 2)
            
            # Check if the user wants to change the label
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('c'):  # Press 'c' to change label
                print(f"Detected {name}.",end = ' ')

                new_label = input("New label: ")

                if new_label:
                    # Change the label for all occurrences of this known face
                    for i, known_name in enumerate(known_face_names):
                        if known_name == name:
                            known_face_names[i] = new_label 
                    save_data(known_face_encodings, known_face_names)
                    
        # Highlight the selected face
        if selected_index is not None and selected_index < len(unknown_faces):
            x, y, w, h = unknown_faces[selected_index]
            draw_border(frame, (x, y), (x + w, y + h), (255, 0, 0), 3, RADIUS, DISTANCE)  # Blue for selected face

    # Display the frame
    cv2.imshow('Live Face Recognition', frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):  # Press 'q' to exit
        break
    
    # Select the unknown face
    elif ord('0') <= key <= ord('9'):
        selected_index = key - ord('0')

    elif key == ord('d'): # Press 'd' to unselect
        selected_index = None

    elif key == ord('s') and selected_index is not None: # Press 's' to label the selected unknown face
        if selected_index < len(unknown_faces):  # Check if the index is valid

            # Label the selected face with user input
            x, y, w, h = unknown_faces[selected_index]
            face_image = frame[y : y+h, x : x+w]

            new_label = input("Enter a label for the selected face: ")
            
            if new_label:
                face_encoding = DeepFace.represent(face_image, enforce_detection=False)[0]['embedding']
                known_face_encodings.append(face_encoding)
                known_face_names.append(new_label)
                save_data(known_face_encodings, known_face_names)

            selected_index = None

# Release resources
cap.release()
cv2.destroyAllWindows()