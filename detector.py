import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# Initialize MTCNN and InceptionResnetV1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Directory for storing the dataset
dataset_dir = 'dataset'
os.makedirs(dataset_dir, exist_ok=True)

def capture_images(user_id, num_samples=50):  # Increased number of samples
    user_dir = os.path.join(dataset_dir, str(user_id))
    os.makedirs(user_dir, exist_ok=True)

    cap = cv2.VideoCapture(700)  # Use 0 for the primary camera
    sample_num = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    while sample_num < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        boxes, _ = mtcnn.detect(Image.fromarray(frame_rgb))

        # Check if boxes are detected
        if boxes is not None:
            for box in boxes:
                # Draw rectangle around detected face
                x1, y1, x2, y2 = [int(b) for b in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Save the cropped face image
                face = frame[y1:y2, x1:x2]
                face_path = os.path.join(user_dir, f'{user_id}_{sample_num}.jpg')
                cv2.imwrite(face_path, face)
                sample_num += 1

        cv2.imshow('Capture Images', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def create_embeddings():
    embeddings = {}
    for user in os.listdir(dataset_dir):
        user_dir = os.path.join(dataset_dir, user)
        if os.path.isdir(user_dir):
            user_embeddings = []
            for image_name in os.listdir(user_dir):
                image_path = os.path.join(user_dir, image_name)
                img = Image.open(image_path).convert('RGB')
                img_cropped = mtcnn(img)

                # Check if faces are detected
                if img_cropped is not None and len(img_cropped) > 0:
                    # Get the embedding and ensure it's a single array
                    embedding = model(img_cropped.to(device)).detach().cpu().numpy()
                    user_embeddings.append(embedding.mean(axis=0))  # Expecting (n, 512) shape

            if user_embeddings:
                embeddings[user] = np.mean(user_embeddings, axis=0)
    return embeddings

def recognize_faces(embeddings, threshold=0.8):  # Set a similarity threshold
    cap = cv2.VideoCapture(700)  # Use 0 for the primary camera
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        boxes, _ = mtcnn.detect(Image.fromarray(frame_rgb))

        # Check if boxes are detected
        if boxes is not None:
            for box in boxes:
                # Draw rectangle around detected face
                x1, y1, x2, y2 = [int(b) for b in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Get the face for recognition
                face = frame[y1:y2, x1:x2]
                face_pil = Image.fromarray(face)

                # Detect and crop the face
                face_cropped = mtcnn(face_pil)

                # Check if faces are detected after cropping
                if face_cropped is not None and len(face_cropped) > 0:
                    # Get the embedding
                    embedding = model(face_cropped.to(device)).detach().cpu().numpy()

                    if embedding.ndim == 2:
                        # Compare with stored embeddings
                        distances = {user: np.linalg.norm(embedding.mean(axis=0) - emb) for user, emb in embeddings.items()}
                        recognized_user = min(distances, key=distances.get)

                        # Only recognize if below threshold
                        if distances[recognized_user] < threshold:
                            cv2.putText(frame, recognized_user, (x1, y1 - 10), font, 0.8, (255, 255, 255), 2)
                        else:
                            cv2.putText(frame, "Unknown", (x1, y1 - 10), font, 0.8, (255, 0, 0), 2)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    user_id = input("Enter user ID to capture images: ")
    capture_images(user_id)  # Capture images for a new user
    print("Creating embeddings...")
    user_embeddings = create_embeddings()
    print("Embeddings created. Starting face recognition...")
    recognize_faces(user_embeddings)
