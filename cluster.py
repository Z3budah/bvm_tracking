import cv2

# Load the video
cap = cv2.VideoCapture('../keypoint/test.mp4')

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# Read and process each frame of the video
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        # Convert the frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert the grayscale frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        # Write the frame to the output video
        out.write(frame)

        # Display the frame
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture and writer objects
cap.release()
out.release()

# Close all windows
cv2.destroyAllWindows()


from models.tsn import TSN

num_classes = 400
model = TSN(num_classes, 3, 'RGB', base_model='resnet50', consensus_type='avg', dropout=0.5)
model.load_state_dict(torch.load('path/to/model.pth'))

