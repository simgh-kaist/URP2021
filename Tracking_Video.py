import numpy as np
import cv2
# ref: https://www.learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/
# NO GOOD!!!!
# Read video
cap = cv2.VideoCapture('20201201_001346.mp4')
# Get video information
n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Define codec for output video
fourcc=cv2.VideoWriter_fourcc(*'MJPG')
# Set up output video
fps=24
out=cv2.VideoWriter('tracked.mp4',fourcc,fps,(w,h))

"""
for i in range(n_frames):
    success, frame = cap.read()
    out.write(frame)
"""
# Read first frame
_, prev = cap.read()
prev_gray=cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)

# Pre-define transformation-store array
transforms = np.zeros((n_frames-1,3), np.float32)

for i in range(n_frames-2):
    # Detect feature points in previous frame
    prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                       maxCorners=200,
                                       qualityLevel=0.01,
                                       minDistance=30,
                                       blockSize=3)
    # if best quality is 150, quality less than 150*qualitiLevel are rejected.
    success, curr = cap.read()  #read next frame
    if not success:
        break
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    # Calculate Optical Flow (i.e. tarck feature points)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, nextPts=None)
    # Sanity Check (Optional)
    assert prev_pts.shape == curr_pts.shape
    # Filter valid points
    idx = np.where(status==1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]
    # Find transformation matrix
    m, inliers = cv2.estimateAffinePartial2D(prev_pts, curr_pts)     #alternative estimateRigidTransform of OpenCV3 or less
    # Extract ranslation
    dx = m[0,2]
    dy = m[1,2]
    # Extract rotation angle
    da = np.arctan2(m[1,0],m[0,0])
    # Store transformation
    transforms[i] = [dx,dy,da]
    # Move to next frame
    prev_gray = curr_gray

# Compute trajectory using cumulative sum of transformations
trajectory = np.cumsum(transforms, axis=0)

def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size)/window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    return curve_smoothed


def smooth(trajactory):
    smoothed_trajectory = np.copy(trajectory)
    SMOOTHING_RADIUS = 50
    # Filter the x, y, angle curves
    for i in range(3):
        smoothed_trajectory[:,i] = movingAverage(trajactory[:,i], radius=SMOOTHING_RADIUS)
    return smoothed_trajectory

# Final trajectory
trajectory = np.cumsum(transforms, axis=0)
smoothed_trajectory = smooth(trajectory)
# Calculate difference in smoothed_trajectory and trajectory
difference = smoothed_trajectory - trajectory
# Calculate newer transformation array
transforms_smooth = transforms + difference

# Reset straem to first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

def fixBorder(frame):
  s = frame.shape
  # Scale the image 4% without moving the center
  T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
  frame = cv2.warpAffine(frame, T, (s[1], s[0]))
  return frame
#Write n_frames-1 transformed frames

for i in range(n_frames - 2):
    # Read next frame
    success, frame = cap.read()
    if not success:
        break

    # Extract transformations from the new transformation array
    dx = transforms_smooth[i, 0]
    dy = transforms_smooth[i, 1]
    da = transforms_smooth[i, 2]
    # Reconstruct transformation matrix accordingly to new values
    m = np.zeros((2, 3), np.float32)
    m[0, 0] = np.cos(da)
    m[0, 1] = -np.sin(da)
    m[1, 0] = np.sin(da)
    m[1, 1] = np.cos(da)
    m[0, 2] = dx
    m[1, 2] = dy
    # Apply affine wrapping to the given frame
    frame_stabilized = cv2.warpAffine(frame, m, (w, h))
    # Fix border artifacts
    frame_stabilized = fixBorder(frame_stabilized)
    # Write the frame to the file
    frame_out = cv2.hconcat([frame, frame_stabilized])
    # If the image is too big, resize it.
    #if (frame_out.shape[1] & gt > 1920):
    #    frame_out = cv2.resize(frame_out, (frame_out.shape[1] / 2, frame_out.shape[0] / 2));
    #cv2.imshow("Before and After", frame_out)
    #cv2.waitKey(10)
    #out.write(frame_out)
    out.write(frame_stabilized)

out.release()
#cv2.imshow("Before and After", frame_out)
#cv2.waitKey(10)

print('end')
#out.release()
