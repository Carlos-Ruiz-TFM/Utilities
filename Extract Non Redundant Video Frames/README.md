This script applies non redundant frame extraction by measuring the pixel-wise change between a previously saved frame and the current one to determine if enough "new information" exists to save the next non redundant frame.

When starting from the first frame, it is saved as the initial reference frame. For each subsequent frame, the script converts both the current frame and the reference frame to grayscale and calculates the pixel-wise difference between them.

The difference is measured by calculating the mean absolute error (MAE) between the two grayscale frames. If the MAE exceeds a predefined threshold, the current frame is considered non redundant and is saved.