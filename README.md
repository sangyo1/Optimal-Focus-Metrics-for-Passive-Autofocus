# Focus Monitor Demo Application
This application demonstrates various focus metrics applied to a live webcam feed. It displays the original image alongside multiple processed views using different focus metrics, highlighting the Region of Interest (ROI) used for calculating focus values.

##Focus Metrics Explained
The application showcases several focus metrics applied to the live webcam feed. Each metric processes the image differently to evaluate focus. Below is a brief explanation of each metric:

###Variance of Sobel
    - Applies the Sobel operator to detect edges.
    - Calculates the variance of the Sobel response within the ROI.
    - Higher variance indicates sharper edges and better focus.
    
###Squared Gradient
    - Computes the squared differences of adjacent pixels (gradient).
    - Measures the variance of the squared gradient in the ROI.
    - Sensitive to changes in intensity, highlighting focused areas.

###Squared Sobel
    - Similar to the Variance of Sobel but squares the Sobel responses.
    - Emphasizes stronger edges in the focus measurement.

###FSWM (Frequency Selective Weighted Mean)
    - Applies a bandpass filter using the Difference of Gaussians (DoG).
    - Focus value is the variance of the bandpass-filtered image in the ROI.
    - Captures mid-frequency details associated with focus.

###FFT (Fast Fourier Transform)
    - Computes the frequency spectrum of the image.
    - Focus value is derived from the variance of the magnitude spectrum in the ROI.
    - Analyzes high-frequency content indicative of sharpness.

###Mix Sobel

    - Combines gradient magnitude and diagonal Sobel responses.
    - Provides a comprehensive edge detection for focus evaluation.

###Sobel+Laplacian
    - Combines Sobel edge detection with the Laplacian operator.
    - Enhances both edge and texture information for focus measurement.

###Combined Focus Measure
    - Integrates Sobel-based and FSWM metrics.
    - Offers a balanced focus measure sensitive to various image features.

###Combined Focus Measure 2
    - Merges Sobel-based focus with FFT-based focus.
    - Leverages both spatial and frequency domain information.

Each metric view displays the processed image with the ROI highlighted by a white rectangle. The focus value and metric name are displayed in the top-left corner.

#Robot Motion Planning
'''python3 demo.py''' will run the file
~/inspection_robot_service/scripts/robot_service.py is the robot motion planning for UR5e. This contains manipulator focus algorithm to find the maxmimum focus value position and take the image.


