import cv2
import numpy as np

class FocusMonitor:

    def __init__(self, metric='fswm'):
        # ROI is set to a 100x100 region at the center
        self.cx = 0.5 
        self.cy = 0.5 
        self.w = 100 
        self.h = 100 
        self.metric = metric

        # Mapping dictionary to hold focus metric functions
        self.dict = {}

    @staticmethod
    def get_metrics():
        return [
            'Variance of Sobel', 'Squared Gradient', 'Squared Sobel', 
            'FSWM', 'FFT', 'Mix Sobel', 'Sobel+Laplacian', 
            'Sobel+FSWM', 'Sobel+FFT'
        ]

    def set_metric(self, name):
        name_to_metric = {
            'Variance of Sobel': 'sobel',
            'Squared Gradient': 'squared_gradient',
            'Squared Sobel': 'squared_sobel',
            'FSWM': 'fswm',
            'FFT': 'fft',
            'Mix Sobel': 'mix_sobel',
            'Sobel+Laplacian': 'sobel_laplacian',
            'Sobel+FSWM': 'combined_focus_measure',
            'Sobel+FFT': 'combined_focus_measure2'
        }
        if name in name_to_metric:
            self.metric = name_to_metric[name]
        else:
            raise ValueError(f"Unknown metric name: {name}")

    def measure_focus(self, image_in):
        if self.metric == 'sobel':
            focus_value, focus_image = self.sobel(image_in)
        elif self.metric == 'squared_gradient':
            focus_value, focus_image = self.squared_gradient(image_in)
        elif self.metric == 'squared_sobel':
            focus_value, focus_image = self.squared_sobel(image_in)
        elif self.metric == 'fswm':
            focus_value, focus_image = self.fswm(image_in)
        elif self.metric == 'fft':
            focus_value, focus_image = self.fft(image_in)
        elif self.metric == 'mix_sobel':
            focus_value, focus_image = self.mix_sobel(image_in)
        elif self.metric == 'sobel_laplacian':
            focus_value, focus_image = self.sobel_laplacian(image_in)
        elif self.metric == 'combined_focus_measure':
            focus_value, focus_image = self.combined_focus_measure(image_in)
        elif self.metric == 'combined_focus_measure2':
            focus_value, focus_image = self.combined_focus_measure2(image_in)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        return focus_value, focus_image

    def calculate_roi(self, image_in):
        # Calculate ROI coordinates (100x100 centered on the image)
        height, width, _ = image_in.shape
        x0 = max(0, int(self.cx * width - self.w / 2))
        y0 = max(0, int(self.cy * height - self.h / 2))
        x1 = min(width, x0 + self.w)
        y1 = min(height, y0 + self.h)
        return x0, y0, x1, y1

    def sobel(self, image_in):
        x0, y0, x1, y1 = self.calculate_roi(image_in)
        gray = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
        sobel_image = cv2.Sobel(gray, ddepth=cv2.CV_16S, dx=1, dy=1, ksize=3)
        sobel_value = sobel_image[y0:y1, x0:x1].var()
        image_out = sobel_image[y0:y1, x0:x1]
        image_out = cv2.convertScaleAbs(sobel_image)
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2RGB)
        return sobel_value, image_out

    def squared_gradient(self, image_in):
        x0, y0, x1, y1 = self.calculate_roi(image_in)

        # Convert to grayscale
        gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)

        # Compute the squared differences of adjacent pixels in both directions
        gradient_x = np.diff(gray_image, axis=0)
        gradient_y = np.diff(gray_image, axis=1)
        squared_gradient_x = gradient_x**2
        squared_gradient_y = gradient_y**2

        # # Adjust the shapes to be compatible for addition
        min_height = min(
            squared_gradient_x.shape[0], squared_gradient_y.shape[0])
        min_width = min(
            squared_gradient_x.shape[1], squared_gradient_y.shape[1])

        squared_gradient_x = squared_gradient_x[:min_height, :min_width]
        squared_gradient_y = squared_gradient_y[:min_height, :min_width]

        # ((np.var(squared_gradient_x[y0:y1, x0:x1]) + np.mean(squared_gradient_y[y0:y1, x0:x1]))**1.5)/2
        focus_value = np.var(squared_gradient_x[y0:y1, x0:x1])

        combined_gradient = np.sqrt(
            squared_gradient_x+squared_gradient_y).astype(np.float32)
        normalized_image = cv2.normalize(
            combined_gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_out = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)

        return focus_value, image_out

    def squared_sobel(self, image_in):
        x0, y0, x1, y1 = self.calculate_roi(image_in)

        gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)

        # Compute the squared differences of adjacent pixels in both directions using Sobel
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # np.var(smoothed_combined_gradient[yl:yh, xl:xh]) #+ np.mean(smoothed_combined_gradient[yl:yh, xl:xh])**1.5
        focus_value = np.var(gradient_magnitude[y0:y1, x0:x1])
        
        normalized_image = cv2.normalize(
            gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_out = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)
        return focus_value, image_out

    def fswm(self, image_in):
        x0, y0, x1, y1 = self.calculate_roi(image_in)

        gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)

        # Apply a bandpass filter using Difference of Gaussians (DoG)
        sigma_low = 2.5
        sigma_high = 3.0
        blur_low = cv2.GaussianBlur(gray_image, (0, 0), sigmaX=sigma_low)
        blur_high = cv2.GaussianBlur(gray_image, (0, 0), sigmaX=sigma_high)
        bandpass = blur_low - blur_high

        # Create a weight matrix
        rows, cols = bandpass.shape
        center_y, center_x = rows // 2, cols // 2
        Y, X = np.ogrid[:rows, :cols]
        distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_distance = np.max(distance)
        weights = 1 - (distance / max_distance)  # Weights decrease with distance from center

        # Compute the weighted mean
        weighted_bandpass = bandpass * weights
        focus_value = np.var(bandpass[y0:y1, x0:x1])

        # For visualization, normalize the weighted bandpass image
        bandpass_normalized = cv2.normalize(weighted_bandpass, None, 0, 255, cv2.NORM_MINMAX)
        image_out = cv2.cvtColor(bandpass_normalized.astype(np.uint8), cv2.COLOR_GRAY2RGB)


        return focus_value, image_out

    def fft(self, image_in):
        x0, y0, x1, y1 = self.calculate_roi(image_in)
        
        gray = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
        # Apply a window function to reduce edge effects
        window = np.hanning(gray.shape[0])[:, None] * np.hanning(gray.shape[1])[None, :]
        gray_windowed = gray * window
        # Compute the FFT
        f = np.fft.fft2(gray_windowed)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift)
        
        # Ground zero low frequencies
        center_y, center_x = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
        low_freq_size = 10 
        magnitude_spectrum[center_y - low_freq_size:center_y + low_freq_size,
                    center_x - low_freq_size:center_x + low_freq_size] = 0
        # Focus measure: sum of magnitude spectrum values
        focus_value = np.var(magnitude_spectrum[y0:y1, x0:x1])
        magnitude_spectrum_log = 20 * np.log1p(magnitude_spectrum)
        image_out = cv2.normalize(magnitude_spectrum_log, None, 0, 255, cv2.NORM_MINMAX)
        image_out = cv2.cvtColor(image_out.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        return focus_value, image_out
    
    def mix_sobel(self, image_in): 
        x0, y0, x1, y1 = self.calculate_roi(image_in)
        gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)

        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        sobel_xy = cv2.Sobel(gray_image, cv2.CV_64F, 1, 1, ksize=3)
        combined_gradients = gradient_magnitude + np.abs(sobel_xy)
        focus_value = np.var(combined_gradients[y0:y1, x0:x1])
        
        normalized_image = cv2.normalize(
            combined_gradients, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_out = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)
        return focus_value, image_out        

    def sobel_laplacian(self, image_in):
        x0, y0, x1, y1 = self.calculate_roi(image_in)

        # Convert to grayscale
        gray = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)

        # Apply Sobel filter
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # Apply Laplacian filter
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Combine Sobel and Laplacian
        combined = sobel_magnitude + np.abs(laplacian)

        # Compute focus value
        focus_value = np.var(combined[y0:y1, x0:x1])

        # Normalize for visualization
        normalized_image = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_out = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)
        return focus_value, image_out
    
    def combined_focus_measure(self, image_in):
        x0, y0, x1, y1 = self.calculate_roi(image_in)
        gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)

        #sobel
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        sobel_xy = cv2.Sobel(gray_image, cv2.CV_64F, 1, 1, ksize=3)
        combined_gradients = gradient_magnitude + np.abs(sobel_xy)
        sobel_var = np.var(combined_gradients[y0:y1, x0:x1])

        #fswm
        sigma_low = 2.5
        sigma_high = 3.0
        blur_low = cv2.GaussianBlur(gray_image, (0, 0), sigmaX=sigma_low)
        blur_high = cv2.GaussianBlur(gray_image, (0, 0), sigmaX=sigma_high)
        bandpass = blur_low - blur_high        
        fswm_var = np.var(bandpass[y0:y1, x0:x1]) 
        
        focus_value = sobel_var + 0.5*(fswm_var**0.75)
        normalized_image = cv2.normalize(
            combined_gradients, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_out = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)

        return focus_value, image_out 
        
    def combined_focus_measure2(self, image_in):
        x0, y0, x1, y1 = self.calculate_roi(image_in)
        gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)

        #Sobel-based focus value
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        sobel_xy = cv2.Sobel(gray_image, cv2.CV_64F, 1, 1, ksize=3)
        combined_gradients = gradient_magnitude + np.abs(sobel_xy)
        sobel_var = np.var(combined_gradients[y0:y1, x0:x1])

        #Compute FFT-based focus value
        window = np.hanning(gray_image.shape[0])[:, None] * np.hanning(gray_image.shape[1])[None, :]
        gray_windowed = gray_image * window

        f = np.fft.fft2(gray_windowed)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift)

        center_y, center_x = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
        low_freq_size = 10 
        magnitude_spectrum[center_y - low_freq_size:center_y + low_freq_size,
                           center_x - low_freq_size:center_x + low_freq_size] = 0

        fft_var = np.var(magnitude_spectrum[y0:y1, x0:x1])

        focus_value = sobel_var + (0.5*fft_var/(1e5))

        # magnitude_spectrum_log = 20 * np.log1p(magnitude_spectrum)
        normalized_image = cv2.normalize(combined_gradients, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_out = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)

        return focus_value, image_out 