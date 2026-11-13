% =========================================================================
% MATLAB Script: Choline Basis Set Processing & Comparison
% =========================================================================

% --- 1. Clean and Prepare Data ---
fid_data   = squeeze(outA.fids);
spec_data  = squeeze(outA.specs);
time_vec   = squeeze(outA.t) * 1000; % Convert to milliseconds
ppm_vec    = squeeze(outA.ppm);
dt         = 2.500000000000000e-04;  % Dwell time
N          = length(fid_data);
t          = squeeze(outA.t);          

% Ensure everything is strictly a column vector to prevent the legend warning
fid_data   = 5.40013*fid_data(:);
spec_data  = spec_data(:);
time_vec   = time_vec(:);
ppm_vec    = ppm_vec(:);
t          = t(:);

% Set up matching time vector for decay calculation
% Apply exponential decay element-wise
alpha      = 0; 
fid_data_decayed = fid_data .* exp(-alpha * t);

% --- 2. Spectral Calculations & Corrections ---
magnitude_spectrum = (spec_data);

% Using pure ifftshift(ifft) as requested
calculated_fft     = ifftshift(ifft(fid_data_decayed));
magnitude_spectrum = (calculated_fft);

% CONFIGURATION TOGGLE: Set to true if the manual FFT peak appears 
% mirrored/flipped relative to the outA.specs reference peak.
FLIP_SPECTRUM = false; 
if FLIP_SPECTRUM
    calculated_fft = flipud(calculated_fft);
end

magnitude_calc_fft = abs(calculated_fft);
% --- 3. Area Under the Peak (Frequency Domain Integration) ---
% Define the ppm limits for the Choline peak (typically centered around 3.2 ppm)
ppm_min = 0;
ppm_max = 5;

% Find the indices corresponding to this ppm range
peak_indices = find(ppm_vec >= ppm_min & ppm_vec <= ppm_max);

% Extract the ppm values and corresponding amplitudes for integration
peak_ppm   = ppm_vec(peak_indices);
peak_spec   = spec_data(peak_indices);
delta_ppm=    ppm_vec(2)-ppm_vec(1);
% Calculate area using the trapezoidal rule


fid_data(1)% 1. Compute the sum over the ENTIRE calculated spectrum (No PPM limits)
% Since it is an ifft, the sum of the spectrum multiplied by N recovers x(0)
recovered_t0 = sum(calculated_fft) ;

% 2. Compare directly with your first FID point
fprintf('--- Validation Check ---\n');
fprintf('First point of FID (t=0):          %.4f + %.4fi\n', real(fid_data(1)), imag(fid_data(1)));
fprintf('Sum of entire calculated spectrum: %.4f + %.4fi\n', real(recovered_t0), imag(recovered_t0));
% =========================================================================
% WINDOW 1: Time Domain (Real Part Only)
% =========================================================================
figure('Name', 'Time Domain - Real Part', 'NumberTitle', 'off');
plot(time_vec, real(fid_data), 'b-', 'LineWidth', 1.5);
title('Choline Basis Set: Time Domain (Real Part)');
xlabel('Time (ms)');
ylabel('Amplitude');
grid on;
set(gca, 'FontSize', 11);

% =========================================================================
% WINDOW 2: Time Domain (Imaginary Part Only)
% =========================================================================
figure('Name', 'Time Domain - Imaginary Part', 'NumberTitle', 'off');
plot(time_vec, imag(fid_data), 'r-', 'LineWidth', 1.5);
title('Choline Basis Set: Time Domain (Imaginary Part)');
xlabel('Time (ms)');
ylabel('Amplitude');
grid on;
set(gca, 'FontSize', 11);

% =========================================================================
% WINDOW 3: Frequency Domain (Magnitude vs Calculated FFT)
% =========================================================================
figure('Name', 'Frequency Domain - Comparison', 'NumberTitle', 'off');

% Plot pre-calculated spectrum

plot(ppm_vec, abs(magnitude_spectrum), 'b-', 'LineWidth', 2, 'DisplayName', 'outA.specs (Pre-calculated)');
title('myoinositol Basis Set: spectrum before scaling');
hold on;
figure('Name', 'Frequency Domain - Comparison', 'NumberTitle', 'off');

plot(ppm_vec, imag(magnitude_spectrum), 'b-', 'LineWidth', 2, 'DisplayName', 'outA.specs (Pre-calculated)');
hold on;
% Plot your manually calculated, decayed FFT spectrum
%plot(ppm_vec, magnitude_calc_fft, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Calculated ifftshift(ifft)');
hold off;

title('Choline Basis Set: Frequency Domain Comparison');
xlabel('Chemical Shift (ppm)');
ylabel('Magnitude (a.u.)');
xlim([0, 5]); % Lock view strictly to the 0-5 ppm range specified
grid on;
legend('show', 'Location', 'best');

% Standard MRS visualization: High ppm on the left, low ppm on the right
set(gca, 'XDir', 'reverse'); 
set(gca, 'FontSize', 11);