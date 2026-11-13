% =========================================================================
% MATLAB Script: Plot Rescaled MRS Spectra from Dual-Pass Analysis
% =========================================================================
clear; clc; close all;

% 1. Load the new multi-table results file
resultsFile = 'MRS_DualPass_Results.mat';
if exist(resultsFile, 'file')
    % Loads baseline structures, tables, and variables into workspace
    load(resultsFile); 
else
    error(['Could not find %s. Please ensure your new dual-pass ' ...
           'analysis script has finished executing.'], resultsFile);
end

% Safety check for the specific new table variable name
if ~exist('Table2_Rescaled', 'var') && exist('PerPeakTable', 'var')
    Table2_Rescaled = PerPeakTable; % Fallback if named differently in .mat
elseif ~exist('Table2_Rescaled', 'var')
    error('Target rescaled data table not found in the loaded .mat file.');
end

% 2. Setup plotting parameters
plotMetabs = {'myoinositol', 'lactate'};
figure('Color', [1 1 1], 'Position', [100, 100, 1000, 450]);

plotCount = 0;
for m = 1:length(plotMetabs)
    targetMetab = plotMetabs{m};
    
    % Find row index in Table 2 (Rescaled Analysis)
    tblIdx = find(strcmp(Table2_Rescaled.Metabolite, targetMetab), 1, 'first');
    if isempty(tblIdx)
        warning('Metabolite "%s" not found in Table2_Rescaled. Skipping.', targetMetab);
        continue;
    end
    
    % Dynamically read the applied factor column from the data table
    if ismember('Applied_Factor', Table2_Rescaled.Properties.VariableNames)
        fFactor = Table2_Rescaled.Applied_Factor(tblIdx);
    elseif ismember('Scaling_Factor', Table2_Rescaled.Properties.VariableNames)
        fFactor = Table2_Rescaled.Scaling_Factor(tblIdx);
    else
        fFactor = 1.0; 
        warning('Could not find factor column. Defaulting to 1.0');
    end
    
    fileName = [targetMetab, '_1.mat'];
    
    try
        % Load raw basis component vectors
        rawBasis = load(fileName);
        outA      = rawBasis.outA;
        spec_data = squeeze(outA.specs(:));
        ppm_vec   = squeeze(outA.ppm(:));
        
        [ppm_sorted, sortIdx] = sort(ppm_vec, 'ascend');
        spec_sorted = spec_data(sortIdx);
        
        % Process real phased components
        spec_orig     = real(spec_sorted);
        spec_rescaled = spec_orig .* fFactor;
        
        % Render Plot Window
        plotCount = plotCount + 1;
        subplot(1, 2, plotCount);
        
        plot(ppm_sorted, spec_orig, 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5, 'LineStyle', '--'); hold on;
        plot(ppm_sorted, spec_rescaled, 'Color', [0.12 0.53 0.90], 'LineWidth', 2); % Blue-highlight for FID pass
        
        set(gca, 'XDir', 'reverse');
        
        % Set spectroscopic viewports around dominant multiplets
        if strcmp(targetMetab, 'lactate')
            xlim([0.8, 4.5]);
        elseif strcmp(targetMetab, 'myoinositol')
            xlim([3.0, 4.3]);
        else
            xlim([min(ppm_sorted), max(ppm_sorted)]);
        end
        
        grid on;
        xlabel('Chemical Shift (ppm)', 'FontSize', 10);
        ylabel('Intensity (Real Absorption)', 'FontSize', 10);
        title(sprintf('%s Spectrum\n(FID Factor Applied = %.5f)', upper(targetMetab), fFactor), 'FontSize', 11);
        legend('Original Baseline', 'Rescaled Pass 2', 'Location', 'best');
        
    catch ME
        warning('Could not process/plot %s. Error: %s', fileName, ME.message);
    end
end

sgtitle('MRS Basis Spectra Dual-Pass Scaling Verification', 'FontSize', 13, 'FontWeight', 'bold');