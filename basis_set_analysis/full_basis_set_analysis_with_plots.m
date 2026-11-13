% =========================================================================
% MATLAB Script: Complete Unified MRS Basis Set Analysis Table with Plots
% =========================================================================
% This script integrates the full-spectrum analysis and per-peak analysis
% into a single consolidated table and generates clean boundary plots per metabolite.
%
% NOTE: NAA and Creatine now include downfield exchangeable-proton peaks
% (NAA amide -NH- ~7.6-7.9 ppm; Creatine guanidino -NH2/-NH ~6.4-6.9 ppm).
% These are genuine resonances in full-spin-system basis sets but are
% usually suppressed/broadened out of in-vivo spectra by water suppression
% and fast exchange dynamics - see report Sec. "Downfield Exchangeable
% Proton Resonances" for details. The plotting window auto-expands to
% include these peaks since minPlotPPM/maxPlotPPM are derived from the
% peakDefs windows themselves.
% =========================================================================
clear; clc; close all;

% -------------------------------------------------------------------
% 1. Define peak/multiplet windows per metabolite (chemical shift, ppm)
%    Each row: {PeakLabel, ppm_low, ppm_high, Protons, Multiplicity}
% -------------------------------------------------------------------
peakDefs = containers.Map();
peakDefs('naa') = {
    'Acetyl CH3 (2.01 ppm)',        1.7, 2.2, 3, 'singlet';
    'Amide NH (7.6-7.9 ppm)',       7.6, 7.9, 1, 'broad singlet (exchangeable)'
};
peakDefs('choline') = {
    'N-(CH3)3 (3.19-3.22 ppm)',     2.8, 3.5, 9, 'singlet'
};
peakDefs('creatine') = {
    'N-CH3 (3.03 ppm)',             2.6, 3.40, 3, 'singlet';
    'N-CH2 (3.91-3.94 ppm)',        3.6, 4.4, 2, 'singlet';
    'Guanidino NH2/NH (6.4-6.9 ppm)', 6.4, 6.9, 2, 'broad multiplet (exchangeable)'
};
peakDefs('lactate') = {
    'CH3 doublet (1.31 ppm)',       1, 1.5, 3, 'doublet';
    'CH quartet (4.10 ppm)',        3.8, 4.3, 1, 'quartet'
};
peakDefs('myoinositol') = {
    'Ring CH main cluster (3.52-3.56 ppm)', 3.4, 3.6, 4, 'multiplet';
    'Ring CH secondary (3.61 ppm)',         3.6, 3.8, 1, 'multiplet';
    'H-2 triplet (4.05 ppm)',               3.8, 4.2, 1, 'triplet'
};
peakDefs('water') = {
    'H2O singlet (~4.65-4.70 ppm)', 4.2, 5, 2, 'singlet'
};

% -------------------------------------------------------------------
% 2. Find all metabolite .mat files
% -------------------------------------------------------------------
matFiles = dir('*_1.mat');
numFiles = length(matFiles);
if numFiles == 0
    error('No metabolite .mat files found in the current folder.');
end

% -------------------------------------------------------------------
% 3. Preallocate and loop over files to extract all data
% -------------------------------------------------------------------
Metabolite          = {};
PeakLabel           = {};
Multiplicity        = {};
Protons             = [];
PPM_Low             = [];
PPM_High             = [];
Area_Complex_T0     = [];   
Area_Peak           = [];   
Peak_Height         = [];   
Area_Peaks_Full     = [];   
T0_Value_FID        = [];   
Area_Magnitude_Spec = [];   

for i = 1:numFiles
    fileName = matFiles(i).name;
    [~, nameWithoutExt, ~] = fileparts(fileName);
    cleanName = lower(strrep(nameWithoutExt, '_1', ''));
    
    if ~isKey(peakDefs, cleanName)
        warning('No peak definitions for "%s" - skipping.', cleanName);
        continue;
    end
    
    loadedData = load(fileName);
    if ~isfield(loadedData, 'outA')
        warning('File "%s" has no outA field - skipping.', fileName);
        continue;
    end
    
    outA       = loadedData.outA;
    fid_data   = squeeze(outA.fids);
    spec_data  = squeeze(outA.specs);
    ppm_vec    = squeeze(outA.ppm);
    
    fid_data   = fid_data(:);
    spec_data  = spec_data(:);
    ppm_vec    = ppm_vec(:);
    
    % --- Full Spectrum Script Calculations ---
    calculated_fft = ifftshift(ifft(fid_data));
    full_fft_sum   = sum(calculated_fft);
    t0_val         = abs(fid_data(1));
    
    full_mag_area = NaN;
    full_spec_idx = find(ppm_vec >= 0 & ppm_vec <= 5);
    if ~isempty(full_spec_idx)
        full_mag_area = sum(abs(spec_data(full_spec_idx)));
    end
    
    % --- Per-Peak Window Sorting & Extraction ---
    [ppm_sorted, sortIdx] = sort(ppm_vec, 'ascend');
    spec_sorted = spec_data(sortIdx);
    mag_sorted  = abs(spec_sorted);
    
    thisMetabPeaks = peakDefs(cleanName);
    nPeaks = size(thisMetabPeaks, 1);
    
    % --- Initialize Figure for Current Metabolite ---
    fig = figure('Name', sprintf('Spectrum Analysis - %s', upper(cleanName)),'Color', 'w', 'Position', [200, 200, 800, 500]);
    hSpec = plot(ppm_sorted, mag_sorted, 'k-', 'LineWidth', 1.6);
    hold on;
    
    % Track bounding limits to dynamically scale plot window
    minPlotPPM = 8.0;
    maxPlotPPM = 0.0;
    maxAmpInWindows = 0;
    
    % Store handles for the legend to avoid duplicates
    hLowLine = [];
    hHighLine = [];
    hShade = [];
    
    for p = 1:nPeaks
        label   = thisMetabPeaks{p, 1};
        ppmLow  = thisMetabPeaks{p, 2};
        ppmHigh = thisMetabPeaks{p, 3};
        nProt   = thisMetabPeaks{p, 4};
        multip  = thisMetabPeaks{p, 5};
        
        idx = find(ppm_sorted >= ppmLow & ppm_sorted <= ppmHigh);
        
        if numel(idx) < 2
            area      = NaN;
            comp_area = NaN;
            height    = NaN;
            warning('Peak "%s" for %s has <2 points in [%.2f %.2f] ppm.', ...
                     label, cleanName, ppmLow, ppmHigh);
        else
            area      = sum(mag_sorted(idx));
            comp_area = sum(spec_sorted(idx)); 
            height    = max(mag_sorted(idx));
            
            minPlotPPM = min(minPlotPPM, ppmLow);
            maxPlotPPM = max(maxPlotPPM, ppmHigh);
            maxAmpInWindows = max(maxAmpInWindows, height);
        end
        
        % Calculate a safe vertical boundary ceiling
        yMaxLimit = max(mag_sorted) * 1.15;
        
        % --- Light Shading for the Window Region ---
        hShade = fill([ppmLow, ppmHigh, ppmHigh, ppmLow], [0, 0, yMaxLimit, yMaxLimit], ...
                      [0.92 0.92 0.95], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
        
        % --- Draw Vertical Integration Bounds ---
        hLowLine  = line([ppmLow, ppmLow],   [0, yMaxLimit], 'Color', [0.85 0.33 0.1], 'LineStyle', '--', 'LineWidth', 1.4);
        hHighLine = line([ppmHigh, ppmHigh], [0, yMaxLimit], 'Color', [0 0.45 0.74], 'LineStyle', '--', 'LineWidth', 1.4);
        
        % Text tag placement slightly above the peak max
        if ~isnan(height)
            text(mean([ppmLow, ppmHigh]), height + (max(mag_sorted)*0.03), ...
                 sprintf('%s\n(%.2f-%.2f ppm)', label, ppmLow, ppmHigh), ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
                 'FontSize', 9, 'FontWeight', 'bold', 'Color', [0.2 0.2 0.2]);
        end

        % Append values row by row
        Metabolite{end+1,1}      = cleanName;         
        PeakLabel{end+1,1}       = label;
        Multiplicity{end+1,1}    = multip;
        Protons(end+1,1)         = nProt;
        PPM_Low(end+1,1)         = ppmLow;
        PPM_High(end+1,1)        = ppmHigh;
        Area_Complex_T0(end+1,1) = comp_area; 
        Area_Peak(end+1,1)       = area;
        Peak_Height(end+1,1)     = height;
        
        Area_Peaks_Full(end+1,1)     = full_fft_sum;
        T0_Value_FID(end+1,1)        = t0_val;
        Area_Magnitude_Spec(end+1,1) = full_mag_area;
    end
    
    % --- Finalize and Beautify Plot Axes ---
    set(gca, 'XDir', 'reverse', 'Box', 'off', 'TickDir', 'out', 'LineWidth', 1.1); 
    xlabel('Chemical Shift (ppm)', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Magnitude (a.u.)', 'FontSize', 11, 'FontWeight', 'bold');
    title(sprintf('Metabolite Window Identification: %s', upper(cleanName)), 'FontSize', 13, 'FontWeight', 'bold');
    grid on;
    
    % Clean up limits with custom padding buffers
    % Calculate absolute minimum and maximum with padding
    xMin = minPlotPPM - 0.5;
    xMax = maxPlotPPM + 0.5;

    % Ensure values are strictly ascending for MATLAB's validation
    xlim([xMin, xMax]); 

    % This handles reversing the visual display (MRS convention) safely
    set(gca, 'XDir', 'reverse', 'Box', 'off', 'TickDir', 'out', 'LineWidth', 1.1);
    ylim([0, maxAmpInWindows * 1.3]);
    
    % Build exact legend mapping without object duplication errors
    legendPlots = [hSpec, hLowLine, hHighLine, hShade];
    legendLabels = {'Magnitude Spectrum', 'PPM Low Bound', 'PPM High Bound', 'Integration Window'};
    % Filter out empty references if error conditions occurred
    validIdx = ~cellfun(@isempty, {hSpec, hLowLine, hHighLine, hShade});
    legend(legendPlots(validIdx), legendLabels(validIdx), 'Location', 'NorthEast', 'AutoUpdate', 'off');
end

% -------------------------------------------------------------------
% 4. Build and Process the Master Results Table
% -------------------------------------------------------------------
MasterTable = table(Metabolite, PeakLabel, Multiplicity, Protons, ...
                    PPM_Low, PPM_High, Area_Complex_T0, ...
                    Area_Peak, Peak_Height, Area_Peaks_Full, T0_Value_FID, Area_Magnitude_Spec, ...
                    'VariableNames', {'Metabolite','Peak','Multiplicity','Protons',...
                    'PPM_Low','PPM_High','Area_Complex_T0',...
                    'Area_Peak','Peak_Height','Area_Peaks_Full','t_0_Value','Area_Magnitude_Spec'});

summaryTmp = groupsummary(MasterTable, 'Metabolite', 'sum', 'Area_Complex_T0');
[~, loc] = ismember(MasterTable.Metabolite, summaryTmp.Metabolite); 
Sum_Peaks_For_Metabolite = summaryTmp.sum_Area_Complex_T0(loc);

MasterTable = addvars(MasterTable, Sum_Peaks_For_Metabolite, 'Before', 'Area_Peak', ...
    'NewVariableNames', {'Sum_Peaks_For_Metabolite'});

Area_Per_Proton = MasterTable.Area_Peak ./ MasterTable.Protons;
MasterTable.Area_Per_Proton = Area_Per_Proton;

% -------------------------------------------------------------------
% 5. Display and Save Consolidated Outputs
% -------------------------------------------------------------------
disp('================================================================================================');
disp('                               CONSOLIDATED MRS BASIS SET ANALYSIS                              ');
disp('================================================================================================');
disp(MasterTable);
writetable(MasterTable, 'Consolidated_MRS_Results.csv');
save('Consolidated_MRS_Results.mat', 'MasterTable');
disp('Saved perfectly merged data structure to: Consolidated_MRS_Results.csv / .mat');