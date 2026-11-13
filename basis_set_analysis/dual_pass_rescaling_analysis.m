% =========================================================================
% MATLAB Script: Two-Pass MRS Analysis (Baseline vs. Rescaled Time-Domain FID)
% =========================================================================
clear; clc;

% -------------------------------------------------------------------
% 1. Configurations, Target Windows, and Metadata Definitions
% -------------------------------------------------------------------
peakDefs = containers.Map();
peakDefs('naa') = {
    'Acetyl CH3 (2.01 ppm)',        1.95, 2.07, 3, 'singlet'
};
peakDefs('pch') = {
    'N-(CH3)3 (3.19-3.22 ppm)',     3.14, 3.26, 9, 'singlet'
};
peakDefs('cr') = {
    'N-CH3 (3.03 ppm)',             2.96, 3.20, 3, 'singlet';
    'N-CH2 (3.91-3.94 ppm)',        3.85, 3.98, 2, 'singlet'
};
peakDefs('lac') = {
    'CH3 doublet (1.31 ppm)',       1.24, 1.38, 3, 'doublet'
};
peakDefs('mi') = {
    'Ring CH main cluster (3.52-3.56 ppm)', 3.49, 3.58, 4, 'multiplet'
};
peakDefs('h2o') = {
    'H2O singlet (~4.65-4.70 ppm)', 4.60, 4.80, 2, 'singlet'
};

% T2 relaxation values (ms)
t2Defs = containers.Map();
t2Defs('lac')     = 240;
t2Defs('pch')     = 265;
t2Defs('cr')      = 147;
t2Defs('naa')     = 258;
t2Defs('mi')      = 197;
t2Defs('h2o')     = 83.7;

% Define the precise scaling factors derived from your ratio target
scalingMap = containers.Map();
scalingMap('pch')     = 1.00000;
scalingMap('cr')      = 1.00000;
scalingMap('lac')     = 4.10799;
scalingMap('mi')      = 5.40013;
scalingMap('naa')     = 1.00000;
scalingMap('h2o')     = 1.00000;

TE = 100; % Echo time in ms

% Updated file search pattern based on your screenshot
matFiles = dir('GE_UnEdited_PRESS_TE30_*.mat');

if isempty(matFiles)
    error('No matching GE_UnEdited_PRESS_TE30_*.mat files found in the current folder.');
end

% -------------------------------------------------------------------
% 2. PASS 1: Baseline Analysis (No Scaling Applied)
% -------------------------------------------------------------------
[Table_Baseline, ~] = run_mrs_pipeline(matFiles, peakDefs, t2Defs, scalingMap, TE, false);

% -------------------------------------------------------------------
% 3. PASS 2: Rescaled Analysis (FIDs Multiplied by Scaling Factors)
% -------------------------------------------------------------------
[Table_Rescaled, calculatedFactors] = run_mrs_pipeline(matFiles, peakDefs, t2Defs, scalingMap, TE, true);

% -------------------------------------------------------------------
% 4. Display Summary Output and Tables
% -------------------------------------------------------------------
clc;
disp('====================================================================');
disp('                APPLIED TIME-DOMAIN MULTIPLICATION FACTORS          ');
disp('====================================================================');
for row = 1:length(calculatedFactors)
    fprintf('Metabolite: %-12s | FID Scaling Factor Applied = %.5f\n', ...
            calculatedFactors{row, 1}, calculatedFactors{row, 2});
end

fprintf('\n');
disp('==================== TABLE 1: BASELINE MRS ANALYSIS ====================');
disp(Table_Baseline);

disp('==================== TABLE 2: RESCALED MRS ANALYSIS ====================');
disp(Table_Rescaled);

% Save Both Tables Separately
writetable(Table_Baseline, 'MRS_Analysis_Baseline.csv');
writetable(Table_Rescaled, 'MRS_Analysis_Rescaled_FID.csv');
save('MRS_DualPass_Results.mat', 'Table_Baseline', 'Table_Rescaled');

disp('Saved successfully: MRS_Analysis_Baseline.csv, MRS_Analysis_Rescaled_FID.csv, and MRS_DualPass_Results.mat');

% =========================================================================
% Helper Processing Pipeline Function
% =========================================================================
function [FinalTable, FactorLog] = run_mrs_pipeline(matFiles, peakDefs, t2Defs, scalingMap, TE, applyScaling)
    Metabolite   = {};
    PeakLabel    = {};
    Protons      = [];
    Area_Complex = [];   
    Peak_Height  = [];   
    FactorLog    = {};

    for i = 1:length(matFiles)
        fileName = matFiles(i).name;
        [~, nameWithoutExt, ~] = fileparts(fileName);
        
        % Extract the metabolite tag after 'GE_UnEdited_PRESS_TE30_'
        cleanName = lower(strrep(nameWithoutExt, 'GE_UnEdited_PRESS_TE30_', ''));
        
        % Skip separate CrCH2 file if Cr already covers the multi-peak analysis
        if strcmp(cleanName, 'crch2')
            continue; 
        end

        if ~isKey(peakDefs, cleanName), continue; end

        loadedData = load(fileName);
        if ~isfield(loadedData, 'outA'), continue; end
        outA = loadedData.outA;
        
        % Fetch current scaling factor
        factor = 1.0;
        if applyScaling && isKey(scalingMap, cleanName)
            factor = scalingMap(cleanName);
        end
        
        % Apply transformation to the raw fields if requested
        if applyScaling
            outA.fids  = outA.fids  .* factor;
            outA.specs = outA.specs .* factor;
        end

        spec_data = squeeze(outA.specs(:));
        ppm_vec   = squeeze(outA.ppm(:));

        [ppm_sorted, sortIdx] = sort(ppm_vec, 'ascend');
        spec_sorted = spec_data(sortIdx);
        mag_sorted  = abs(spec_sorted);

        thisMetabPeaks = peakDefs(cleanName);
        nPeaks = size(thisMetabPeaks, 1);

        tempLabels   = cell(nPeaks, 1);
        tempProtons  = zeros(nPeaks, 1);
        tempComplex  = zeros(nPeaks, 1);
        tempHeights  = zeros(nPeaks, 1);

        for p = 1:nPeaks
            label   = thisMetabPeaks{p, 1};
            ppmLow  = thisMetabPeaks{p, 2};
            ppmHigh = thisMetabPeaks{p, 3};
            nProt   = thisMetabPeaks{p, 4};

            idx = find(ppm_sorted >= ppmLow & ppm_sorted <= ppmHigh);

            if numel(idx) < 2
                comp_area = NaN; height = NaN;
            else
                comp_area = sum(spec_sorted(idx));      
                height    = max(mag_sorted(idx));
            end

            tempLabels{p}   = label;
            tempProtons(p)  = nProt;
            tempComplex(p)  = comp_area;
            tempHeights(p)  = height;
        end

        if strcmp(cleanName, 'cr')
            keepIdx = [1, 2];
        else
            [~, maxIdx] = max(tempHeights);
            keepIdx = maxIdx;
        end

        for k = 1:length(keepIdx)
            pIdx = keepIdx(k);
            Metabolite{end+1,1}   = cleanName;         
            PeakLabel{end+1,1}    = tempLabels{pIdx};
            Protons(end+1,1)      = tempProtons(pIdx);
            Area_Complex(end+1,1) = tempComplex(pIdx);
            Peak_Height(end+1,1)  = tempHeights(pIdx);
            FactorLog{end+1,1}    = cleanName;
            FactorLog{end,2}      = factor;
        end
    end

    % Calculations
    totalRows = length(Metabolite);
    T2_ms = zeros(totalRows, 1);
    Applied_Factor = zeros(totalRows, 1);

    Area_Magnitude_segmented = abs(Area_Complex);
    Area_per_peak_Per_Proton = Area_Magnitude_segmented ./ Protons;

    for row = 1:totalRows
        metab = Metabolite{row};
        T2_ms(row) = t2Defs(metab);
        Applied_Factor(row) = FactorLog{row, 2};
    end

    Exp_neg_TE_T2 = exp(-TE ./ T2_ms);
    Ratio_Area_Proton_to_Exp = Area_per_peak_Per_Proton ./ Exp_neg_TE_T2;

    outStruct = struct(...
        'Metabolite', {Metabolite}, ...
        'Peak', {PeakLabel}, ...
        'Protons', Protons, ...
        'Area_Magnitude_segmented', Area_Magnitude_segmented, ...
        'Peak_Height', Peak_Height, ...
        'Area_per_Proton', Area_per_peak_Per_Proton, ...
        'Applied_Factor', Applied_Factor, ...
        'T2_ms', T2_ms, ...
        'Ratio_Area_Proton_to_Exp', Ratio_Area_Proton_to_Exp ...
    );

    FinalTable = struct2table(outStruct);
end