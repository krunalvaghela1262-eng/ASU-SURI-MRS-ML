% =========================================================================
% MATLAB Script: Per-Peak Area Analysis with Target Ratio-Based Rescaling
% =========================================================================
clear; clc;

% -------------------------------------------------------------------
% 1. Define peak/multiplet windows per metabolite (chemical shift, ppm)
% -------------------------------------------------------------------
peakDefs = containers.Map();
peakDefs('naa') = {
    'Acetyl CH3 (2.01 ppm)',        1.95, 2.07, 3, 'singlet';
    'Amide NH (7.6-7.9 ppm)',       7.6,  7.9,  1, 'broad singlet (exchangeable)'
};
peakDefs('choline') = {
    'N-(CH3)3 (3.19-3.22 ppm)',     3.14, 3.26, 9, 'singlet'
};
peakDefs('creatine') = {
    'N-CH3 (3.03 ppm)',             2.96, 3.20, 3, 'singlet';
    'N-CH2 (3.91-3.94 ppm)',        3.85, 3.98, 2, 'singlet';
    'Guanidino NH2/NH (6.4-6.9 ppm)', 6.4, 6.9, 2, 'broad multiplet (exchangeable)'
};
peakDefs('lactate') = {
    'CH3 doublet (1.31 ppm)',       1.24, 1.38, 3, 'doublet';
    'CH quartet (4.10 ppm)',        4.03, 4.17, 1, 'quartet'
};
peakDefs('myoinositol') = {
    'Ring CH main cluster (3.52-3.56 ppm)', 3.49, 3.58, 4, 'multiplet';
    'Ring CH secondary (3.61 ppm)',         3.58, 3.66, 1, 'multiplet';
    'H-2 triplet (4.05 ppm)',               3.99, 4.11, 1, 'triplet'
};
peakDefs('water') = {
    'H2O singlet (~4.65-4.70 ppm)', 4.60, 4.80, 2, 'singlet'
};

% T2 relaxation values (ms)
t2Defs = containers.Map();
t2Defs('lactate')     = 240;
t2Defs('choline')     = 265;
t2Defs('creatine')    = 147;
t2Defs('naa')         = 258;
t2Defs('myoinositol') = 197;
t2Defs('water')       = 83.7;

% -------------------------------------------------------------------
% 2. Find all metabolite .mat files
% -------------------------------------------------------------------
matFiles = dir('*_1.mat');
numFiles = length(matFiles);
if numFiles == 0
    error('No metabolite .mat files found in the current folder.');
end

% -------------------------------------------------------------------
% 3. Loop over files, evaluate all peaks, and filter for dominant ones
% -------------------------------------------------------------------
Metabolite   = {};
PeakLabel    = {};
Protons      = [];
Area_Complex = [];   
Peak_Height  = [];   

for i = 1:numFiles
    fileName = matFiles(i).name;
    [~, nameWithoutExt, ~] = fileparts(fileName);
    cleanName = lower(strrep(nameWithoutExt, '_1', ''));
    
    if ~isKey(peakDefs, cleanName)
        continue;
    end
    
    loadedData = load(fileName);
    if ~isfield(loadedData, 'outA')
        continue;
    end
    
    outA      = loadedData.outA;
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
            comp_area = NaN;
            height    = NaN;
        else
            comp_area = sum(spec_sorted(idx));      
            height    = max(mag_sorted(idx));
        end
        
        tempLabels{p}   = label;
        tempProtons(p)  = nProt;
        tempComplex(p)  = comp_area;
        tempHeights(p)  = height;
    end
    
    if strcmp(cleanName, 'creatine')
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
    end
end

% -------------------------------------------------------------------
% 4. Vector Calculations, Conversions & Targeted Scaling
% -------------------------------------------------------------------
totalRows = length(Metabolite);
T2_ms = zeros(totalRows, 1);
Scaling_Factor = ones(totalRows, 1); % Default scaling factor is 1.0
TARGET_RATIO = 0.65;
TE = 100; % Echo time in ms

Area_Magnitude_peaks_segmented = abs(Area_Complex);
Area_per_peak_Per_Proton = Area_Magnitude_peaks_segmented ./ Protons;

% Assign T2 values
for row = 1:totalRows
    metab = Metabolite{row};
    if isKey(t2Defs, metab)
        T2_ms(row) = t2Defs(metab);
    else
        T2_ms(row) = NaN;
    end
end

% Calculate base attenuation factor
Exp_neg_TE_T2 = exp(TE ./ T2_ms);

% Calculate specific scaling factors to force the final ratio to 0.65
for row = 1:totalRows
    metab = Metabolite{row};
    if strcmp(metab, 'lactate') || strcmp(metab, 'myoinositol')
        % Target_Ratio = (Area_per_Proton * Factor) / Exp_neg_TE_T2
        Scaling_Factor(row) = (TARGET_RATIO * Exp_neg_TE_T2(row)) / Area_per_peak_Per_Proton(row);
    end
end

% Apply scaling to get new values
Rescaled_Area_per_Proton = Area_per_peak_Per_Proton .* Scaling_Factor;
Ratio_Area_Proton_to_Exp = Rescaled_Area_per_Proton ./ Exp_neg_TE_T2;

% -------------------------------------------------------------------
% 5. Build Final Table
% -------------------------------------------------------------------
outStruct = struct(...
    'Metabolite', {Metabolite}, ...
    'Peak', {PeakLabel}, ...
    'Protons', Protons, ...
    'Area_Magnitude_segmented', Area_Magnitude_peaks_segmented, ...
    'Peak_Height', Peak_Height, ...
    'Area_per_Proton', Area_per_peak_Per_Proton, ...
    'Scaling_Factor', Scaling_Factor, ...
    'Rescaled_Area_per_Proton', Rescaled_Area_per_Proton, ...
    'T2_ms', T2_ms, ...
    'Ratio_Area_Proton_to_Exp', Ratio_Area_Proton_to_Exp ...
);

PerPeakTable = struct2table(outStruct);

% -------------------------------------------------------------------
% 6. Display Summary Output
% -------------------------------------------------------------------
disp('====================================================================');
disp('            CORRECTED BASIS SET SCALING FACTORS (FOR RATIO = 0.65)   ');
disp('====================================================================');
UniqueMetabs = unique(Metabolite);
for m = 1:length(UniqueMetabs)
    idx = find(strcmp(Metabolite, UniqueMetabs{m}), 1, 'first');
    fprintf('Metabolite: %-12s | Multiplicative Rescaling Factor = %.5f\n', ...
            UniqueMetabs{m}, Scaling_Factor(idx));
end
fprintf('\n');

disp('==================== DOMINANT PEAK MRS ANALYSIS ====================');
disp(PerPeakTable);

% Save updated structures
writetable(PerPeakTable, 'PerPeak_Area_Results.csv');
save('PerPeak_Area_Results.mat', 'PerPeakTable');
disp('Saved successfully: PerPeak_Area_Results.csv / .mat');