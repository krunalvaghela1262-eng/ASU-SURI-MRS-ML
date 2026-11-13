% =========================================================================
% MATLAB Script: Per-Peak / Per-Multiplet Area Analysis for MRS Basis Sets
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
t2Defs('choline')     = 242;
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
% 3. Loop over files and collect data vectors
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
        
        Metabolite{end+1,1}   = cleanName;         
        PeakLabel{end+1,1}    = label;
        Protons(end+1,1)      = nProt;
        Area_Complex(end+1,1) = comp_area;
        Peak_Height(end+1,1)  = height;
    end
end

% -------------------------------------------------------------------
% 4. Vector Calculations & Conversions
% -------------------------------------------------------------------
totalRows = length(Metabolite);
T2_ms = zeros(totalRows, 1);

% Convert the segmented complex areas into absolute magnitude values
Area_Magnitude_peaks_segmented = abs(Area_Complex);

Area_per_peak_Per_Proton = Area_Magnitude_peaks_segmented ./ Protons;

TE = 100; % Echo time in ms
for row = 1:totalRows
    metab = Metabolite{row};
    if isKey(t2Defs, metab)
        T2_ms(row) = t2Defs(metab);
    else
        T2_ms(row) = NaN;
    end
end

Exp_neg_TE_T2 = exp(-TE ./ T2_ms);
Ratio_Area_Proton_to_Exp = Area_per_peak_Per_Proton ./ Exp_neg_TE_T2;

% -------------------------------------------------------------------
% 5. Build Final Streamlined Table via Struct Map
% -------------------------------------------------------------------
outStruct = struct(...
    'Metabolite', {Metabolite}, ...
    'Peak', {PeakLabel}, ...
    'Protons', Protons, ...
    'Area_Magnitude_peaks_segmented', Area_Magnitude_peaks_segmented, ...
    'Peak_Height', Peak_Height, ...
    'Area_per_peak_Per_Proton', Area_per_peak_Per_Proton, ...
    'T2_ms', T2_ms, ...
    'Ratio_Area_Proton_to_Exp', Ratio_Area_Proton_to_Exp ...
);

PerPeakTable = struct2table(outStruct);

disp('==================== PER-PEAK STREAMLINED MRS ANALYSIS ====================');
disp(PerPeakTable);

% -------------------------------------------------------------------
% 6. Save results
% -------------------------------------------------------------------
writetable(PerPeakTable, 'PerPeak_Area_Results.csv');
save('PerPeak_Area_Results.mat', 'PerPeakTable');
disp('Saved successfully: PerPeak_Area_Results.csv / .mat');