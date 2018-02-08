% This code reads inputs data from four bands of MODIS instrument and
% trains a feed forward Neural Network to reconstruct normalized Solar
% Induced Fluorescence (SIF) from GOME-2 instrument. Training data is
% normalized SIF at biweekly timescale and 0.5 deg spatial resolution. 
%
% Details of the methodology is described in the following paper which is
% under review in GRL:
%
%
% Gentine P., Alemohammad S.H., RSIF (Reconstructed Solar Induced 
% Fluorescence): a machine-learning vegetation product based on MODIS
% surface reflectance to reproduce GOME-2 solar induced fluorescence,
% Geophysical Research Letters, in revision. 
%
%
%
% Version: 1.0, Feb. 2018.
% Author:  S. Hamed Alemohammad, h.alemohammad@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



clc
clear 
close all
pathInputData = '';


%% Parallel Toolbox Initiation
myCluster = parcluster('local');
myCluster.NumWorkers = 24;
saveProfile(myCluster);
parpool('local', 24)

%% Paramaters
N1 = 119; % Number of months to be used for training NN
N2 = 239; % Total number of months of data

%% Loading Ancillary Data

load('RSIFDesertMask.mat')

%% Parameters
for NLayers = 1 : 5
    for NNeurons = [2:5, 7, 10]


        %% Loading Inputs
        B1 = load([pathInputData, 'MODIS_Reflectance_Band_1.mat']); 
        B2 = load([pathInputData, 'MODIS_Reflectance_Band_2.mat']);  
        B3 = load([pathInputData, 'MODIS_Reflectance_Band_3.mat']);  
        B4 = load([pathInputData, 'MODIS_Reflectance_Band_4.mat']);  
        
        %% Removing Desert Pixels
        B1 = remove_desert(B1, RSIFDesertMask);
        B2 = remove_desert(B2, RSIFDesertMask);
        B3 = remove_desert(B3, RSIFDesertMask);
        B4 = remove_desert(B4, RSIFDesertMask);
        
        %% Training Data
        B1Training = B1.MODISReflectance(:, :, 1 : N1);
        B2Training = B2.MODISReflectance(:, :, 1 : N1);
        B3Training = B3.MODISReflectance(:, :, 1 : N1);
        B4Training = B4.MODISReflectance(:, :, 1 : N1);
        
        %% Validation Data
        B1Validation = B1.MODISReflectance(:, :, N1 + 1 : N2);
        B2Validation = B2.MODISReflectance(:, :, N1 + 1 : N2);
        B3Validation = B3.MODISReflectance(:, :, N1 + 1 : N2);
        B4Validation = B4.MODISReflectance(:, :, N1 + 1 : N2);
        
        %% Constructing Input Variables 
        inputsTraining = [B1Training(:)'; B2Training(:)'; B3Training(:)'; B4Training(:)'];
        inputsValidation = [B1Validation(:)'; B2Validation(:)'; B3Validation(:)'; B4Validation(:)'];
        inputsEstimation = [B1.MODISReflectance(:)'; B2.MODISReflectance(:)'; B3.MODISReflectance(:)'; B4.MODISReflectance(:)'];
        clear B1 B2 B3 B4 
        clear B1Training B2Training B3Training B4Training 
        clear B1Validation B2Validation B3Validation B4Validation 

        %% loading Target Data
        load([pathInputData, 'SIF.mat'], 'PAR_normalized_SIF')
        
        PAR_SIF = PAR_normalized_SIF(:, :, N1 + 1 : N2); 
        targetsValidation = PAR_SIF(:)';
        
        PAR_SIF = PAR_normalized_SIF(:, :, 1 : 1 : N1); 
        targetsTraining = PAR_SIF(:)';

        clear PAR_SIF PAR_normalized_SIF
        %% Removing NaNs
        [inputsTraining, targetsTraining] = removeNaN(inputsTraining, targetsTraining);
        [inputsValidation, targetsValidation] = removeNaN(inputsValidation, targetsValidation);
        
        %% Training NN
        disp(['Training Started: N=', int2str(NNeurons), ', L=', int2str(NLayers)])
        [netRSIF, trRSIF] = trainNet(inputsTraining, targetsTraining, NLayers, NNeurons);
        disp('Training Ended')

        % Generating PAR_SIF from training data 
        trainingRSIF_PAR = netRSIF(inputsTraining, 'useParallel', 'no');

        % Generating PAR_SIF from validation data 
        validationRSIF_PAR = netRSIF(inputsValidation, 'useParallel', 'yes');

        
        [validationRSIF_PAR, targetsValidationRSIF_PAR] = removeNaN(validationRSIF_PAR, targetsValidation);

        rRSIF_PAR = corr(validationRSIF_PAR', targetsValidationRSIF_PAR');
        
        performanceRSIF_PAR = mean((targetsValidationRSIF_PAR - validationRSIF_PAR) .^ 2);
        
        %% Saving Network Structure
        save(['RSIFNet_', sprintf('%02i', NNeurons), 'N_', sprintf('%02i', NLayers), 'L.mat'], 'netRSIF', 'trRSIF', 'performanceRSIF_PAR', 'rRSIF_PAR', 'targetsValidation', 'validationRSIF_PAR', 'targetsTraining', 'trainingRSIF_PAR');

   
        %% Estimating PAR_SIF for the whole study period
        OutputTemp = netRSIF(inputsEstimation, 'useParallel', 'yes');
        RSIF_PAR = reshape(OutputTemp, 360, 720, N2);
        clear OutputTemp

        %% Saving PAR_SIF 
        save(['RSIF_2007_2016_', sprintf('%02i', NNeurons), 'N_', sprintf('%02i', NLayers), 'L.mat'], 'RSIF_PAR')

    end
end

