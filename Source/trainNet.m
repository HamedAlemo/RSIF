function [net, tr] = trainNet(inputs, targets, NLayers, NNeurons)

%% Initiating NN
hiddenLayerSize = NNeurons * ones(1, NLayers);
net = fitnet(hiddenLayerSize);

%% Setting Training Parameters
net.performFcn = 'mse'; %'mse'; 
net.layers{1:NLayers}.transferFcn = 'poslin';
net.trainParam.max_fail = 200; 
net.trainParam.min_grad = 1e-10;
net.trainParam.mu_max = 1e20;
net.trainParam.mu_inc = 2;
net.trainParam.mu = 0.00001;  
net.trainParam.epochs = 5000; 


%% Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 60/100;
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 20/100;
 
%% Training the Network
[net, tr] = train(net, inputs, targets, 'useParallel', 'yes');
