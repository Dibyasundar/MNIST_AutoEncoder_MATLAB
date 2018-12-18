clear;clc;close all;
%% Read tarining and training
if ~exist('MNIST_test_img.mat','file')
    images=loadMNISTImages('t10k-images.idx3-ubyte');
    lb=loadMNISTLabels('t10k-labels.idx1-ubyte');
    test_images=reshape(images,[28,28,1,size(images,2)]);
    test_labels=lb';
    save('MNIST_test_img.mat','test_images');
    save('MNIST_test_label.mat','test_labels');
end
if ~exist('MNIST_train_img.mat','file')
    images=loadMNISTImages('train-images.idx3-ubyte');
    lb=loadMNISTLabels('train-labels.idx1-ubyte');
    train_images=reshape(images,[28,28,1,size(images,2)]);
    train_labels=lb';
    save('MNIST_train_img.mat','train_images');
    save('MNIST_train_label.mat','train_labels');
end
clear;clc;
load('MNIST_train_img.mat');
load('MNIST_train_label.mat');
[m,n,p,q]=size(train_images);
u=unique(train_labels);
for i=1:size(u,2)
    cu{i}=strcat('c',num2str(u(i)));
end
train_target=categorical(train_labels,u,cu);
    s=0;
    if ~exist('network.mat','file')
        layers=[imageInputLayer([m,n,p])...
        convolution2dLayer(7,100)...
        reluLayer...
        maxPooling2dLayer(2)...
        convolution2dLayer(5,200)...
        maxPooling2dLayer(2)...
        fullyConnectedLayer(100)...
        reluLayer...
        fullyConnectedLayer(size(u,2))...
        softmaxLayer...
        classificationLayer]';
      s=1;  
    else
        load('network.mat');
        layers=net_pop.Layers;
    end

    if s==1
        ep=input('enter number of epoch :');
                options=trainingOptions('sgdm','MiniBatchSize',200,'InitialLearnRate',0.01,...
                    'MaxEpochs',ep,'ExecutionEnvironment','gpu',...
                    'Shuffle','once','VerboseFrequency',10,'Verbose',true,'L2Regularization',0.001,'Plots','training-progress');
                [net_pop,v]=trainNetwork(train_images,train_target,layers,options);
                save('network.mat','net_pop','v');
    else
        a=input('Do you want to train press 1 to yes:');
            if a==1
                ep=10;%input('enter number of epoch :')
                options=trainingOptions('sgdm','MiniBatchSize',200,'InitialLearnRate',0.0001,...
                    'MaxEpochs',ep,'ExecutionEnvironment','gpu',...
                    'VerboseFrequency',10,'Verbose',true,'L2Regularization',0.001,'Plots','training-progress');
                [net_pop,v]=trainNetwork(train_images,train_target,layers,options);
                delete('network.mat')
                pause(0.001)
                save('network.mat','net_pop','v');
                pause(0.1)
            end
    end
%%testing 
clear;
disp('starting the testing phase ......')
load('network.mat')
load('MNIST_test_img.mat');
load('MNIST_test_label.mat');
[m,n,p,q]=size(test_images);
u=unique(test_labels);
for i=1:size(u,2)
    cu{i}=strcat('c',num2str(u(i)));
end
test_target=categorical(test_labels,u,cu);
p=predict(net_pop,test_images);
[v,p]=max(p');
acc=(sum(p-1==test_labels)/size(test_labels,2))*100;
disp(strcat('Accuracy is :',char({' '}),num2str(acc)))
