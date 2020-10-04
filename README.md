# image_classification_for_COVID_wearing_mask_or_not

Make sure Deep learning toolbox is installed

Alexnet and Googlenet is also installed;
 https://www.mathworks.com/matlabcentral/fileexchange/64456-deep-learning-toolbox-model-for-googlenet-network

After downloading the project, go to "code_aspects" folder and run ehza_COVID_alexnet.m to run the pretrained model.
To run googlenet one, run ehza_COVID_googlenet.m

to test the program on static images, run the program 
demo_alex_net_ehz.m(to test it out on googlenet, you need to change it for the specific image size and pretrained model, which usually passed on predict function ) (for example, for alexnet accept image only size of 227 227) 

to test the program on real time using webcam, run demo_in_realtime_duh.m (alexnet only, to test it out for googlenet, you might need to change a few lines accordingly)
