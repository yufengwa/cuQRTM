clc;
clear;

%% Size of the Marmousi model.
nx=663;
nz=234;

%% Read ASCII data.
fp=fopen('acc_vp.txt','r');
vp=fread(fp,[nz, nx], 'float');
fclose(fp);

fp=fopen('acc_Qp.txt','r');
Qp=fread(fp,[nz, nx], 'float');
fclose(fp);

%% Converting ASCII data to binary data.
fp=fopen('acc_vp.dat','wb');
fwrite(fp,vp,'float');
fclose(fp);

fp=fopen('acc_Qp.dat','wb');
fwrite(fp,Qp,'float');
fclose(fp);

%% Display the Marmousi models.
figure;
imagesc(vp);
title('Marmousi velocity model');
figure;
imagesc(Qp);
title('Marmousi Q model');