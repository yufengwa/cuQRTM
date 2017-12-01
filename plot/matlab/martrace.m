clc,clear;
nx=663;
nz=234;
dx=10;
dz=10;
dt=0.001;
nt=2001;

dk=2*pi*1/(dz*nz);

it=1:nt;
%plot snapshots trace 
fid1=fopen('../../output/Final_image_cor_type0.dat','rb');
fid2=fopen('../../output/Final_image_cor_type1.dat','rb');
fid3=fopen('../../output/Final_image_cor_type2.dat','rb');
fid4=fopen('../../output/Final_image_cor_type3.dat','rb');


ima1=fread(fid1,[nz,nx],'float');
ima2=fread(fid2,[nz,nx],'float');
ima3=fread(fid3,[nz,nx],'float');
ima4=fread(fid4,[nz,nx],'float');

%%
figure(1);

Reftrace=150;

trace1=ima1(:,Reftrace);
trace2=ima2(:,Reftrace);
trace3=ima3(:,Reftrace);
trace4=ima4(:,Reftrace);

xi=1:0.1:length(trace1);

trace11=interp1(trace1,xi,'spline');
trace22=interp1(trace2,xi,'spline');
trace33=interp1(trace3,xi,'spline');
trace44=interp1(trace4,xi,'spline');


plot(trace11, 'k', 'Linewidth', 0.5); hold on;
plot(trace22, 'r','Linewidth', 0.5); hold on;
plot(trace33, 'b', 'Linewidth', 0.5); hold on;
plot(trace44, '--g', 'Linewidth', 0.5); hold on;

legend('Acoustic', 'Attenuated', 'Low-pass filtering','Adaptive stabilization','fontsize',6);

axis([0 10*nz -3e6 3e6]);
xlabel('Depth (m)', 'fontsize',12);
ylabel('Amplitude', 'fontsize',12);
set(gca,'Xtick',[0:500:2500],'Ytick',[-2e6:1e6:2e6],'fontsize',10);
set(gca,'Xticklabel',{[0:500:2000]},'fontsize',10);
set(gcf,'Position',[100 100 600 250]); 
set(gca,'Position',[.10 .15 .85 .75]); 

%%
figure(2);

Reftrace=360;

trace1=ima1(:,Reftrace);
trace2=ima2(:,Reftrace);
trace3=ima3(:,Reftrace);
trace4=ima4(:,Reftrace);

xi=1:0.1:length(trace1);

trace11=interp1(trace1,xi,'spline');
trace22=interp1(trace2,xi,'spline');
trace33=interp1(trace3,xi,'spline');
trace44=interp1(trace4,xi,'spline');


plot(trace11, 'k', 'Linewidth', 0.5); hold on;
plot(trace22, 'r','Linewidth', 0.5); hold on;
plot(trace33, 'b', 'Linewidth', 0.5); hold on;
plot(trace44, '--g', 'Linewidth', 0.5); hold on;


axis([0 10*nz -3e6 3e6]);
xlabel('Depth (m)', 'fontsize',12);
ylabel('Amplitude', 'fontsize',12);
set(gca,'Xtick',[0:500:2500],'Ytick',[-2e6:1e6:2e6],'fontsize',10);
set(gca,'Xticklabel',{[0:500:2000]},'fontsize',10);
set(gcf,'Position',[100 100 600 250]); 
set(gca,'Position',[.10 .15 .85 .75]); 

%%
figure(3);

Reftrace=520;

trace1=ima1(:,Reftrace);
trace2=ima2(:,Reftrace);
trace3=ima3(:,Reftrace);
trace4=ima4(:,Reftrace);

xi=1:0.1:length(trace1);

trace11=interp1(trace1,xi,'spline');
trace22=interp1(trace2,xi,'spline');
trace33=interp1(trace3,xi,'spline');
trace44=interp1(trace4,xi,'spline');


plot(trace11, 'k', 'Linewidth', 0.5); hold on;
plot(trace22, 'r','Linewidth', 0.5); hold on;
plot(trace33, 'b', 'Linewidth', 0.5); hold on;
plot(trace44, '--g', 'Linewidth', 0.5); hold on;


axis([0 10*nz -3e6 3e6]);
xlabel('Depth (m)', 'fontsize',12);
ylabel('Amplitude', 'fontsize',12);
set(gca,'Xtick',[0:500:2500],'Ytick',[-2e6:1e6:2e6],'fontsize',10);
set(gca,'Xticklabel',{[0:500:2000]},'fontsize',10);
set(gcf,'Position',[100 100 600 250]); 
set(gca,'Position',[.10 .15 .85 .75]); 