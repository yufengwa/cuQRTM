/* 2D GPU+MPI-based QRTM using decoupled constant Q 
viscoacoustic wave equation proposed by Zhu 2014.

Some basic descriptions of this code are in order.
1) Coordinate configuration of seismic data:

		o---------> x (2nd dim: *.x)
		|
		|
		|
		|
		|
		z (1st dim: *.y)

	1st dim: iz=threadIdx.y+blockDim.y*blockIdx.y;
	2nd dim: ix=threadIdx.x+blockDim.x*blockIdx.x;

	(iz, ix)=iz*ntx+ix;

	(it, iz, ix)=it*ntz*ntx+iz*ntx+ix;

2) stability condition:	

	for second-order pseudo-spectral method:
	dt<2/PI*vmax*sqrt(1/dx^2+1/dy^2+1/dz^2)

	for fourth-order pseudo-spectral method:
	dt<2*sqrt(3)/PI*vmax*sqrt(1/dx^2+1/dy^2+1/dz^2)

3) This code just save the history of forward wavefileds at checpoints in pairs.
	And we save the ONE LAYER four boundaries of every time step and the two final steps of the wavefield.
	Low-pass filtering is used for stabilizing reconstruction.
	It is noteworthy that to implement large scale seismic imaging, pinned memory is 
	employed to save the boundaries of each step so that all the saved data can be computed on the device directly.

4) In our implementation, we employ PSM coupled with LIAO absrobing boundary condition. 
	To make your code fast, you should consider that the GPU codes implementation unit 
	is half-warp (16 threads). The thickness of the boundary should be times of 16. 
	Here we use 16 points at each side.

5) The final images can be two kinds: result of correlation imaging condition and the normalized one. 
	In this code, we use laplacian filtering to remove the low frequency artifacts of the imaging


6) In this code, we set a series of flags to flexibly handle each different situations.

	int Save_Not=0;		Save_Not=1 for save forward, reconstruction and backward wavefileds
						Save_Not=0 for don't save forward, reconstruction and backward wavefileds

	int Sto_Rec=0;		Sto_Rec=1 for wavefiled storage
						Sto_Rec=0 for wavefiled reconstruction

	int Cut_Sub=0;		Cut_Sub=1 for Cut direct wave from records
						Cut_Sub=0 for Substract direct wave by modeling

	int Filtertype=0;	Filtertype=0 for EF-TIF: Cutoff wavenumber is identified by our empirical selection.
											(Empirical time-invaiant filtering)
						Filtertype=1 for FS-TIF: Cutoff wavenumber is identified by 
											the spectrum of reference trace from records.
											(Frequency spectrum based time-invariant filtering)
						Filtertype=2 for TFS-TVF: Cutoff wavenumber is identified by 
											the time-frequency spectrum of reference trace from records.
											(Time-frequency spectrum based time-variant filtering)
						Filtertype=3 for FF-TVF: Cutoff wavenumber is identified by
											the known spectrum of backward wavefield at the previous time step.
											(Feedback based time-variant filtering)
						Filtertype=4 for AS-TVF: Cutoff wavenumber is identified by adaptive stablization.
											(Adaptive stabilization)
											
	int Ckptstype=0;	Ckptstype=0 for Ave-Distribution
						Ckptstype=1 for Log-Distribution
						Ckptstype=2 for Hyb-Distribution 

	int vp_type=0;		vp_type=0 for homogeneous model
						vp_type=1 for ture model
						vp_type=2 for initial model
*/

/*
  Copyright (C) 2017  China University of Petroleum-Beijing (Yufeng Wang)
    Email: hellowangyf@163.com	    

  Acknowledgement:

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

*/

/* 
Some improtant reference: 


*/

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define PI 3.1415926

using namespace std;

#include "Myfunctions.h"



int main(int argc, char* argv[])             
{

	/*=========================================================
	  MPI Indix
	  =========================================================*/

	int myid,numprocs,namelen;
	
	MPI_Comm comm=MPI_COMM_WORLD;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(comm,&myid);
	MPI_Comm_size(comm,&numprocs);
	MPI_Get_processor_name(processor_name,&namelen);

	if(myid==0)
		printf("Number of MPI thread is %d\n",numprocs);


	/*=========================================================
	  Parameters of the time of the system...
	  =========================================================*/

	time_t begin_time, end_time;

	clock_t start, end;


	float runtime=0.0;


	/*=========================================================
	  Flags
	  =========================================================*/

	int Save_Not=0;		//Save_Not=1 for save forward, reconstruction and backward wavefileds
						//Save_Not=0 for don't save forward, reconstruction and backward wavefileds

	int Sto_Rec=0;		//Sto_Rec=1 for wavefiled storage
						//Sto_Rec=0 for wavefiled reconstruction

	int Cut_Sub=0;		//Cut_Sub=1 for Cut direct wave from records
						//Cut_Sub=0 for Substract direct wave by modeling

	int Filtertype=0;	//Filtertype=0 for EF-TIF
						//Filtertype=1 for FS-TIF
						//Filtertype=2 for TFS-TVF
						//Filtertype=3 for FF-TVF
						//Filtertype=4 for AS-TVF
											
	int Ckptstype=0;	//Ckptstype=0 for Ave-Distribution
						//Ckptstype=1 for Log-Distribution
						//Ckptstype=2 for Hyb-Distribution 

	int vp_type=0;		//vp_type=0 for homogeneous model
						//vp_type=1 for ture model
						//vp_type=2 for initial model
	
	/*=========================================================
	  Parameters of Cartesian coordinate...
	  ========================================================*/ 

	int ix,iz,ip,ipp;

	int nz=234;
	int nx=663;	
	int L=20;
	int ntz=nz+2*L;
	int ntx=nx+2*L;
	int nxh=ntx/2;
	int nzh=ntz/2;
	int ntp=ntz*ntx;
	int np=nx*nz;

	float dz=10.0;
	float dx=10.0;

	FILE *fp;
	char filename[40];	

	/*=========================================================
	  Parameters of ricker wave...
	  ========================================================*/

	int it;

	float f0=20.0;
	float t0=1/f0;
	float Omega0=20*PI*f0;
	int nt=2001;
	float dt=0.001;
	float *ricker;	
	ricker=(float *) malloc(nt*sizeof(float));

	if(myid==0)
	{
		ricker_wave(ricker,nt,f0,t0,dt,3); //ricker_derivative
		printf("Ricker wave is done!\n");
	}

	MPI_Bcast(ricker,nt,MPI_FLOAT,0,comm);

	/*=========================================================
	  Parameters of Sources and Receivers...
	  ========================================================*/

  	int is, irx,irz;

	int nsid,modsr,prcs;
	int iss,eachsid,offsets;

	int ds=16;
	int ns0=18; //18;	//the first shot position is L+ns0
	int ns=(ntx-2*L-2*ns0)/ds+1;
	int rnmax=0;
	int nrx_obs=100; //100 receivers every side of shot for backward propagation

	struct Source ss[ns];

	for(is=0;is<ns;is++)
	{
		ss[is].s_ix=is*ds+L+ns0;
		ss[is].s_iz=L+5;
		ss[is].r_iz=L+5;
		ss[is].r_n=nx; //one shot to all receivers

		ss[is].r_ix=(int*)malloc(sizeof(int)*ss[is].r_n);

		for(ip=0;ip<ss[is].r_n;ip++)
		{
			ss[is].r_ix[ip]=L+ip;
		}	

		if(rnmax<ss[is].r_n)
			rnmax=ss[is].r_n;	
	}

	if(myid==0)
	{
		printf("The total shot number is %d\n",ns);
		printf("The maximum trace number for source is %d\n",rnmax);
	}

	/*=========================================================
	  Parameters of checkpoints...
	  ========================================================*/

	int check_steps;
	int N_cp;	
	int *t_cp;

	if(Ckptstype==0)
	{
		check_steps=200;
		N_cp=2*(int)(nt/check_steps);
		t_cp= (int *) malloc(N_cp*sizeof(int));
		for(int icp=0;icp<N_cp;icp++)
		{
			if(icp%2==0)
				t_cp[icp]=check_steps*(icp/2+1);
			else
				t_cp[icp]=t_cp[icp-1]+1;

			if(myid==0)
			{
				printf("checkpoints time is %d\n", t_cp[icp]);
			}
		}			
	}		

	else if(Ckptstype==1)
	{
		check_steps=20;
		N_cp=2*(int)((log(nt/check_steps)/log(2)+1));

		if(pow(2,N_cp/2-1)+pow(2,N_cp/2-2) < nt/check_steps)
			N_cp+=2;

		t_cp= (int *) malloc(N_cp*sizeof(int));

		for(int icp=0;icp<N_cp;icp++)
		{
			if(icp%2==0)
			{
				t_cp[icp]=check_steps*(int)(pow(2,icp/2));
				if(t_cp[icp]> nt)
					t_cp[icp]-=t_cp[icp-2]/2;
			}
			else
				t_cp[icp]=t_cp[icp-1]+1;

			if(myid==0)
			{
				printf("checkpoints time is %d\n", t_cp[icp]);
			}		
		}		
	}

	else if(Ckptstype==2)
	{
		check_steps=200;

		float min_steps=20;

		int min_N_cp;

		min_N_cp=2*(int)((log(check_steps/min_steps)/log(2)+1));

		if(pow(2,min_N_cp/2-1)+pow(2,min_N_cp/2-2) < check_steps/min_steps)
			min_N_cp+=2;

		N_cp= min_N_cp+2*(int)(nt/check_steps);

		t_cp= (int *) malloc(N_cp*sizeof(int*));


		for(int icp=0;icp<min_N_cp;icp++)
		{
			if(icp%2==0)
			{
				t_cp[icp]=min_steps*(int)(pow(2,icp/2));
				if(t_cp[icp]> check_steps)
					t_cp[icp]-=t_cp[icp-2]/2;
			}
			else
				t_cp[icp]=t_cp[icp-1]+1;

			if(myid==0)
			{
				printf("checkpoints time is %d\n", t_cp[icp]);
			}	
		}

		for(int icp=min_N_cp;icp<N_cp;icp++)
		{
			if(icp%2==0)
				t_cp[icp]=check_steps*((icp-min_N_cp)/2+1);
			else
				t_cp[icp]=t_cp[icp-1]+1;

			if(myid==0)
			{
				printf("checkpoints time is %d\n", t_cp[icp]);
			}	
		}
				
	}

	if(myid==0)
	{
		printf("%d Checkpoints are done!\n", N_cp);
	}


	/*=========================================================
	  Parameters of GPU...
	  ========================================================*/

	int i,GPU_N;
	getdevice(&GPU_N);
	printf("The available Device number is %d on %s\n",GPU_N,processor_name);


	struct MultiGPU plan[GPU_N];

	nsid=ns/(GPU_N*numprocs);
	modsr=ns%(GPU_N*numprocs);
	prcs=modsr/GPU_N;
	if(myid<prcs)
	{
		eachsid=nsid+1;
		offsets=myid*(nsid+1)*GPU_N;
	}
	else
	{
		eachsid=nsid;
		offsets=prcs*(nsid+1)*GPU_N+(myid-prcs)*nsid*GPU_N;
	}
	

	/*=========================================================
	  Parameters of model...
	  ========================================================*/

	float *vp, *Qp;
	float *Gamma, averGamma;
	float vp_max,Qp_max;
	float vp_min,Qp_min;
	float avervp;

	vp = (float*)malloc(sizeof(float)*ntp);
	Qp = (float*)malloc(sizeof(float)*ntp);
	Gamma = (float*)malloc(sizeof(float)*ntp);	


	if(myid==0)
	{
		start=clock();

		get_acc_model(vp,Qp,ntp,ntx,ntz,L);

		fp=fopen("./output/acc_vp.dat","wb");
		for(ix=L;ix<=ntx-L-1;ix++)
		{
			for(iz=L;iz<=ntz-L-1;iz++)
			{
				fwrite(&vp[iz*ntx+ix],sizeof(float),1,fp);

			}
		}
		fclose(fp);

		fp=fopen("./output/acc_Qp.dat","wb");
		for(ix=L;ix<=ntx-L-1;ix++)
		{
			for(iz=L;iz<=ntz-L-1;iz++)
			{
				fwrite(&Qp[iz*ntx+ix],sizeof(float),1,fp);

			}
		}
		fclose(fp);

		vp_max=0.0;
		Qp_max=0.0;
		vp_min=5000.0;
		Qp_min=5000.0;

		for(ip=0;ip<ntp;ip++)
		{     
			if(vp[ip]>=vp_max)
			{
				vp_max=vp[ip];
			}
			if(Qp[ip]>=Qp_max)
			{
				Qp_max=fabs(Qp[ip]);
			}
			if(vp[ip]<=vp_min)
			{
				vp_min=vp[ip];
			}
			if(Qp[ip]<=Qp_min)
			{
				Qp_min=fabs(Qp[ip]);
			}
		}

		printf("vp_max = %f\n",vp_max); 
		printf("Qp_max = %f\n",Qp_max);

		printf("vp_min = %f\n",vp_min); 
		printf("Qp_min = %f\n",Qp_min);

		averGamma=0.0;
		avervp=0.0;

		for(iz=0;iz<=ntz-1;iz++)  
			for(ix=0;ix<=ntx-1;ix++)
			{
				Qp[iz*ntx+ix] /= 1;
				Gamma[iz*ntx+ix]=atan(1/Qp[iz*ntx+ix])/PI;
				averGamma+=Gamma[iz*ntx+ix]/(ntx*ntz);
				avervp+=vp[iz*ntx+ix]/(ntx*ntz);		
			}

		printf("The true model is done!\n"); 
	}


	MPI_Bcast(vp, ntp, MPI_FLOAT, 0, comm);	
	MPI_Bcast(Qp, ntp, MPI_FLOAT, 0, comm);
	MPI_Bcast(&vp_max, 1, MPI_FLOAT, 0, comm);
	MPI_Bcast(&Qp_max, 1, MPI_FLOAT, 0, comm);
	MPI_Bcast(&vp_min, 1, MPI_FLOAT, 0, comm);
	MPI_Bcast(&Qp_min, 1, MPI_FLOAT, 0, comm);	
	MPI_Bcast(Gamma, ntp, MPI_FLOAT, 0, comm);
	MPI_Bcast(&averGamma, 1, MPI_FLOAT, 0, comm);
	MPI_Bcast(&avervp, 1, MPI_FLOAT, 0, comm);


	float *Inner_image_cor, *Inner_image_nor, *Final_image_cor, *Final_image_nor;

	Inner_image_cor=(float*)malloc(sizeof(float)*np);
	Inner_image_nor=(float*)malloc(sizeof(float)*np);
	Final_image_cor=(float*)malloc(sizeof(float)*np);
	Final_image_nor=(float*)malloc(sizeof(float)*np);

	memset(Inner_image_cor,0,np*sizeof(float));
	memset(Inner_image_nor,0,np*sizeof(float));
	memset(Final_image_cor,0,np*sizeof(float));
	memset(Final_image_nor,0,np*sizeof(float));

	cuda_Device_malloc(ntx, ntz, ntp, nx, nz, nt, dx, dz, L, rnmax, N_cp, plan, GPU_N);
	cuda_Host_initialization(ntx, ntz, ntp, nx, nz, nt, dx, dz, L, rnmax, N_cp, plan, GPU_N);

	/*=========================================================
	  Filtering and Stabilization parameters
	  ========================================================*/

	float kx_cut=4*2*PI*f0/vp_max;
	float kz_cut=4*2*PI*f0/vp_max;

	float kx_cut_inv=4*2*PI*f0/vp_max;
	float kz_cut_inv=4*2*PI*f0/vp_max;

	float sigma=2.5e-2;
	int Order=2;

	float taper_ratio=0.2;

	float *kfilter, *kstabilization;

	kfilter = (float*)malloc(sizeof(float)*ntp);
	kstabilization = (float*)malloc(sizeof(float)*ntp);

	int beta1, beta2;

	MPI_Barrier(comm);


	time(&begin_time);
	if(myid==0)
		printf("Today's data and time: %s",ctime(&begin_time));



/*

	//=======================================================
	//  Calculate the Observed seismograms...
	//========================================================

	for(iss=0; iss<eachsid; iss++)  
	{
		
		is=offsets+iss*GPU_N;

		beta1=1;
		beta2=1;
		vp_type = 1; // ture model

		Save_Not=0;

		cuda_visco_PSM_2d_forward
		(
			beta1, beta2,
			nt, ntx, ntz, ntp, nx, nz, L, dx, dz, dt,
			vp, Gamma, avervp, averGamma, f0, Omega0, ricker,
			myid, is, ss, plan, GPU_N, rnmax, nrx_obs, N_cp, t_cp,
			kx_cut, kz_cut, sigma, Order, taper_ratio, kfilter, kstabilization,
			Sto_Rec, vp_type, Save_Not
		);

		for(i=0;i<GPU_N;i++)
		{
			sprintf(filename,"./output/%dsource_seismogram_obs.dat",is+i+1);
			fp=fopen(filename,"wb");
			for(ix=0;ix<ss[is+i].r_n;ix++)
			{
				for(it=0;it<nt;it++)
				{
					fwrite(&plan[i].seismogram_obs[it*ss[is+i].r_n+ix],sizeof(float),1,fp);
				}
			}
			fclose(fp);
		}//end GPU

	
		if(Cut_Sub==1)
		{
			for(i=0;i<GPU_N;i++)
			{
				cut_dir(plan[i].seismogram_obs, plan[i].seismogram_rms, rnmax, nt, is, dx, dz, dt, 
					ss[is+i].r_iz, ss[is+i].s_ix, ss[is+i].s_iz, t0, vp);

				sprintf(filename,"./output/%dsource_seismogram_rms.dat",is+i+1);
				fp=fopen(filename,"wb");
				for(ix=0;ix<ss[is+i].r_n;ix++)
				{
					for(it=0;it<nt;it++)
					{
						fwrite(&plan[i].seismogram_rms[it*ss[is+i].r_n+ix],sizeof(float),1,fp);
					}
				}
				fclose(fp);
			}//end GPU				
		}

	}

	if(myid==0)
	{
		printf("seismogram_obs is obtained!\n");
	}

	if(Cut_Sub==1&&myid==0)
	{
		printf("seismogram_rms is obtained!\n");
	}






	//=======================================================
	//  Calculate the direct and rms seismograms...
	//========================================================

	if(Cut_Sub==0)
	{
		if(myid==0)
		{
			get_homo_model(vp,ntp,ntx,ntz,L);
			get_homo_model(Gamma,ntp,ntx,ntz,L);

			fp=fopen("./output/homo_vp.dat","wb");
			for(ix=L;ix<=ntx-L-1;ix++)
			{
				for(iz=L;iz<=ntz-L-1;iz++)
				{
					fwrite(&vp[iz*ntx+ix],sizeof(float),1,fp);
				}
			}
			fclose(fp);

			printf("The homogeneous model is done!\n"); 			
		}

		MPI_Bcast(vp, ntp, MPI_FLOAT, 0, comm);
		MPI_Bcast(Gamma, ntp, MPI_FLOAT, 0, comm);

		for(int iss=0; iss<eachsid; iss++)  
		{
			is=offsets+iss*GPU_N;

			beta1=1;
			beta2=1;
			vp_type = 0; // homogeneous model

			Save_Not=0;

			cuda_visco_PSM_2d_forward
			(
				beta1, beta2,
				nt, ntx, ntz, ntp, nx, nz, L, dx, dz, dt,
				vp, Gamma, avervp, averGamma, f0, Omega0, ricker,
				myid, is, ss, plan, GPU_N, rnmax, nrx_obs, N_cp, t_cp,
				kx_cut, kz_cut, sigma, Order, taper_ratio, kfilter, kstabilization,
				Sto_Rec, vp_type, Save_Not
			);

		
			for(i=0;i<GPU_N;i++)
			{
				sprintf(filename,"./output/%dsource_seismogram_dir.dat",is+i+1);
				fp=fopen(filename,"wb");
				for(ix=0;ix<ss[is+i].r_n;ix++)
				{
					for(it=0;it<nt;it++)
					{
						fwrite(&plan[i].seismogram_dir[it*ss[is+i].r_n+ix],sizeof(float),1,fp);
					}
				}
				fclose(fp);

			}//end GPU
		}

		if(myid==0)
		{
			printf("seismogram_rms is obtained!\n");
		}

	}

*/

	//=======================================================
	//  Construct the forward wavefields and Back-propagate
	//  the RMS seismograms, Meanwhile the images are computed... 
	//========================================================

	if(myid==0)
	{
		printf("====================\n");
		printf("    RTM BEGIN\n");
		printf("====================\n");

		start=clock();
	}


	if(myid==0)
	{
		// obtain the ture model

		get_acc_model(vp,Qp,ntp,ntx,ntz,L);

		for(iz=0;iz<=ntz-1;iz++)  
			for(ix=0;ix<=ntx-1;ix++)
			{
				Qp[iz*ntx+ix] /= 1;
				Gamma[iz*ntx+ix]=atan(1/Qp[iz*ntx+ix])/PI;	
			}
		printf("The true model is done!\n"); 


		// obtain the initial model

		get_ini_model(vp,ntp,ntx,ntz,20);

		fp=fopen("./output/ini_vp.dat","wb");
		for(ix=L;ix<=ntx-L-1;ix++)
		{
			for(iz=L;iz<=ntz-L-1;iz++)
			{
				fwrite(&vp[iz*ntx+ix],sizeof(float),1,fp);
			}
		}
		fclose(fp);

		printf("The initial model is done!\n"); 
	}

	MPI_Bcast(vp, ntp, MPI_FLOAT, 0, comm);
	MPI_Bcast(Gamma, ntp, MPI_FLOAT, 0, comm);




	for(int iss=0; iss<eachsid; iss++)  
	{
		is=offsets+iss*GPU_N;



	
		beta1 = 1;
		beta2 = -1;
		vp_type = 2;	//initial model

		Save_Not=0;

		cuda_visco_PSM_2d_forward
		(
			beta1, beta2,
			nt, ntx, ntz, ntp, nx, nz, L, dx, dz, dt,
			vp, Gamma, avervp, averGamma, f0, Omega0, ricker,
			myid, is, ss, plan, GPU_N, rnmax, nrx_obs, N_cp, t_cp,
			kx_cut, kz_cut, sigma, Order, taper_ratio, kfilter, kstabilization,
			Sto_Rec, vp_type, Save_Not
		);

		for(i=0;i<GPU_N;i++)
		{
			sprintf(filename,"./output/%dsource_seismogram_syn.dat",is+i+1);
			fp=fopen(filename,"wb");
			for(ix=0;ix<ss[is+i].r_n;ix++)
			{
				for(it=0;it<nt;it++)
				{
					fwrite(&plan[i].seismogram_syn[it*ss[is+i].r_n+ix],sizeof(float),1,fp);
				}
			}
			fclose(fp);
		}  


		for(i=0;i<GPU_N;i++)
		{
			if(Cut_Sub==1)
			{
				sprintf(filename,"./output/%dsource_seismogram_rms.dat",is+i+1);
				fp=fopen(filename,"rb");
				for(ix=0;ix<ss[is+i].r_n;ix++)
				{
					for(it=0;it<nt;it++)
					{
						fread(&plan[i].seismogram_rms[it*ss[is+i].r_n+ix],sizeof(float),1,fp);
					}
				}
				fclose(fp);
			}

					
			if(Cut_Sub==0)
			{

				sprintf(filename,"./output/%dsource_seismogram_rms.dat",is+i+1);
				fp=fopen(filename,"rb");
				for(ix=0;ix<ss[is+i].r_n;ix++)
				{
					for(it=0;it<nt;it++)
					{
						fread(&plan[i].seismogram_rms[it*ss[is+i].r_n+ix],sizeof(float),1,fp);
					}
				}
				fclose(fp);
/*

				sprintf(filename,"./output/%dsource_seismogram_dir.dat",is+i+1);
				fp=fopen(filename,"rb");
				for(ix=0;ix<ss[is+i].r_n;ix++)
				{
					for(it=0;it<nt;it++)
					{
						fread(&plan[i].seismogram_dir[it*ss[is+i].r_n+ix],sizeof(float),1,fp);
					}
				}
				fclose(fp);

				sprintf(filename,"./output/%dsource_seismogram_obs.dat",is+i+1);
				fp=fopen(filename,"rb");
				for(ix=0;ix<ss[is+i].r_n;ix++)
				{
					for(it=0;it<nt;it++)
					{
						fread(&plan[i].seismogram_obs[it*ss[is+i].r_n+ix],sizeof(float),1,fp);
					}
				}
				fclose(fp);

				for(ip=0;ip<ss[is+i].r_n*nt;ip++)
				{
					plan[i].seismogram_rms[ip]=plan[i].seismogram_obs[ip]-plan[i].seismogram_dir[ip];
				}

				sprintf(filename,"./output/%dsource_seismogram_rms.dat",is+i+1);
				fp=fopen(filename,"wb");
				for(ix=0;ix<ss[is+i].r_n;ix++)
				{
					for(it=0;it<nt;it++)
					{
						fwrite(&plan[i].seismogram_rms[it*ss[is+i].r_n+ix],sizeof(float),1,fp);
					}
				}
				fclose(fp);
*/
			}
		}


		beta1 = 1;
		beta2 = -1;
		vp_type = 2;	//initial model

		Save_Not=0;


		cuda_visco_PSM_2d_backward
		(
			beta1, beta2,
			nt, ntx, ntz, ntp, nx, nz, L, dx, dz, dt,
			vp, Gamma, avervp, averGamma, f0, Omega0, ricker,
			myid, is, ss, plan, GPU_N, rnmax, nrx_obs, N_cp, t_cp,
			kx_cut, kz_cut, sigma, Order, taper_ratio, kfilter, kstabilization,
			Sto_Rec, Save_Not
		);


		for(i=0;i<GPU_N;i++)
		{

			float image_sources_max=0.0;

			for(ip=0;ip<ntp;ip++)
			{
				if(image_sources_max<fabs(plan[i].image_sources[ip]))
				{
					image_sources_max=fabs(plan[i].image_sources[ip]);
				}
			}

			for(ip=0;ip<ntp;ip++)
			{
				plan[i].image_nor[ip]=plan[i].image_cor[ip]
						/(plan[i].image_sources[ip]+1.0e-5*image_sources_max);
			}

			//Laplace_filtering(plan[i].image_cor,ntx,ntz,dx,dz);
			//Laplace_filtering(plan[i].image_nor,ntx,ntz,dx,dz);

			Laplace_FD_filtering(plan[i].image_cor,ntx,ntz,dx,dz);
			Laplace_FD_filtering(plan[i].image_nor,ntx,ntz,dx,dz);

			sprintf(filename,"./output/image_sources%d.dat",is+i+1);
			fp=fopen(filename,"wb");
			fwrite(&plan[i].image_sources[0],sizeof(float),ntp,fp);
			fclose(fp);

			sprintf(filename,"./output/image_receivers%d.dat",is+i+1);
			fp=fopen(filename,"wb");
			fwrite(&plan[i].image_receivers[0],sizeof(float),ntp,fp);
			fclose(fp);

			sprintf(filename,"./output/image_cor%d.dat",is+i+1);
			fp=fopen(filename,"wb");
			fwrite(&plan[i].image_cor[0],sizeof(float),ntp,fp);
			fclose(fp);

			sprintf(filename,"./output/image_nor%d.dat",is+i+1);
			fp=fopen(filename,"wb");
			fwrite(&plan[i].image_nor[0],sizeof(float),ntp,fp);
			fclose(fp);


			for(iz=L;iz<=ntz-L-1;iz++)
			{
				for(ix=L;ix<=ntx-L-1;ix++)
				{
					ip=iz*ntx+ix;
					ipp=(ix-L)*nz+iz-L;

					Inner_image_cor[ipp]+=plan[i].image_cor[ip]*vp[ip]*vp[ip];
					Inner_image_nor[ipp]+=plan[i].image_nor[ip]*vp[ip]*vp[ip];
				}
			}

		}//end GPU


	}//end is (shotnumbers)

	MPI_Barrier(comm);

	MPI_Allreduce(Inner_image_cor,Final_image_cor,np,MPI_FLOAT,MPI_SUM,comm);
	MPI_Allreduce(Inner_image_nor,Final_image_nor,np,MPI_FLOAT,MPI_SUM,comm);


	//==========================================================
	//  Output the final images,...
	//===========================================================

	if(myid==0)
	{
		sprintf(filename,"./output/Final_image_cor_comp.dat");
		fp=fopen(filename,"wb");
		fwrite(&Final_image_cor[0],sizeof(float),np,fp);
		fclose(fp);

		sprintf(filename,"./output/Final_image_nor_comp.dat");
		fp=fopen(filename,"wb");
		fwrite(&Final_image_nor[0],sizeof(float),np,fp);
		fclose(fp);

	}



	MPI_Barrier(comm);

	if(myid==0)
	{
		printf("====================\n");
		printf("      THE END\n");
		printf("====================\n");

		end=clock();
		printf("The cost of the run time is %f seconds\n",
				(double)(end-start)/CLOCKS_PER_SEC);
	}




	/*==========================================================
	  Free the variables...
	  ===========================================================*/

	cuda_Device_free(ntx, ntz, ntp, nx, nz, nt, dx, dz, L, rnmax, N_cp, plan, GPU_N);

	for(is=0;is<ns;is++)
	{
		free(ss[is].r_ix);
	} 

	free(ricker);
	free(vp); free(Qp); free(Gamma);
	free(t_cp);

	free(kfilter); 
	free(kstabilization);

	free(Inner_image_cor);
	free(Inner_image_nor);
	free(Final_image_cor);
	free(Final_image_nor);

	MPI_Barrier(comm);
	MPI_Finalize();
	

	return 0;
}







/*==========================================================
  This subroutine is used for calculating the ricker wave
  ===========================================================*/

void ricker_wave
(
	float *ricker, int nt, float f0, float t0, float dt, int flag
)
{
	float pi=3.1415927;
	int   it;
	float temp,max=0.0;

	FILE *fp;

	if(flag==1)
	{
		for(it=0;it<nt;it++)
		{
			temp=pi*f0*(it*dt-t0);
			temp=temp*temp;
			ricker[it]=(1.0-2.0*temp)*exp(-temp);
		}

		fp=fopen("./output/ricker.dat","wb");    
		for(it=0;it<nt;it++)
		{
			fwrite(&ricker[it],sizeof(float),1,fp);
		}    
		fclose(fp);
	}

	if(flag==2)
	{
		for(it=0;it<nt;it++)
		{
			temp=pi*f0*(it*dt-t0);
			temp=temp*temp;         
			ricker[it]=(it*dt-t0)*exp(-temp);

			if(max<fabs(ricker[it]))
			{
				max=fabs(ricker[it]);
			}
		}

		for(it=0;it<nt;it++)
		{
			ricker[it]=ricker[it]/max;
		}

		fp=fopen("./output/ricker_integration.dat","wb");    
		for(it=0;it<nt;it++)
		{
			fwrite(&ricker[it],sizeof(float),1,fp);
		}    
		fclose(fp);
	}


	if(flag==3)
	{	
		for(it=0;it<nt;it++)
		{
			temp=pi*f0*(it*dt-t0);
			ricker[it]=(4*powf(pi*f0,4)*powf((it*dt-t0),3)-6*powf(pi*f0,2)*(it*dt-t0))*exp(-powf(temp,2));  

			if(max<fabs(ricker[it]))
			{
				max=fabs(ricker[it]);
			}
		}

		for(it=0;it<nt;it++)
		{
			ricker[it]=ricker[it]/max;
		}

		fp=fopen("./output/ricker_derivative.dat","wb");    
		for(it=0;it<nt;it++)
		{
			fwrite(&ricker[it],sizeof(float),1,fp);
		}    
		fclose(fp);
	}




	return;
}


/*==========================================================
  This subroutine is used for initializing the true model...
  ===========================================================*/

void get_acc_model
(
	float *vp, float *Qp, int ntp, int ntx, int ntz, int L
)
{
	int ip,ipp,iz,ix;
   
	FILE *fp;

	fp=fopen("./input/acc_vp.dat","rb");
	for(ix=L;ix<ntx-L;ix++)
	{
		for(iz=L;iz<ntz-L;iz++)
		{
			ip=iz*ntx+ix;
			fread(&vp[ip],sizeof(float),1,fp);           
		}
	}
	fclose(fp);

	for(iz=0;iz<=L-1;iz++)
	{

		for(ix=0;ix<=L-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=L*ntx+L;

			vp[ip]=vp[ipp];
		}

		for(ix=L;ix<=ntx-L-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=L*ntx+ix;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-L;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=L*ntx+ntx-L-1;

			vp[ip]=vp[ipp];
		}
	}

	for(iz=L;iz<=ntz-L-1;iz++)
	{
		for(ix=0;ix<=L-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+L;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-L;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+ntx-L-1;

			vp[ip]=vp[ipp];
		}

	}

	for(iz=ntz-L;iz<ntz;iz++)
	{

		for(ix=0;ix<=L-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-L-1)*ntx+L;

			vp[ip]=vp[ipp];
		}

		for(ix=L;ix<=ntx-L-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-L-1)*ntx+ix;

			vp[ip]=vp[ipp];
		}

		for(ix=ntx-L;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-L-1)*ntx+ntx-L-1;

			vp[ip]=vp[ipp];
		}
	}

	
	fp=fopen("./input/acc_Qp.dat","rb");
	for(ix=L;ix<ntx-L;ix++)
	{
		for(iz=L;iz<ntz-L;iz++)
		{
			ip=iz*ntx+ix;
			fread(&Qp[ip],sizeof(float),1,fp);

			Qp[ip]=Qp[ip];
		}
	}
	fclose(fp);

	for(iz=0;iz<=L-1;iz++)
	{

		for(ix=0;ix<=L-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=L*ntx+L;

			Qp[ip]=Qp[ipp];
		}

		for(ix=L;ix<=ntx-L-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=L*ntx+ix;

			Qp[ip]=Qp[ipp];
		}

		for(ix=ntx-L;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=L*ntx+ntx-L-1;

			Qp[ip]=Qp[ipp];
		}
	}

	for(iz=L;iz<=ntz-L-1;iz++)
	{
		for(ix=0;ix<=L-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+L;

			Qp[ip]=Qp[ipp];
		}

		for(ix=ntx-L;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=iz*ntx+ntx-L-1;

			Qp[ip]=Qp[ipp];
		}

	}

	for(iz=ntz-L;iz<ntz;iz++)
	{

		for(ix=0;ix<=L-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-L-1)*ntx+L;

			Qp[ip]=Qp[ipp];
		}

		for(ix=L;ix<=ntx-L-1;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-L-1)*ntx+ix;

			Qp[ip]=Qp[ipp];
		}

		for(ix=ntx-L;ix<ntx;ix++)
		{
			ip=iz*ntx+ix;
			ipp=(ntz-L-1)*ntx+ntx-L-1;

			Qp[ip]=Qp[ipp];
		}
	}

	return;
}


/*==========================================================
  This subroutine is used for initializing the homogeneous model...
  ===========================================================*/

void get_homo_model
(
	float *vp, int ntp, int ntx, int ntz, int L
)
{
	int ip,ipp,iz,ix;
   
	FILE *fp;

	for(ix=0;ix<ntx;ix++)
	{
		for(iz=0;iz<ntz;iz++)
		{
			ip=iz*ntx+ix;

			if(iz>L+1)
			{
				ipp=(L+1)*ntx+ix;

				vp[ip]=vp[ipp];
			}
		}
	}
	return;
}



/*==========================================================
  This subroutine is used for initializing the initial model...
  ===========================================================*/

void get_ini_model
(
	float *vp, int ntp, int ntx, int ntz, int span
)
{
	int ix, ixw, ixx;
	int iz, izw, izz;

	float *s_a;
	s_a=(float*)malloc(sizeof(float)*ntp);

	float *sx_a;
	sx_a=(float*)malloc(sizeof(float)*ntp);

	for(iz=0;iz<ntz;iz++)
	{
		for(ix=0; ix<ntx; ix++)
		{
			s_a[iz*ntx+ix]=0.0;
			for(ixw=ix-span;ixw<=ix+span;ixw++)
			{
				if(ixw<0)
					ixx=0;
				else if(ixw>ntx-1)
					ixx=ntx-1;
				else
					ixx=ixw;
				s_a[iz*ntx+ix]+=vp[iz*ntx+ixx]/(2*span+1);
			}		
		}	
	}

	for(iz=0;iz<ntz;iz++)
	{
		for(ix=0; ix<ntx; ix++)
		{		
			sx_a[iz*ntx+ix]=s_a[iz*ntx+ix];		
		}	
	}

	for(iz=0;iz<ntz;iz++)
	{
		for(ix=0; ix<ntx; ix++)
		{
			s_a[iz*ntx+ix]=0.0;
			for(izw=iz-span;izw<=iz+span;izw++)
			{
				if(izw<0)
					izz=0;
				else if(izw>ntz-1)
					izz=ntz-1;
				else
					izz=izw;
				s_a[iz*ntx+ix]+=sx_a[izz*ntx+ix]/(2*span+1);
			}		
		}	
	}

	for(iz=0;iz<ntz;iz++)
	{
		for(ix=0; ix<ntx; ix++)
		{
			vp[iz*ntx+ix]=s_a[iz*ntx+ix];		
		}	
	}

	free(s_a);	
	free(sx_a);		
}


/*==========================================================
  This subroutine is used for cutting the direct wave
  ===========================================================*/

void cut_dir
(
	float *seismogram_obs, float *seismogram_rms, 
	int rnmax, int nt, int is, float dx, float dz, float dt, 
	int r_iz, int s_ix, int s_iz, float t0, float *vp
)
{
	int it, ix;
	float removeline[rnmax];

	for(it=0;it<nt; it++)
	{
		for(ix=0; ix<rnmax; ix++)
		{
			removeline[ix]=(sqrt(powf((ix-s_ix)*dx,2)+powf((r_iz-s_iz)*dz,2))/vp[1*rnmax+ix]+4.0*t0)/dt;

			if(it<removeline[ix])
				seismogram_rms[it*rnmax+ix]=0.0;
			else
				seismogram_rms[it*rnmax+ix]=seismogram_obs[it*rnmax+ix];
		}	
	}
}

/*==========================================================
  This subroutine is used for Laplace filtering
  ===========================================================*/

void Laplace_FD_filtering
(
	float *image, int ntx, int ntz, float dx, float dz
)
{ 
	int ix,iz,ip;
	float diff1, diff2;
	float *tmp;
	tmp = (float*)malloc(sizeof(float)*ntx*ntz);
	memset(tmp, 0, ntx*ntz*sizeof(float));

	for(iz=1;iz<ntz-1;iz++)
	{
		for(ix=1;ix<ntx-1;ix++)
		{
			ip=iz*ntx+ix;

			diff1=(image[ip+ntx]-2.0*image[ip]+image[ip-ntx])/(dz*dz);
			diff2=(image[ip+1]-2.0*image[ip]+image[ip-1])/(dx*dx);	

			tmp[ip]=diff1+diff2;          
		}
	}

	for(iz=0;iz<=ntz-1;iz++)
	{
		for(ix=0;ix<=ntx-1;ix++)
		{
			ip=iz*ntx+ix;

			image[ip]=tmp[ip];          
		}
	}	

	free(tmp);

	return;
}
