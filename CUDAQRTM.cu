#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda.h>

#define BLOCK_SIZE 16
#define PI 3.1415926

#include "Myfunctions.h"

using namespace std;

struct Multistream
{
	cudaStream_t stream,stream_back;
};



/*==========================================================
  This two subroutines are used for initialization and update
  ===========================================================*/

__global__ void cuda_kernel_initialization
(
	int ntx, int ntz, cufftComplex *u0, cufftComplex *u1, cufftComplex *u2, 
	cufftComplex *uk0, cufftComplex *uk, cufftComplex *Lap, cufftComplex *amp_Lap, cufftComplex *pha_Lap
)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;	

	if(iz>=0&&iz<=ntz-1&&ix>=0&&ix<=ntx-1)
	{
		u0[ip].x=0.0; u0[ip].y=0.0;
		u1[ip].x=0.0; u1[ip].y=0.0;
		u2[ip].x=0.0; u2[ip].y=0.0;
		uk0[ip].x=0.0; uk0[ip].y=0.0; 
		uk[ip].x=0.0; uk[ip].y=0.0; 
		Lap[ip].x=0.0; Lap[ip].y=0.0; 
		amp_Lap[ip].x=0.0; amp_Lap[ip].y=0.0; 
		pha_Lap[ip].x=0.0; pha_Lap[ip].y=0.0; 
	}

	__syncthreads();	

}

__global__ void cuda_kernel_update
(
	int ntx, int ntz, cufftComplex *u0, cufftComplex *u1, cufftComplex *u2
)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;	

	if(iz>=0&&iz<=ntz-1&&ix>=0&&ix<=ntx-1)
	{
		u0[ip].x=u1[ip].x;
		u0[ip].y=u1[ip].y;

		u1[ip].x=u2[ip].x;
		u1[ip].y=u2[ip].y;
	}

	__syncthreads();

}

__global__ void cuda_kernel_initialization_images
(
	int ntx, int ntz, float *image_cor, float *image_nor, float *image_sources, float *image_receivers
)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;	

	if(iz>=0&&iz<=ntz-1&&ix>=0&&ix<=ntx-1)
	{
		image_cor[ip]=0;
		image_nor[ip]=0;
		image_sources[ip]=0;
		image_receivers[ip]=0;
	}

	__syncthreads();
}
/*=========================================================
  This two subroutines are regularization kernel 
  ========================================================*/

__global__ void cuda_kernel_AdaSta
(
	int it, int ntx, int ntz, float dx, float dz, float dt, float *kstabilization,
	float *vp, float *Gamma, float avervp, float averGamma, float Omega0,
	float *kx, float *kz, cufftComplex *uk, float sigma, int Order
)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;

	float tau, ksi;


	if(iz>=0&&iz<=ntz-1&&ix>=0&&ix<=ntx-1)
	{
		//tau = -powf(vp[ip],2*Gamma[ip]-1)*powf(Omega0,-2*Gamma[ip])*sin(Gamma[ip]*PI);
		//ksi = -tau*powf(vp[ip]*cos(Gamma[ip]*PI/2),2)*powf((powf(kx[ix],2)+powf(kz[iz],2)), averGamma+0.5)/2;

		tau = -powf(avervp,2*averGamma-1)*powf(Omega0,-2*averGamma)*sin(averGamma*PI);
		ksi = -tau*powf(avervp*cos(averGamma*PI/2),2)*powf((powf(kx[ix],2)+powf(kz[iz],2)), averGamma+0.5)/2;

		kstabilization[ip] = (1+sigma*exp(Order*ksi*dt*it))/(1+sigma*exp(Order*ksi*dt*(it+1)));

		uk[ip].x *= kstabilization[ip]/(ntx*ntz);
		uk[ip].y *= kstabilization[ip]/(ntx*ntz);				
	}

	__syncthreads();

}

__global__ void cuda_kernel_filter2d
(
	int ntx, int ntz, float dx, float dz, float *kfilter, float *kx, float *kz, cufftComplex *uk, 
	float kx_cut, float kz_cut, float taper_ratio
)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;

	float xc=kx_cut*ntx*dx;
	float zc=kz_cut*ntz*dz;

	float xs=(1-taper_ratio)*xc;
	float zs=(1-taper_ratio)*zc;

	int nxh=ntx/2;
	int nzh=ntz/2;	

	if(iz>=0&&iz<=ntz-1&&ix>=0&&ix<=ntx-1)
	{
		kfilter[ip] = 0;	

		if(ix>=0&&ix<xs&&iz>=0&&iz<ntz)
		{
			kfilter[ip]=1.0;
		}
		else if(ix>=xs&&ix<xc&&iz>=0&&iz<ntz)
		{
			kfilter[ip]=cos(PI/2.0*(ix-xs)/(xc-xs));
		}
		else if(ix>=xc&&ix<=nxh&&iz>=0&&iz<ntz)
		{
			kfilter[ip]=0.0;
		}

		else if(ix>=nxh&&ix<ntx-xc&&iz>=0&&iz<ntz)
		{
			kfilter[ip]=0.0;
		}
		else if(ix>=ntx-xc&&ix<ntx-xs&&iz>=0&&iz<ntz)
		{
			kfilter[ip]=sin(PI/2.0*(ix-(ntx-xc))/(xc-xs));
		}
		else if(ix>=ntx-xs&&ix<ntx&&iz>=0&&iz<ntz)
		{
			kfilter[ip]=1.0;
		}


		if(iz>=0&&iz<zs&&ix>=0&&ix<ntx)
		{
			kfilter[ip]*=1.0;
		}
		else if(iz>=zs&&iz<zc&&ix>=0&&ix<ntx)
		{
			kfilter[ip]*=cos(PI/2.0*(iz-zs)/(zc-zs));
		}
		else if(iz>=zc&&iz<=nzh&&ix>=0&&ix<ntx)
		{
			kfilter[ip]*=0.0;
		}	

		else if(iz>=nzh&&iz<ntz-zc&&ix>=0&&ix<ntx)
		{
			kfilter[ip]*=0.0;
		}
		else if(iz>=ntz-zc&&iz<ntz-zs&&ix>=0&&ix<ntx)
		{
			kfilter[ip]*=sin(PI/2.0*(iz-(ntz-zc))/(zc-zs));
		}
		else if(iz>=ntz-zs&&iz<ntz&&ix>=0&&ix<ntx)
		{
			kfilter[ip]*=1.0;
		}

	}

	if(iz>=0&&iz<=ntz-1&&ix>=0&&ix<=ntx-1)
	{
		uk[ip].x *= kfilter[ip];
		uk[ip].y *= kfilter[ip];		
	}


	__syncthreads();

}




/*==========================================================
  This subroutine is used for defining k
  ===========================================================*/

__global__ void cuda_kernel_k_define
(
	int ntx, int ntz, float dx, float dz, float *kx, float *kz
)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int nxh=ntx/2;
	int nzh=ntz/2;

	float dkx=1.0/(ntx*dx);
	float dkz=1.0/(ntz*dz);

	if(ix>=0&&ix<=nxh)
	{
		kx[ix]=2*PI*ix*dkx;
	}
	if(ix>=nxh&&ix<ntx)
	{
		kx[ix]=kx[ntx-ix];
	}
	if(iz>=0&&iz<=nzh)
	{
		kz[iz]=2*PI*iz*dkz;
	}
	if(iz>=nzh&&iz<ntz)
	{
		kz[iz]=kz[ntz-iz];
	}

	__syncthreads();

}


/*==========================================================
  This subroutine is used for calculating forward wavefileds in k-space
  ===========================================================*/

__global__ void cuda_kernel_visco_PSM_2d_forward_k_space
(
	float beta1, float beta2,
	int it, int nt, int ntx, int ntz, float dx, float dz, float dt, 
	float *vp, float *Gamma, float averGamma, float f0, float Omega0, 
	float *kx, float *kz, 
	cufftComplex *uk, cufftComplex *uk0, 
	cufftComplex *Lap_uk, cufftComplex *amp_uk, cufftComplex *pha_uk
)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;


	if(iz>=0&&iz<=ntz-1&&ix>=0&&ix<=ntx-1)
	{
		Lap_uk[ip].x=-(powf(kx[ix],2)+powf(kz[iz],2))*uk[ip].x;
		Lap_uk[ip].y=-(powf(kx[ix],2)+powf(kz[iz],2))*uk[ip].y;

		if(beta1!=0)
		{
			pha_uk[ip].x=powf((powf(kx[ix],2)+powf(kz[iz],2)), averGamma+1)*uk[ip].x;
			pha_uk[ip].y=powf((powf(kx[ix],2)+powf(kz[iz],2)), averGamma+1)*uk[ip].y;
		}

		if(beta2!=0)
		{
			amp_uk[ip].x=powf((powf(kx[ix],2)+powf(kz[iz],2)), averGamma+0.5)*(uk[ip].x-uk0[ip].x)/dt;
			amp_uk[ip].y=powf((powf(kx[ix],2)+powf(kz[iz],2)), averGamma+0.5)*(uk[ip].y-uk0[ip].y)/dt;
		}					

		uk0[ip].x=uk[ip].x;
		uk0[ip].y=uk[ip].y;
	}

	__syncthreads();

}

/*==========================================================
  This subroutine is used for calculating forward wavefileds in x-space
  ===========================================================*/

__global__ void cuda_kernel_visco_PSM_2d_forward_x_space
(
	float beta1, float beta2,
	int it, int nt, int ntx, int ntz, int nx, int nz, int L, float dx, float dz, float dt, 
	float *vp, float *Gamma, float averGamma, float f0, float Omega0, 
	float *seismogram, int *r_ix, int r_iz, int rnmax, float *ricker, int s_ix, int s_iz,
	cufftComplex *u0, cufftComplex *u1, cufftComplex *u2,
	cufftComplex *Lap, cufftComplex *amp_Lap, cufftComplex *pha_Lap,
	float *borders_up, float *borders_bottom, float *borders_left, float *borders_right,
	float *u2_final0, float *u2_final1,
	int Sto_Rec, int vp_type
)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;

	int icp;

	float eta, tau;


	if(iz>=0&&iz<=ntz-1&&ix>=0&&ix<=ntx-1)
	{
		
		eta= -powf(vp[ip],2*Gamma[ip])*powf(Omega0,-2*Gamma[ip])*cos(Gamma[ip]*PI);
		tau= -powf(vp[ip],2*Gamma[ip]-1)*powf(Omega0,-2*Gamma[ip])*sin(Gamma[ip]*PI);

		// scale ...
		Lap[ip].x=Lap[ip].x/(ntx*ntz);
		pha_Lap[ip].x=pha_Lap[ip].x/(ntx*ntz);
		amp_Lap[ip].x=amp_Lap[ip].x/(ntx*ntz);

		u2[ip].x=powf(vp[ip]*cos(Gamma[ip]*PI/2),2)*powf(dt,2)
			*(
				Lap[ip].x
				+beta1*(eta*pha_Lap[ip].x-Lap[ip].x)
				+beta2*tau*amp_Lap[ip].x
			)
			+2*u1[ip].x-u0[ip].x;
	}


	// Ricker...  
	if(iz==s_iz&&ix==s_ix)
	{
		u2[ip].x+=ricker[it];
	}	

	// Seismogram...  
	if(ix>=r_ix[0]&&ix<=r_ix[rnmax-1]&&iz==r_iz)
		seismogram[it*rnmax+ix-r_ix[0]]=u2[ip].x;


	// store the Checkpoints, borders and final two-step wavefileds...

	if(Sto_Rec==0&&vp_type==2)
	{

		if(ix>=L&&ix<=ntx-L-1&&iz==L)
		{
			borders_up[it*nx+ix-L]=u2[ip].x;
		}
		if(ix>=L&&ix<=ntx-L-1&&iz==ntz-L-1)
		{
			borders_bottom[it*nx+ix-L]=u2[ip].x;
		}
		if(iz>=L&&iz<=ntz-L-1&&ix==L)
		{
			borders_left[it*nz+iz-L]=u2[ip].x;
		}
		if(iz>=L&&iz<=ntz-L-1&&ix==ntx-L-1)
		{
			borders_right[it*nz+iz-L]=u2[ip].x;
		}

		if(it==nt-1)
		{
			if(iz>=0&&iz<=ntz-1&&ix>=0&&ix<=ntx-1)
			{				
				u2_final0[ip]=u2[ip].x;
				u2_final1[ip]=u1[ip].x;				
			}			
		}

	}

	__syncthreads();

}


/*==========================================================
  This subroutine is used for writing checkpoints
  ===========================================================*/

__global__ void cuda_kernel_checkpoints_Out
(
	int it, int nt, int ntx, int ntz, int nx, int nz, int L, float dx, float dz, float dt, 
	cufftComplex *u1, cufftComplex *u2,
	float *u_cp, int N_cp, int *t_cp
)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;
	int icp;

	for(icp=0;icp<N_cp;icp++)
	{
		if(icp%2==1&&it==t_cp[icp])
		{
			if(iz>=0&&iz<=ntz-1&&ix>=0&&ix<=ntx-1)
			{
				u_cp[icp*ntx*ntz+ip]=u2[ip].x;
				u_cp[(icp-1)*ntx*ntz+ip]=u1[ip].x;
			}
		}
	}

	__syncthreads();

}


/*==========================================================
  This two subroutines are used for initializing Final two wavefileds
  ===========================================================*/

__global__ void cuda_kernel_initialization_Finals
(
	int ntx, int ntz, cufftComplex *u0, cufftComplex *u1, float *u2_final0, float *u2_final1
)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;	

	if(iz>=0&&iz<=ntz-1&&ix>=0&&ix<=ntx-1)
	{
		u0[ip].x=u2_final0[ip];
		u1[ip].x=u2_final1[ip];
	}

	__syncthreads();	

}


/*==========================================================
  This subroutine is used for calculating reconstructed wavefileds in k-space
  ===========================================================*/

__global__ void cuda_kernel_visco_PSM_2d_reconstruction_k_space
(
	float beta1, float beta2,
	int it, int nt, int ntx, int ntz, int nx, int nz, int L, float dx, float dz, float dt, 
	float *vp, float *Gamma, float averGamma, float f0, float Omega0, 
	float *kx, float *kz, 
	cufftComplex *uk, cufftComplex *uk0, 
	cufftComplex *Lap_uk, cufftComplex *amp_uk, cufftComplex *pha_uk
)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;

	if(iz>=0&&iz<=ntz-1&&ix>=0&&ix<=ntx-1)
	{
		Lap_uk[ip].x=-(powf(kx[ix],2)+powf(kz[iz],2))*uk[ip].x;
		Lap_uk[ip].y=-(powf(kx[ix],2)+powf(kz[iz],2))*uk[ip].y;

		if(beta1!=0)
		{
			pha_uk[ip].x=powf((powf(kx[ix],2)+powf(kz[iz],2)), averGamma+1)*uk[ip].x;
			pha_uk[ip].y=powf((powf(kx[ix],2)+powf(kz[iz],2)), averGamma+1)*uk[ip].y;
		}

		if(beta2!=0)
		{
			amp_uk[ip].x=powf((powf(kx[ix],2)+powf(kz[iz],2)), averGamma+0.5)*(uk[ip].x-uk0[ip].x)/dt;
			amp_uk[ip].y=powf((powf(kx[ix],2)+powf(kz[iz],2)), averGamma+0.5)*(uk[ip].y-uk0[ip].y)/dt;
		}					

		uk0[ip].x=uk[ip].x;
		uk0[ip].y=uk[ip].y;			
	}

	__syncthreads();

}


/*==========================================================
  This subroutine is used for calculating reconstructed wavefileds in x-space
  ===========================================================*/

__global__ void cuda_kernel_visco_PSM_2d_reconstruction_x_space
(
	float beta1, float beta2,
	int it, int nt, int ntx, int ntz, int nx, int nz, int L, float dx, float dz, float dt, 
	float *vp, float *Gamma, float averGamma, float f0, float Omega0, 
	float *ricker, int s_ix, int s_iz,
	cufftComplex *u0, cufftComplex *u1, cufftComplex *u2,
	cufftComplex *Lap, cufftComplex *amp_Lap, cufftComplex *pha_Lap,
	float *borders_up, float *borders_bottom, float *borders_left, float *borders_right
)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;
	int icp;

	float eta, tau;


	if(iz>=0&&iz<=ntz-1&&ix>=0&&ix<=ntx-1)
	{
		
		eta= -powf(vp[ip],2*Gamma[ip])*powf(Omega0,-2*Gamma[ip])*cos(Gamma[ip]*PI);
		tau= -powf(vp[ip],2*Gamma[ip]-1)*powf(Omega0,-2*Gamma[ip])*sin(Gamma[ip]*PI);

		// scale ...
		Lap[ip].x=Lap[ip].x/(ntx*ntz);
		pha_Lap[ip].x=pha_Lap[ip].x/(ntx*ntz);
		amp_Lap[ip].x=amp_Lap[ip].x/(ntx*ntz);

		u2[ip].x=powf(vp[ip]*cos(Gamma[ip]*PI/2),2)*powf(dt,2)
			*(
				Lap[ip].x
				+beta1*(eta*pha_Lap[ip].x-Lap[ip].x)
				+beta2*tau*amp_Lap[ip].x
			)
			+2*u1[ip].x-u0[ip].x;
	}

/*
	// Ricker...  
	if(iz==s_iz&&ix==s_ix)
	{
		u2[ip].x-=ricker[it];
	}
*/


	// borders ...
	if(ix>=L&&ix<=ntx-L-1&&iz==L)
	{
		u2[ip].x=borders_up[it*nx+ix-L];
	}
	if(ix>=L&&ix<=ntx-L-1&&iz==ntz-L-1)
	{
		u2[ip].x=borders_bottom[it*nx+ix-L];
	}
	if(iz>=L&&iz<=ntz-L-1&&ix==L)
	{
		u2[ip].x=borders_left[it*nz+iz-L];
	}
	if(iz>=L&&iz<=ntz-L-1&&ix==ntx-L-1)
	{
		u2[ip].x=borders_right[it*nz+iz-L];
	}



	__syncthreads();

}


/*==========================================================
  This subroutine is used for reading checkpoints
  ===========================================================*/

__global__ void cuda_kernel_checkpoints_In
(
	int it, int nt, int ntx, int ntz, int nx, int nz, int L, float dx, float dz, float dt, 
	cufftComplex *u1, cufftComplex *u2,
	float *u_cp, int N_cp, int *t_cp
)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;
	int icp;

	for(icp=0;icp<N_cp;icp++)
	{
		if(icp%2==0&&it==t_cp[icp])
		{
			if(iz>=0&&iz<=ntz-1&&ix>=0&&ix<=ntx-1)
			{
				u2[ip].x=u_cp[icp*ntx*ntz+ip];
				u1[ip].x=u_cp[(icp+1)*ntx*ntz+ip];
			}
		}
	}

	__syncthreads();

}


/*==========================================================
  This subroutine is used for calculating backward wavefileds in k-space
  ===========================================================*/

__global__ void cuda_kernel_visco_PSM_2d_backward_k_space
(
	float beta1, float beta2,
	int it, int nt, int ntx, int ntz, float dx, float dz, float dt, 
	float *vp, float *Gamma, float averGamma, float f0, float Omega0, 
	float *kx, float *kz, 
	cufftComplex *uk, cufftComplex *uk0, 
	cufftComplex *Lap_uk, cufftComplex *amp_uk, cufftComplex *pha_uk
)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;

	if(iz>=0&&iz<=ntz-1&&ix>=0&&ix<=ntx-1)
	{
		Lap_uk[ip].x=-(powf(kx[ix],2)+powf(kz[iz],2))*uk[ip].x;
		Lap_uk[ip].y=-(powf(kx[ix],2)+powf(kz[iz],2))*uk[ip].y;

		if(beta1!=0)
		{
			pha_uk[ip].x=powf((powf(kx[ix],2)+powf(kz[iz],2)), averGamma+1)*uk[ip].x;
			pha_uk[ip].y=powf((powf(kx[ix],2)+powf(kz[iz],2)), averGamma+1)*uk[ip].y;
		}

		if(beta2!=0)
		{
			amp_uk[ip].x=powf((powf(kx[ix],2)+powf(kz[iz],2)), averGamma+0.5)*(uk[ip].x-uk0[ip].x)/dt;
			amp_uk[ip].y=powf((powf(kx[ix],2)+powf(kz[iz],2)), averGamma+0.5)*(uk[ip].y-uk0[ip].y)/dt;
		}					

		uk0[ip].x=uk[ip].x;
		uk0[ip].y=uk[ip].y;			
	}

	__syncthreads();

}


/*==========================================================
  This subroutine is used for calculating backward wavefileds in x-space
  ===========================================================*/

__global__ void cuda_kernel_visco_PSM_2d_backward_x_space
(
	float beta1, float beta2,
	int it, int nt, int ntx, int ntz, float dx, float dz, float dt, 
	float *vp, float *Gamma, float averGamma, float f0, float Omega0, 
	float *seismogram_rms, int *r_ix, int r_iz, int s_ix, int rnmax, int nrx_obs,
	cufftComplex *u0, cufftComplex *u1, cufftComplex *u2,
	cufftComplex *Lap, cufftComplex *amp_Lap, cufftComplex *pha_Lap
)
{

	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;

	float eta, tau;

	if(iz>=0&&iz<=ntz-1&&ix>=0&&ix<=ntx-1)
	{
		eta= -powf(vp[ip],2*Gamma[ip])*powf(Omega0,-2*Gamma[ip])*cos(Gamma[ip]*PI);
		tau= -powf(vp[ip],2*Gamma[ip]-1)*powf(Omega0,-2*Gamma[ip])*sin(Gamma[ip]*PI);

		// scale ...
		Lap[ip].x=Lap[ip].x/(ntx*ntz);
		pha_Lap[ip].x=pha_Lap[ip].x/(ntx*ntz);
		amp_Lap[ip].x=amp_Lap[ip].x/(ntx*ntz);

		u2[ip].x=powf(vp[ip]*cos(Gamma[ip]*PI/2),2)*powf(dt,2)
			*(
				Lap[ip].x
				+beta1*(eta*pha_Lap[ip].x-Lap[ip].x)
				+beta2*tau*amp_Lap[ip].x
			)
			+2*u1[ip].x-u0[ip].x;	
	}

	// Seismogram...  
	//if(ix>=r_ix[0]&&ix<=r_ix[rnmax-1]&&iz==r_iz)
		//u2[ip].x=seismogram_rms[it*rnmax+ix-r_ix[0]];


	int irx_min = s_ix-nrx_obs;
	int irx_max = s_ix+nrx_obs;

	if(irx_min<r_ix[0])
		irx_min = r_ix[0];
	if(irx_max>r_ix[rnmax-1])
		irx_max = r_ix[rnmax-1];

	if(ix>=irx_min&&ix<=irx_max&&iz==r_iz)
		u2[ip].x=seismogram_rms[it*rnmax+ix-r_ix[0]];


	__syncthreads();

}


/*==========================================================
  This subroutine is used for imaging
  ===========================================================*/

__global__ void cuda_kernel_image
(
	int ntx, int ntz, int L,
	cufftComplex *u2_inv, cufftComplex *u2,
	float *image_cor, float *image_sources, float *image_receivers
)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;


	if(iz>=L&&iz<=ntz-L-1&&ix>=L&&ix<=ntx-L-1)
	{
		image_cor[ip]+=u2_inv[ip].x*u2[ip].x;
		image_sources[ip]+=u2_inv[ip].x*u2_inv[ip].x;
		image_receivers[ip]+=u2[ip].x*u2[ip].x; 
     
	}
	__syncthreads();

}

/*==========================================================
  This subroutine is used for absorbing boundary condition
  ===========================================================*/

__global__ void cuda_kernel_MTF_2nd
(
	int L, int ntx, int ntz, float dx, float dz, float dt, 
	float *vp, cufftComplex *u0, cufftComplex *u1, cufftComplex *u2
)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iz=by*BLOCK_SIZE+ty;
	int ix=bx*BLOCK_SIZE+tx;

	int ip=iz*ntx+ix;
	int ipp=iz*ntx+(ntx-1-ix);
	int ippp=(ntz-1-iz)*ntx+ix;

	float alpha=1.0;
	float w, s, t1, t2, t3;


	// left ABC ...

	if(ix>=0&&ix<=L-1&&iz>=0&&iz<=ntz-1)
	{
		w=1-1.0*ix/L;

		s=alpha*vp[ip]*dt/dx;
		t1=(2-s)*(1-s)/2;
		t2=s*(2-s);
		t3=s*(s-1)/2;	

		u2[ip].x=w*
			(
				(1*2)*
				(
					t1*u1[ip].x+t2*u1[ip+1].x+t3*u1[ip+2].x
				)
				+(-1*1)*
				(
					t1*t1*u0[ip].x
					+2*t1*t2*u0[ip+1].x
					+(2*t1*t3+t2*t2)*u0[ip+2].x
					+2*t2*t3*u0[ip+3].x
					+t3*t3*u0[ip+4].x
				)
			)
			+(1-w)*u2[ip].x;								
	}

	// right ABC ...

	if(ix>=ntx-L&&ix<=ntx-1&&iz>=0&&iz<=ntz-1)
	{
		w=1-1.0*(ntx-1-ix)/L;

		s=alpha*vp[ip]*dt/dx;
		t1=(2-s)*(1-s)/2;
		t2=s*(2-s);
		t3=s*(s-1)/2;			


		u2[ip].x=w*
				(
					(1*2)*
					(
						t1*u1[ip].x
						+t2*u1[ip-1].x
						+t3*u1[ip-2].x
					)
					+(-1*1)*
					(
						t1*t1*u0[ip].x
						+2*t1*t2*u0[ip-1].x
						+(2*t1*t3+t2*t2)*u0[ip-2].x
						+2*t2*t3*u0[ip-3].x
						+t3*t3*u0[ip-4].x
					)
				)
				+(1-w)*u2[ip].x;			
						
	}


	// up ABC ...

	if(iz>=0&&iz<=L-1&&ix>=0&&ix<=ntx-1)
	{
		w=1-1.0*iz/L;
		s=alpha*vp[ip]*dt/dz;	
		t1=(2-s)*(1-s)/2;
		t2=s*(2-s);
		t3=s*(s-1)/2;

		u2[ip].x=w*
			(
				(1*2)*
				(
					t1*u1[ip].x
					+t2*u1[ip+ntx].x
					+t3*u1[ip+2*ntx].x
				)
				+(-1*1)*
				(
					t1*t1*u0[ip].x
					+2*t1*t2*u0[ip+ntx].x
					+(2*t1*t3+t2*t2)*u0[ip+2*ntx].x
					+2*t2*t3*u0[ip+3*ntx].x
					+t3*t3*u0[ip+4*ntx].x
				)
			)
			+(1-w)*u2[ip].x;			
	}

	// bottom ABC ...

	if(iz>=ntz-L&&iz<=ntz-1&&ix>=0&&ix<=ntx-1)
	{
		w=1-1.0*(ntz-1-iz)/L;
		s=alpha*vp[ip]*dt/dz;	
		t1=(2-s)*(1-s)/2;
		t2=s*(2-s);
		t3=s*(s-1)/2;

		u2[ip].x=w*
			(
				(1*2)*
				(
					t1*u1[ip].x
					+t2*u1[ip-ntx].x
					+t3*u1[ip-2*ntx].x
				)
				+(-1*1)*
				(
					t1*t1*u0[ip].x
					+2*t1*t2*u0[ip-ntx].x
					+(2*t1*t3+t2*t2)*u0[ip-2*ntx].x
					+2*t2*t3*u0[ip-3*ntx].x
					+t3*t3*u0[ip-4*ntx].x
				)
			)
			+(1-w)*u2[ip].x;		
	}

	__syncthreads();
	
}


/*==========================================================
  This two subroutines are used for forward and backward modeling
  ===========================================================*/

extern "C"
void cuda_visco_PSM_2d_forward
(
	int beta1, int beta2,
	int nt, int ntx, int ntz, int ntp, int nx, int nz, int L, float dx, float dz, float dt,
	float *vp, float *Gamma, float avervp, float averGamma, float f0, float Omega0, float *ricker,
	int myid, int is, struct Source ss[], struct MultiGPU plan[], int GPU_N, int rnmax, int nrx_obs, int N_cp, int *t_cp,
	float kx_cut, float kz_cut, float sigma, int Order,	float taper_ratio, float *kfilter, float *kstabilization,
	int Sto_Rec, int vp_type, int Save_Not
)
{
	int i, it, ix, iz, ip, icp;

	size_t size_model=sizeof(float)*ntp;

	FILE *fp;
	char filename[40];

	float *u2_real;
	u2_real = (float*)malloc(sizeof(float)*ntp);


	Multistream plans[GPU_N];

	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);
		cudaStreamCreate(&plans[i].stream);	
		cufftSetStream(plan[i].PLAN_FORWARD,plans[i].stream);
		cufftSetStream(plan[i].PLAN_BACKWARD,plans[i].stream);
	}	

	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 dimGrid((ntx+dimBlock.x-1)/dimBlock.x,(ntz+dimBlock.y-1)/dimBlock.y);

	// copy the vectors from the host to the device
	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		cudaMemcpyAsync(plan[i].d_r_ix,ss[is+i].r_ix,sizeof(float)*rnmax,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_ricker,ricker,sizeof(float)*nt,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_vp,vp,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_Gamma,Gamma,size_model,cudaMemcpyHostToDevice,plans[i].stream);

		cudaMemcpyAsync(plan[i].d_t_cp,t_cp,N_cp*sizeof(int),cudaMemcpyHostToDevice,plans[i].stream);
	}


	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		cuda_kernel_initialization<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(ntx, ntz, plan[i].d_u0, plan[i].d_u1, plan[i].d_u2, plan[i].d_uk0, plan[i].d_uk, 
			plan[i].d_Lap, plan[i].d_amp_Lap, plan[i].d_pha_Lap);

		cuda_kernel_k_define<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(ntx, ntz, dx, dz, plan[i].d_kx, plan[i].d_kz);

	}


	for(it=0;it<nt;it++)  
	{
		for(i=0;i<GPU_N;i++)
		{
			cudaSetDevice(i);

			cufftExecC2C(plan[i].PLAN_FORWARD,plan[i].d_u1,plan[i].d_uk,CUFFT_FORWARD); //CUFFT_FORWARD

			cuda_kernel_visco_PSM_2d_forward_k_space<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
					beta1, beta2,
					it, nt, ntx, ntz, dx, dz, dt, 
					plan[i].d_vp, plan[i].d_Gamma, averGamma, f0, Omega0,
					plan[i].d_kx, plan[i].d_kz,  
					plan[i].d_uk, plan[i].d_uk0, 
					plan[i].d_Lap_uk, plan[i].d_amp_uk, plan[i].d_pha_uk
				);		

			cufftExecC2C(plan[i].PLAN_BACKWARD,plan[i].d_Lap_uk,plan[i].d_Lap,CUFFT_INVERSE); //CUFFT_INVERSE


			if(beta1!=0)
			{	
				cufftExecC2C(plan[i].PLAN_BACKWARD,plan[i].d_pha_uk,plan[i].d_pha_Lap,CUFFT_INVERSE); //CUFFT_INVERSE					
			}

			if(beta2!=0)
			{
				if (beta2<0)
				{
					cuda_kernel_filter2d<<<dimGrid,dimBlock,0,plans[i].stream>>>
						(ntx, ntz, dx, dz, plan[i].d_kfilter, plan[i].d_kx, plan[i].d_kz, plan[i].d_amp_uk, kx_cut, kz_cut, taper_ratio);
				}

				cufftExecC2C(plan[i].PLAN_BACKWARD,plan[i].d_amp_uk,plan[i].d_amp_Lap,CUFFT_INVERSE); //CUFFT_INVERSE
			}


			cuda_kernel_visco_PSM_2d_forward_x_space<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
					beta1, beta2,
					it, nt, ntx, ntz, nx, nz, L, dx, dz, dt, 
					plan[i].d_vp, plan[i].d_Gamma, averGamma, f0, Omega0,
					plan[i].d_seismogram, plan[i].d_r_ix, ss[is+i].r_iz, rnmax, plan[i].d_ricker, ss[is+i].s_ix, ss[is+i].s_iz,
					plan[i].d_u0, plan[i].d_u1, plan[i].d_u2,
					plan[i].d_Lap, plan[i].d_amp_Lap, plan[i].d_pha_Lap,
					plan[i].d_borders_up, plan[i].d_borders_bottom, plan[i].d_borders_left, plan[i].d_borders_right,
					plan[i].d_u2_final0, plan[i].d_u2_final1,
					Sto_Rec, vp_type				
				);

			
			if(beta2<0)
			{
				cufftExecC2C(plan[i].PLAN_FORWARD,plan[i].d_u2,plan[i].d_uk,CUFFT_FORWARD); //CUFFT_FORWARD

				cuda_kernel_AdaSta<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
					it, ntx, ntz, dx, dz, dt, plan[i].d_kstabilization,
					plan[i].d_vp, plan[i].d_Gamma, avervp, averGamma, Omega0,
					plan[i].d_kx, plan[i].d_kz, plan[i].d_uk, sigma, Order
				);	

				cufftExecC2C(plan[i].PLAN_BACKWARD,plan[i].d_uk,plan[i].d_u2,CUFFT_INVERSE); //CUFFT_INVERSE	
			}
			


			cuda_kernel_MTF_2nd<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(L, ntx, ntz, dx, dz, dt, plan[i].d_vp, plan[i].d_u0, plan[i].d_u1, plan[i].d_u2);


			if(Sto_Rec==0&&vp_type==2)
			{
				cuda_kernel_checkpoints_Out<<<dimGrid,dimBlock,0,plans[i].stream>>>
					(
						it, nt, ntx, ntz, nx, nz, L, dx, dz, dt, 
						plan[i].d_u1, plan[i].d_u2,
						plan[i].d_u_cp, N_cp, plan[i].d_t_cp			
					);
			}

			if(Sto_Rec==1&&vp_type==2||Save_Not==1)
			{
				cudaMemcpyAsync(plan[i].u2,plan[i].d_u2,sizeof(cufftComplex)*ntp,cudaMemcpyDeviceToHost,plans[i].stream);

				sprintf(filename,"./output/GPU_%d_u2_%d.dat",i,it);     
				fp=fopen(filename,"wb");
				for(ix=0;ix<ntx-0;ix++)
				{
					for(iz=0;iz<ntz-0;iz++)
					{
						u2_real[iz*ntx+ix]=plan[i].u2[iz*ntx+ix].x;			
						fwrite(&u2_real[iz*ntx+ix],sizeof(float),1,fp);
					}
				}
				fclose(fp);			
			}

			if(it==800)
			{
				cudaMemcpyAsync(kstabilization,plan[i].d_kstabilization,sizeof(float)*ntp,cudaMemcpyDeviceToHost,plans[i].stream);
				cudaMemcpyAsync(kfilter,plan[i].d_kfilter,sizeof(float)*ntp,cudaMemcpyDeviceToHost,plans[i].stream);

				sprintf(filename,"./output/kstabilization.dat");     
				fp=fopen(filename,"wb");
				for(ix=0;ix<ntx-0;ix++)
				{
					for(iz=0;iz<ntz-0;iz++)
					{	
						fwrite(&kstabilization[iz*ntx+ix],sizeof(float),1,fp);
					}
				}
				fclose(fp);					

				sprintf(filename,"./output/kfilter.dat");     
				fp=fopen(filename,"wb");
				for(ix=0;ix<ntx-0;ix++)
				{
					for(iz=0;iz<ntz-0;iz++)
					{	
						fwrite(&kfilter[iz*ntx+ix],sizeof(float),1,fp);
					}
				}
				fclose(fp);	
			}


			cuda_kernel_update<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(ntx, ntz, plan[i].d_u0, plan[i].d_u1, plan[i].d_u2);


			if(myid==0&&it%100==0)
			{
				printf("shot %d forward %d has finished!\n", is+i+1, it);
			}

		}// GPU_N end	
	}// nt end



	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		// output seismograms ...

		if(vp_type==0)	// homogeneous model
		{
			cudaMemcpyAsync(plan[i].seismogram_dir,plan[i].d_seismogram,
					sizeof(float)*ss[is+i].r_n*nt,cudaMemcpyDeviceToHost,plans[i].stream);
		}
		else if(vp_type==1)	// ture model
		{
			cudaMemcpyAsync(plan[i].seismogram_obs,plan[i].d_seismogram,
					sizeof(float)*ss[is+i].r_n*nt,cudaMemcpyDeviceToHost,plans[i].stream);
		}
		else if(vp_type==2)	// initial model
		{
			cudaMemcpyAsync(plan[i].seismogram_syn,plan[i].d_seismogram,
					sizeof(float)*ss[is+i].r_n*nt,cudaMemcpyDeviceToHost,plans[i].stream);
		}

	}

	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}//end GPU

	//free the memory of DEVICE
	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);	
		cudaStreamDestroy(plans[i].stream);
	}

	free(u2_real);
}




extern "C"
void cuda_visco_PSM_2d_backward
(
	int beta1, int beta2,
	int nt, int ntx, int ntz, int ntp, int nx, int nz, int L, float dx, float dz, float dt,
	float *vp, float *Gamma, float avervp, float averGamma, float f0, float Omega0, float *ricker,
	int myid, int is, struct Source ss[], struct MultiGPU plan[], int GPU_N, int rnmax, int nrx_obs, int N_cp, int *t_cp,
	float kx_cut, float kz_cut, float sigma, int Order,	float taper_ratio, float *kfilter, float *kstabilization,
	int Sto_Rec, int Save_Not
)
{
	int i, it, ix, iz, ip;

	size_t size_model=sizeof(float)*ntp;

	FILE *fp;
	char filename[40];

	float *u2_real;
	u2_real = (float*)malloc(sizeof(float)*ntp);

	Multistream plans[GPU_N];

	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);
		cudaStreamCreate(&plans[i].stream);	
		cufftSetStream(plan[i].PLAN_FORWARD,plans[i].stream);
		cufftSetStream(plan[i].PLAN_BACKWARD,plans[i].stream);
	}	

	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 dimGrid((ntx+dimBlock.x-1)/dimBlock.x,(ntz+dimBlock.y-1)/dimBlock.y);


	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		// initialization for reconstruction

		cuda_kernel_initialization<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(ntx, ntz, plan[i].d_u0_inv, plan[i].d_u1_inv, plan[i].d_u2_inv, plan[i].d_uk0_inv, plan[i].d_uk_inv,
			plan[i].d_Lap, plan[i].d_amp_Lap, plan[i].d_pha_Lap);


		cuda_kernel_initialization_Finals<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(ntx, ntz, plan[i].d_u0_inv, plan[i].d_u1_inv, plan[i].d_u2_final0, plan[i].d_u2_final1);


		// initialization for backward propagation

		cuda_kernel_initialization<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(ntx, ntz, plan[i].d_u0, plan[i].d_u1, plan[i].d_u2, plan[i].d_uk0, plan[i].d_uk,
			plan[i].d_Lap, plan[i].d_amp_Lap, plan[i].d_pha_Lap);


		// initialization for imaging

		cuda_kernel_initialization_images<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(ntx, ntz, plan[i].d_image_cor, plan[i].d_image_nor, plan[i].d_image_sources, plan[i].d_image_receivers);			
	}


	// copy the vectors from the host to the device
	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		cudaMemcpyAsync(plan[i].d_seismogram_rms,plan[i].seismogram_rms,
			sizeof(float)*rnmax*nt,cudaMemcpyHostToDevice,plans[i].stream);
	}


	int beta1_inv=beta1;
	int beta2_inv=-1*beta2;

	float kx_cut_inv=kx_cut;
	float kz_cut_inv=kz_cut;

	for(it=nt-3;it>=0;it--)  
	{
		for(i=0;i<GPU_N;i++)
		{
			cudaSetDevice(i);


			if(Sto_Rec==0)
			{

				cufftExecC2C(plan[i].PLAN_FORWARD,plan[i].d_u1_inv,plan[i].d_uk_inv,CUFFT_FORWARD); //CUFFT_FORWARD

				cuda_kernel_visco_PSM_2d_reconstruction_k_space<<<dimGrid,dimBlock,0,plans[i].stream>>>
					(
						beta1_inv, beta2_inv,
						it, nt, ntx, ntz, nx, nz, L, dx, dz, dt, 
						plan[i].d_vp, plan[i].d_Gamma, averGamma, f0, Omega0,
						plan[i].d_kx, plan[i].d_kz,  
						plan[i].d_uk_inv, plan[i].d_uk0_inv, 
						plan[i].d_Lap_uk, plan[i].d_amp_uk, plan[i].d_pha_uk
					);	

				cufftExecC2C(plan[i].PLAN_BACKWARD,plan[i].d_Lap_uk,plan[i].d_Lap,CUFFT_INVERSE); //CUFFT_INVERSE


				if(beta1_inv!=0)
				{	
					cufftExecC2C(plan[i].PLAN_BACKWARD,plan[i].d_pha_uk,plan[i].d_pha_Lap,CUFFT_INVERSE); //CUFFT_INVERSE					
				}

				
				if(beta2_inv!=0)
				{
					if (beta2_inv<0)
					{
						cuda_kernel_filter2d<<<dimGrid,dimBlock,0,plans[i].stream>>>
							(ntx, ntz, dx, dz, plan[i].d_kfilter, plan[i].d_kx, plan[i].d_kz, plan[i].d_amp_uk, kx_cut_inv, kz_cut_inv, taper_ratio);
					}

					cufftExecC2C(plan[i].PLAN_BACKWARD,plan[i].d_amp_uk,plan[i].d_amp_Lap,CUFFT_INVERSE); //CUFFT_INVERSE
				}
				

				cuda_kernel_visco_PSM_2d_reconstruction_x_space<<<dimGrid,dimBlock,0,plans[i].stream>>>
					(
						beta1_inv, beta2_inv,
						it, nt, ntx, ntz, nx, nz, L, dx, dz, dt, 
						plan[i].d_vp, plan[i].d_Gamma, averGamma, f0, Omega0,
						plan[i].d_ricker, ss[is+i].s_ix, ss[is+i].s_iz,
						plan[i].d_u0_inv, plan[i].d_u1_inv, plan[i].d_u2_inv,
						plan[i].d_Lap, plan[i].d_amp_Lap, plan[i].d_pha_Lap,			
						plan[i].d_borders_up, plan[i].d_borders_bottom, plan[i].d_borders_left, plan[i].d_borders_right
					);	

					
				if(beta2_inv<0)
				{
					cufftExecC2C(plan[i].PLAN_FORWARD,plan[i].d_u2_inv,plan[i].d_uk_inv,CUFFT_FORWARD); //CUFFT_FORWARD

					cuda_kernel_AdaSta<<<dimGrid,dimBlock,0,plans[i].stream>>>
					(
						nt-it, ntx, ntz, dx, dz, dt, plan[i].d_kstabilization,
						plan[i].d_vp, plan[i].d_Gamma, avervp, averGamma, Omega0,
						plan[i].d_kx, plan[i].d_kz, plan[i].d_uk_inv, sigma, Order
					);	

					cufftExecC2C(plan[i].PLAN_BACKWARD,plan[i].d_uk_inv,plan[i].d_u2_inv,CUFFT_INVERSE); //CUFFT_INVERSE	
				}
				


				cuda_kernel_MTF_2nd<<<dimGrid,dimBlock,0,plans[i].stream>>>
					(L, ntx, ntz, dx, dz, dt, plan[i].d_vp, plan[i].d_u0_inv, plan[i].d_u1_inv, plan[i].d_u2_inv);

				cuda_kernel_checkpoints_In<<<dimGrid,dimBlock,0,plans[i].stream>>>
					(
						it, nt, ntx, ntz, nx, nz, L, dx, dz, dt, 
						plan[i].d_u1_inv, plan[i].d_u2_inv,
						plan[i].d_u_cp, N_cp, plan[i].d_t_cp			
					);							

				cuda_kernel_update<<<dimGrid,dimBlock,0,plans[i].stream>>>
					(ntx, ntz, plan[i].d_u0_inv, plan[i].d_u1_inv, plan[i].d_u2_inv);

			}


			if(Sto_Rec==1)
			{
				sprintf(filename,"./output/GPU_%d_u2_%d.dat",i,it); 
				fp=fopen(filename,"rb");
				for(ix=0;ix<ntx-0;ix++)
				{
					for(iz=0;iz<ntz-0;iz++)
					{								
						fread(&u2_real[iz*ntx+ix],sizeof(float),1,fp);
						plan[i].u2[iz*ntx+ix].x=u2_real[iz*ntx+ix];	
						plan[i].u2[iz*ntx+ix].y=0.0;
					}
				}
				fclose(fp);

				cudaMemcpyAsync(plan[i].d_u2_inv,plan[i].u2,sizeof(cufftComplex)*ntp,cudaMemcpyHostToDevice,plans[i].stream);

			}


			cufftExecC2C(plan[i].PLAN_FORWARD,plan[i].d_u1,plan[i].d_uk,CUFFT_FORWARD); //CUFFT_FORWARD

			cuda_kernel_visco_PSM_2d_backward_k_space<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
					beta1, beta2,
					it, nt, ntx, ntz, dx, dz, dt, 
					plan[i].d_vp, plan[i].d_Gamma, averGamma, f0, Omega0,
					plan[i].d_kx, plan[i].d_kz,  
					plan[i].d_uk, plan[i].d_uk0, 
					plan[i].d_Lap_uk, plan[i].d_amp_uk, plan[i].d_pha_uk
				);

			cufftExecC2C(plan[i].PLAN_BACKWARD,plan[i].d_Lap_uk,plan[i].d_Lap,CUFFT_INVERSE); //CUFFT_INVERSE


			if(beta1!=0)
			{	
				cufftExecC2C(plan[i].PLAN_BACKWARD,plan[i].d_pha_uk,plan[i].d_pha_Lap,CUFFT_INVERSE); //CUFFT_INVERSE					
			}

			if(beta2!=0)
			{
				if (beta2<0)
				{
					cuda_kernel_filter2d<<<dimGrid,dimBlock,0,plans[i].stream>>>
						(ntx, ntz, dx, dz, plan[i].d_kfilter, plan[i].d_kx, plan[i].d_kz, plan[i].d_amp_uk, kx_cut, kz_cut, taper_ratio);
				}

				cufftExecC2C(plan[i].PLAN_BACKWARD,plan[i].d_amp_uk,plan[i].d_amp_Lap,CUFFT_INVERSE); //CUFFT_INVERSE
			}

			cuda_kernel_visco_PSM_2d_backward_x_space<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
					beta1, beta2,
					it, nt, ntx, ntz, dx, dz, dt, 
					plan[i].d_vp, plan[i].d_Gamma, averGamma, f0, Omega0,
					plan[i].d_seismogram_rms, plan[i].d_r_ix, ss[is+i].r_iz, ss[is+i].s_ix, rnmax, nrx_obs,
					plan[i].d_u0, plan[i].d_u1, plan[i].d_u2,
					plan[i].d_Lap, plan[i].d_amp_Lap, plan[i].d_pha_Lap
				);

				
			if(beta2<0)
			{
				cufftExecC2C(plan[i].PLAN_FORWARD,plan[i].d_u2,plan[i].d_uk,CUFFT_FORWARD); //CUFFT_FORWARD

				cuda_kernel_AdaSta<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
					nt-it, ntx, ntz, dx, dz, dt, plan[i].d_kstabilization,
					plan[i].d_vp, plan[i].d_Gamma, avervp, averGamma, Omega0,
					plan[i].d_kx, plan[i].d_kz, plan[i].d_uk, sigma, Order
				);	

				cufftExecC2C(plan[i].PLAN_BACKWARD,plan[i].d_uk,plan[i].d_u2,CUFFT_INVERSE); //CUFFT_INVERSE	
			}
		

			cuda_kernel_MTF_2nd<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(L, ntx, ntz, dx, dz, dt, plan[i].d_vp, plan[i].d_u0, plan[i].d_u1, plan[i].d_u2);

			
			cuda_kernel_image<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(
					ntx, ntz, L,
					plan[i].d_u2_inv, plan[i].d_u2,
					plan[i].d_image_cor, plan[i].d_image_sources, plan[i].d_image_receivers
				);


			if(Save_Not==1)
			{
				cudaMemcpyAsync(plan[i].u1,plan[i].d_u2_inv,sizeof(cufftComplex)*ntp,cudaMemcpyDeviceToHost,plans[i].stream);
				cudaMemcpyAsync(plan[i].u2,plan[i].d_u2,sizeof(cufftComplex)*ntp,cudaMemcpyDeviceToHost,plans[i].stream);
				cudaStreamSynchronize(plans[i].stream);

				sprintf(filename,"./output/GPU_%d_u2_inv_%d.dat",i,it); 
				fp=fopen(filename,"wb");
				for(ix=0;ix<ntx-0;ix++)
				{
					for(iz=0;iz<ntz-0;iz++)
					{
						u2_real[iz*ntx+ix]=plan[i].u1[iz*ntx+ix].x;
						fwrite(&u2_real[iz*ntx+ix],sizeof(float),1,fp);
					}
				}
				fclose(fp);

				sprintf(filename,"./output/GPU_%d_u2_bak_%d.dat",i,it); 
				fp=fopen(filename,"wb");
				for(ix=0;ix<ntx-0;ix++)
				{
					for(iz=0;iz<ntz-0;iz++)
					{	
						u2_real[iz*ntx+ix]=plan[i].u2[iz*ntx+ix].x;				
						fwrite(&u2_real[iz*ntx+ix],sizeof(float),1,fp);
					}
				}
				fclose(fp);
			}

			cuda_kernel_update<<<dimGrid,dimBlock,0,plans[i].stream>>>
				(ntx, ntz, plan[i].d_u0, plan[i].d_u1, plan[i].d_u2);



			if(myid==0&&it%100==0)
			{
				printf("shot %d reconstruction and backward %d has finished!\n", is+i+1, it);
			}

		}// GPU_N end	
	}// nt end



	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		// output images ...

		cudaMemcpyAsync(plan[i].image_cor,plan[i].d_image_cor,sizeof(float)*ntp,cudaMemcpyDeviceToHost,plans[i].stream);
		cudaMemcpyAsync(plan[i].image_sources,plan[i].d_image_sources,sizeof(float)*ntp,cudaMemcpyDeviceToHost,plans[i].stream);
		cudaMemcpyAsync(plan[i].image_receivers,plan[i].d_image_receivers,sizeof(float)*ntp,cudaMemcpyDeviceToHost,plans[i].stream);
	}


	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}//end GPU

	//free the memory of DEVICE
	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);
		cudaStreamDestroy(plans[i].stream);
	}

	free(u2_real);
	
}


/*==========================================================
  This two subroutines are used for Laplace filteing
  ===========================================================*/

extern "C"
void Laplace_filtering
(
	float *image, int ntx, int ntz, float dx, float dz
)
{ 

	int ix,iz,ip,K,NX,NZ;


	K=(int)ceil(log(1.0*ntx)/log(2.0));
	NX=(int)pow(2.0,K);

	K=(int)ceil(log(1.0*ntz)/log(2.0));
	NZ=(int)pow(2.0,K);

	float dkx,dkz;
	float kx,kz;

	dkx=(float)1.0/((NX)*dx);
	dkz=(float)1.0/((NZ)*dz);

	int NTP=NX*NZ;

	cufftComplex *pp,*temp,*tempout;		

	cudaMallocHost((void **)&pp, sizeof(cufftComplex)*NX*NZ);
	cudaMalloc((void **)&temp,sizeof(cufftComplex)*NX*NZ);
	cudaMalloc((void **)&tempout,sizeof(cufftComplex)*NX*NZ);

	cufftHandle plan;
	cufftPlan2d(&plan,NX,NZ,CUFFT_C2C);

	for(ip=0;ip<NTP;ip++)
	{ 
		pp[ip].x=0.0;
		pp[ip].y=0.0; 
	} 

	for(ix=0;ix<ntx;ix++)
	{            
		for(iz=0;iz<ntz;iz++)
		{
			pp[ix*NZ+iz].x=image[iz*ntx+ix];
		}
	} 

	cudaMemcpy(temp,pp,sizeof(cufftComplex)*NX*NZ,cudaMemcpyHostToDevice);
	cufftExecC2C(plan,temp,tempout,CUFFT_FORWARD);
	cudaMemcpy(pp,tempout,sizeof(cufftComplex)*NX*NZ,cudaMemcpyDeviceToHost);

	for(ix=0;ix<NX;ix++)
	{            
		for(iz=0;iz<NZ;iz++)
		{
			if(ix<NX/2)
			{
				kx=2*PI*ix*dkx;
			}
			if(ix>NX/2)	
			{
				kx=2*PI*(NX-1-ix)*dkx;
			}

			if(iz<NZ/2)
			{
				kz=2*PI*iz*dkz;//2*PI*(NZ/2-1-iz)*dkz;//0.0;//
			}
			if(iz>NZ/2)
			{
				kz=2*PI*(NZ-1-iz)*dkz;//2*PI*(iz-NZ/2)*dkz;//0.0;//
			}

			ip=ix*NZ+iz;

			pp[ip].x=pp[ip].x*(kx*kx+kz*kz);
			pp[ip].y=pp[ip].y*(kx*kx+kz*kz);

		}
	} 

	cudaMemcpy(temp,pp,sizeof(cufftComplex)*NX*NZ,cudaMemcpyHostToDevice);
	cufftExecC2C(plan,temp,tempout,CUFFT_INVERSE);
	cudaMemcpy(pp,tempout,sizeof(cufftComplex)*NX*NZ,cudaMemcpyDeviceToHost);

	for(ix=0;ix<ntx;ix++)
	{            
		for(iz=0;iz<ntz;iz++)
		{
			image[iz*ntx+ix]=pp[ix*NZ+iz].x/(NX*NZ);
		}
	} 

	cudaFreeHost(pp);
	cudaFree(temp);
	cudaFree(tempout);
	cufftDestroy(plan);

	return;
}


/*=========================================================
  Allocate and Free the memory for variables in device
  ========================================================*/

extern "C"
void cuda_Device_malloc
(
	int ntx, int ntz, int ntp, int nx, int nz, int nt, 
	float dx, float dz, int L, int rnmax, int N_cp,
	struct MultiGPU plan[], int GPU_N
)
{
	int i;
	size_t size_model=sizeof(float)*ntp;

	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		cufftPlan2d(&plan[i].PLAN_FORWARD,ntz,ntx,CUFFT_C2C);
		cufftPlan2d(&plan[i].PLAN_BACKWARD,ntz,ntx,CUFFT_C2C);

		cudaMallocHost((void **)&plan[i].u0, sizeof(cufftComplex)*ntp);		
		cudaMallocHost((void **)&plan[i].u1, sizeof(cufftComplex)*ntp);
		cudaMallocHost((void **)&plan[i].u2, sizeof(cufftComplex)*ntp);		

		cudaMallocHost((void **)&plan[i].seismogram_obs, sizeof(float)*nt*rnmax);
		cudaMallocHost((void **)&plan[i].seismogram_dir, sizeof(float)*nt*rnmax);
		cudaMallocHost((void **)&plan[i].seismogram_syn, sizeof(float)*nt*rnmax);
		cudaMallocHost((void **)&plan[i].seismogram_rms, sizeof(float)*nt*rnmax);
		
		cudaMallocHost((void **)&plan[i].image_sources, sizeof(float)*ntp);
		cudaMallocHost((void **)&plan[i].image_receivers, sizeof(float)*ntp);
		cudaMallocHost((void **)&plan[i].image_cor, sizeof(float)*ntp);
		cudaMallocHost((void **)&plan[i].image_nor, sizeof(float)*ntp);


		cudaMalloc((void**)&plan[i].d_r_ix,sizeof(int)*rnmax);

		cudaMalloc((void**)&plan[i].d_ricker,sizeof(float)*nt);        //ricker

		cudaMalloc((void**)&plan[i].d_vp,size_model);
		cudaMalloc((void**)&plan[i].d_Gamma,size_model);

		cudaMalloc((void**)&plan[i].d_u0,sizeof(cufftComplex)*ntp);
		cudaMalloc((void**)&plan[i].d_u1,sizeof(cufftComplex)*ntp);
		cudaMalloc((void**)&plan[i].d_u2,sizeof(cufftComplex)*ntp);

		cudaMalloc((void**)&plan[i].d_u0_inv,sizeof(cufftComplex)*ntp);
		cudaMalloc((void**)&plan[i].d_u1_inv,sizeof(cufftComplex)*ntp);
		cudaMalloc((void**)&plan[i].d_u2_inv,sizeof(cufftComplex)*ntp);

		cudaMalloc((void**)&plan[i].d_t_cp,sizeof(int)*N_cp);
		cudaMalloc((void**)&plan[i].d_u_cp,size_model*N_cp);		//checkpoints


		cudaMalloc((void**)&plan[i].d_kx,sizeof(float)*ntx);
		cudaMalloc((void**)&plan[i].d_kz,sizeof(float)*ntz);
		cudaMalloc((void**)&plan[i].d_kfilter,sizeof(float)*ntp);
		cudaMalloc((void**)&plan[i].d_kstabilization,sizeof(float)*ntp);

		cudaMalloc((void **)&plan[i].d_uk,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_uk0,sizeof(cufftComplex)*ntp);

		cudaMalloc((void **)&plan[i].d_uk_inv,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_uk0_inv,sizeof(cufftComplex)*ntp);

		cudaMalloc((void **)&plan[i].d_Lap_uk,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_amp_uk,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_pha_uk,sizeof(cufftComplex)*ntp);

		cudaMalloc((void **)&plan[i].d_Lap,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_amp_Lap,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_pha_Lap,sizeof(cufftComplex)*ntp);
							

		cudaMalloc((void**)&plan[i].d_seismogram,sizeof(float)*nt*rnmax);
		cudaMalloc((void**)&plan[i].d_seismogram_rms,sizeof(float)*nt*rnmax);

		cudaMalloc((void**)&plan[i].d_borders_up,sizeof(float)*nt*nx);
		cudaMalloc((void**)&plan[i].d_borders_bottom,sizeof(float)*nt*nx);
		cudaMalloc((void**)&plan[i].d_borders_left,sizeof(float)*nt*nz);
		cudaMalloc((void**)&plan[i].d_borders_right,sizeof(float)*nt*nz);

		cudaMalloc((void**)&plan[i].d_u2_final0,size_model);
		cudaMalloc((void**)&plan[i].d_u2_final1,size_model);

		cudaMalloc((void**)&plan[i].d_image_sources,size_model);
		cudaMalloc((void**)&plan[i].d_image_receivers,size_model);
		cudaMalloc((void**)&plan[i].d_image_cor,size_model);
		cudaMalloc((void**)&plan[i].d_image_nor,size_model);
	}
}

extern "C"
void cuda_Device_free
(
	int ntx, int ntz, int ntp, int nx, int nz, int nt, 
	float dx, float dz, int L, int rnmax, int N_cp,
	struct MultiGPU plan[], int GPU_N
)
{
	int i;
	 

	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		cufftDestroy(plan[i].PLAN_FORWARD);
		cufftDestroy(plan[i].PLAN_BACKWARD);

		cudaFreeHost(plan[i].u0);
		cudaFreeHost(plan[i].u1);
		cudaFreeHost(plan[i].u2); 

		cudaFreeHost(plan[i].seismogram_obs);
		cudaFreeHost(plan[i].seismogram_dir);
		cudaFreeHost(plan[i].seismogram_syn); 
		cudaFreeHost(plan[i].seismogram_rms);
 
		cudaFreeHost(plan[i].image_cor);
		cudaFreeHost(plan[i].image_nor);
		cudaFreeHost(plan[i].image_sources);
		cudaFreeHost(plan[i].image_receivers);


		cudaFree(plan[i].d_r_ix);

		cudaFree(plan[i].d_ricker);

		cudaFree(plan[i].d_vp);
		cudaFree(plan[i].d_Gamma);

		cudaFree(plan[i].d_u0);
		cudaFree(plan[i].d_u1);
		cudaFree(plan[i].d_u2);

		cudaFree(plan[i].d_u0_inv);
		cudaFree(plan[i].d_u1_inv);
		cudaFree(plan[i].d_u2_inv);

		cudaFree(plan[i].d_t_cp);
		cudaFree(plan[i].d_u_cp);


		cudaFree(plan[i].d_kx);
		cudaFree(plan[i].d_kz);
		cudaFree(plan[i].d_kfilter);
		cudaFree(plan[i].d_kstabilization);


		cudaFree(plan[i].d_uk);
		cudaFree(plan[i].d_uk0);

		cudaFree(plan[i].d_uk_inv);
		cudaFree(plan[i].d_uk0_inv);

		cudaFree(plan[i].d_Lap_uk);
		cudaFree(plan[i].d_amp_uk);
		cudaFree(plan[i].d_pha_uk);

		cudaFree(plan[i].d_Lap);
		cudaFree(plan[i].d_amp_Lap);
		cudaFree(plan[i].d_pha_Lap);


		cudaFree(plan[i].d_seismogram);
		cudaFree(plan[i].d_seismogram_rms);

		cudaFree(plan[i].d_borders_up);
		cudaFree(plan[i].d_borders_bottom);
		cudaFree(plan[i].d_borders_left);
		cudaFree(plan[i].d_borders_right);

		cudaFree(plan[i].d_u2_final0);
		cudaFree(plan[i].d_u2_final1);

		cudaFree(plan[i].d_image_sources);
		cudaFree(plan[i].d_image_receivers);
		cudaFree(plan[i].d_image_cor);
		cudaFree(plan[i].d_image_nor);
	}
}


extern "C"
void cuda_Host_initialization
(
	int ntx, int ntz, int ntp, int nx, int nz, int nt, 
	float dx, float dz, int L, int rnmax, int N_cp,
	struct MultiGPU plan[], int GPU_N
)
{
	int i;
	 

	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		memset(plan[i].u0, 0, ntx*ntz*sizeof(float));
		memset(plan[i].u1, 0, ntx*ntz*sizeof(float));
		memset(plan[i].u2, 0, ntx*ntz*sizeof(float));

		memset(plan[i].seismogram_obs, 0, nt*rnmax*sizeof(float));
		memset(plan[i].seismogram_dir, 0, nt*rnmax*sizeof(float));
		memset(plan[i].seismogram_syn, 0, nt*rnmax*sizeof(float));
		memset(plan[i].seismogram_rms, 0, nt*rnmax*sizeof(float));

 		memset(plan[i].image_cor, 0, ntx*ntz*sizeof(float));
 		memset(plan[i].image_nor, 0, ntx*ntz*sizeof(float));
 		memset(plan[i].image_sources, 0, ntx*ntz*sizeof(float));
 		memset(plan[i].image_receivers, 0, ntx*ntz*sizeof(float));
	}
}

extern "C"
void getdevice(int *GPU_N)
{	
	cudaGetDeviceCount(GPU_N);	
}
