#include <iostream>
#include <string>
#include "dnn.hpp"
#include <cuda_runtime.h>

#include <helper_cuda.h>

using namespace std;

//Define the parameters if not defined externally
#ifndef Sy
  #define Sy 1
  #define Sx 1
#endif

#ifndef Tnn
  //Tiling Sizes
  #define Tnn 32
  #define Tn  16
  #define Ti  16
  
  #define Ty  8
  #define Tx  8
#endif

#define NYPAD (Ny+Ky)
#define NXPAD (Nx+Kx)

#define NYSCL (Ny/Sy)
#define NXSCL (Nx/Sx)

#define SYNAPSE_SIZE (1L*Ky*Kx*Nn*Ni)

#define  threadsPerBlock  256
#define  blocksPerGrid    (Ni + threadsPerBlock - 1) / threadsPerBlock
#define BATCH 4

VTYPE (*synapse)[Ky][Kx][Nn][Ni];
VTYPE  (*neuron_i)[NYPAD][NXPAD][Ni];
VTYPE  (*neuron_n)[NYSCL][NXSCL][Nn];
VTYPE  (*neuron_n_from_dev)[NYSCL][NXSCL][Nn];
VTYPE  (*neuron_n_from_dev1D)[NYSCL][NXSCL][Nn];
VTYPE  (*neuron_n_from_dev2D)[NYSCL][NXSCL][Nn];
VTYPE (*neuron_n2)[NYSCL][NXSCL][Nn];

VTYPE (*dev_synapse)[Ky][Kx][Nn][Ni];
VTYPE  (*dev_neuron_i)[NYPAD][NXPAD][Ni];
VTYPE  (*dev_neuron_n)[NYSCL][NXSCL][Nn];
VTYPE  (*dev_neuron_n1D)[NYSCL][NXSCL][Nn];
VTYPE  (*dev_neuron_n2D)[NYSCL][NXSCL][Nn];

void fill_convolution_shared_simple(VTYPE (&synapse)[Ky][Kx][Nn][Ni], 
                                    VTYPE (&neuron_i)[NYPAD][NXPAD][Ni]) {
  for(int yy = 0; yy < Ky; ++yy) {
    for(int xx = 0; xx < Kx; ++xx) {
      for(int nn = 0; nn < Nn; ++nn) {
        for(int ni = 0; ni < Ni; ++ni) {
          synapse[yy][xx][nn][ni] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
        } } } }
  for(int yy = 0; yy < NYPAD; ++yy) {
    for(int xx = 0; xx < NXPAD; ++xx) {      
      for(int ni = 0; ni < Ni; ++ni) {
        neuron_i[yy][xx][ni] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
  }  }  }
}

//std::pair<int,int> convolution_layer_blocked(
void convolution_layer_blocked(
                              VTYPE (&synapse)[Ky][Kx][Nn][Ni], 
                              VTYPE (&neuron_i)[NYPAD][NXPAD][Ni], 
                              VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]) {
  VTYPE sum[Nn]={0};

  for (int yy = 0; yy < Ny; yy += Ty) {
    for (int xx = 0; xx < Nx; xx += Tx) {
      for (int nnn = 0; nnn < Nn; nnn += Tnn) {
        int yout = yy/Sy;
        for (int y = yy; y < yy + Ty; y += Sy) { // tiling for y;
          int xout = xx/Sx;

          for (int x = xx; x < xx + Tx; x += Sx) { // tiling for x;

            for (int nn = nnn; nn < nnn + Tnn; nn += Tn) {
              for (int n = nn; n < nn + Tn; n++) {
                sum[n] = 0;
              }

              for (int ky = 0; ky < Ky; ky++) {  // sliding window;
                for (int kx = 0; kx < Kx; kx++) {

                  int ii = 0;
                  VTYPE sum_sc;

                  for (; ii < Ni -Ti+1; ii += Ti) {
                    for (int n = nn; n < nn + Tn; n++) {
                      sum_sc=0;
                      for (int i = ii; i < ii + Ti; i++) {
                        VTYPE sv = synapse[ky][kx][n][i];
                        VTYPE nv = neuron_i[ky + y][kx + x][i];
                        sum_sc+=sv*nv;
                      }
                      sum[n]+=sum_sc;
                    }
                  }
                }
              }

              //transfer
              for (int n = nn; n < nn + Tn; n++) {
                neuron_n[yout][xout][n] = transfer(sum[n]);
              }
            }
            xout++; 
          }
          yout++;
        }
      }
    }
  }
}

void  convolution_layer(VTYPE (&synapse)[Ky][Kx][Nn][Ni], 
                               VTYPE (&neuron_i)[NYPAD][NXPAD][Ni], 
                               VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]) {
  VTYPE sum[Nn]={0};

  // — Original code — (excluding nn, ii loops)
  int yout = 0;
  for (int y = 0; y < Ny; y += Sy) { // tiling for y;
    int xout = 0;
    for (int x = 0; x < Nx; x += Sx) { // tiling for x;
      for (int nn = 0; nn < Nn; nn += Tn) {
        for (int n = nn; n < nn + Tn; n++) {
          sum[n]=0;
        }

        // sliding window;
        for (int ky = 0; ky < Ky; ky++)
          for (int kx = 0; kx < Kx; kx++)
            for (int n = nn; n < nn + Tn; n++)
              for (int i = 0; i < Ni; i++) {
                VTYPE sv = synapse[ky][kx][n][i];
                VTYPE nv = neuron_i[ky + y][kx + x][i];
                sum[n]+=sv*nv;
              }
        for (int n = nn; n < nn + Tn; n++) {
          neuron_n[yout][xout][n] = transfer(sum[n]);
        }
      }
      xout++; 
    }
    yout++;
  }
}


__global__ void  cuda_convolution_layer(VTYPE (&synapse)[Ky][Kx][Nn][Ni], 
                               VTYPE (&neuron_i)[NYPAD][NXPAD][Ni], 
                               VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]) {

  // — Original code — (excluding nn, ii loops)
  int yout = 0;
  for (int y = 0; y < Ny; y += Sy) { // tiling for y;
    int xout = 0;
    for (int x = 0; x < Nx; x += Sx) { // tiling for x;
      for (int n = 0; n < Nn; n++) {

        // sliding window;
        for (int ky = 0; ky < Ky; ky++)
          for (int kx = 0; kx < Kx; kx++)
              for (int i = 0; i < Ni; i++) {
                VTYPE sv = synapse[ky][kx][n][i];
                VTYPE nv = neuron_i[ky + y][kx + x][i];
                neuron_n[yout][xout][n]+=sv*nv;
              }
        neuron_n[yout][xout][n] = (neuron_n[yout][xout][n]>0) ? neuron_n[yout][xout][n] : (neuron_n[yout][xout][n]/4.0);
      }
      xout++; 
    }
    yout++;
  }
}


__global__ void  cuda_convolution_layer1D(VTYPE (&synapse)[Ky][Kx][Nn][Ni], 
                               VTYPE (&neuron_i)[NYPAD][NXPAD][Ni], 
                               VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]) {

  __shared__ VTYPE temp[threadsPerBlock];
  int index = blockDim.x*blockIdx.x + threadIdx.x;
  // — Original code — (excluding nn, ii loops)
  int yout = 0;
  if(index < Ni)
  {
    for (int y = 0; y < Ny; y += Sy) { // tiling for y;
      int xout = 0;
      for (int x = 0; x < Nx; x += Sx) { // tiling for x;
        for (int n = 0; n < Nn; n++) {

          // sliding window;
          for (int ky = 0; ky < Ky; ky++)
            for (int kx = 0; kx < Kx; kx++){
                  VTYPE sv = synapse[ky][kx][n][index];
                  VTYPE nv = neuron_i[ky + y][kx + x][index];
                  //neuron_n[yout][xout][n]+=sv*nv;
                  temp[threadIdx.x] = sv*nv;
                  __syncthreads();
                  if(0 == threadIdx.x)
                  {
                    VTYPE sum = 0.0;
                    for( int i = 0; i < threadsPerBlock; i++ )
                      sum += temp[i];
                    atomicAdd(&(neuron_n[yout][xout][n]),sum);
                  }
                  __syncthreads();
                }
        }
        xout++; 
      }
      yout++;
    }
  }
}

int main(const int argc, const char** argv) {
  cout << "allocating memory\n";

  synapse   = (VTYPE (*)[Ky][Kx][Nn][Ni])  aligned_malloc(64,  SYNAPSE_SIZE*sizeof(VTYPE));
  neuron_i  = (VTYPE (*)[NYPAD][NXPAD][Ni])aligned_malloc(64,NYPAD*NXPAD*Ni*sizeof(VTYPE));
  neuron_n  = (VTYPE (*)[NYSCL][NXSCL][Nn])aligned_malloc(64,NYSCL*NXSCL*Nn*sizeof(VTYPE));
  neuron_n2 = (VTYPE (*)[NYSCL][NXSCL][Nn])aligned_malloc(64,NYSCL*NXSCL*Nn*sizeof(VTYPE));

  neuron_n_from_dev    = (VTYPE (*)[NYSCL][NXSCL][Nn]) malloc(NYSCL*NXSCL*Nn*sizeof(VTYPE));
  neuron_n_from_dev1D  = (VTYPE (*)[NYSCL][NXSCL][Nn]) malloc(NYSCL*NXSCL*Nn*sizeof(VTYPE));
  neuron_n_from_dev2D  = (VTYPE (*)[NYSCL][NXSCL][Nn]) malloc(NYSCL*NXSCL*Nn*sizeof(VTYPE));

  cout << "initializing arrays\n";

  fill_convolution_shared_simple(*synapse,*neuron_i);


  cudaError_t err = cudaSuccess;
  err = cudaMalloc(&dev_synapse,  Ky*Kx*Nn*Ni*sizeof(VTYPE));
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device synapse (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  err = cudaMalloc(&dev_neuron_i, NYPAD*NXPAD*Ni*sizeof(VTYPE));
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device neuron_i (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  err = cudaMalloc(&dev_neuron_n, NYSCL*NXSCL*Nn*sizeof(VTYPE));
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device dev_neuron_n (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(dev_synapse, synapse, Kx*Ky*Ni*Nn*sizeof(VTYPE), cudaMemcpyHostToDevice);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy synapse from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(dev_neuron_i, neuron_i, NYPAD*NXPAD*Ni*sizeof(VTYPE), cudaMemcpyHostToDevice);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy neuron_i from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  cout << "starting computation\n";

  //Simple Version
  begin_roi();
  convolution_layer(*synapse,*neuron_i,*neuron_n);
  end_roi();

  cout << "simple version complete!\n";  

  //cout << "starting cuda simple computation\n";

  ////CUDA Simple Version
  //begin_roi();
  //cuda_convolution_layer<<<1,1>>>(*dev_synapse,*dev_neuron_i,*dev_neuron_n);
  //err = cudaMemcpy(neuron_n_from_dev, dev_neuron_n, NYSCL*NXSCL*Nn*sizeof(VTYPE), cudaMemcpyDeviceToHost);

  //if (err != cudaSuccess)
  //{
  //    fprintf(stderr, "Failed to copy neuron_n_from_dev from device to host (error code %s)!\n", cudaGetErrorString(err));
  //    exit(EXIT_FAILURE);
  //}
  ////for (int y = 0; y < NYSCL; y++) 
  ////  for (int x = 0; x < NXSCL; x++)
  ////    for (int n = 0; n < Nn; n++)
  ////      *(neuron_n_from_dev)[y][x][n] = transfer(*(neuron_n_from_dev)[y][x][n]);
  //end_roi();

  //cout << "cuda simple version complete!\n";  

  err = cudaMalloc(&dev_neuron_n1D, NYSCL*NXSCL*Nn*sizeof(VTYPE));
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device dev_neuron_n1D (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  //CUDA 1D Version
  cout << "starting cuda 1D computation\n";
  begin_roi();
  cuda_convolution_layer1D<<<blocksPerGrid,threadsPerBlock>>>(*dev_synapse,*dev_neuron_i,*dev_neuron_n1D);
  err = cudaMemcpy(neuron_n_from_dev1D, dev_neuron_n1D, NYSCL*NXSCL*Nn*sizeof(VTYPE), cudaMemcpyDeviceToHost);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy neuron_n_from_dev1D from device to host (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
  transfer_array((VTYPE*)*neuron_n_from_dev1D,NYSCL*NXSCL*Nn);
  end_roi();

  cout << "cuda 1D version complete!\n";  

  //Blocked Version
  begin_roi();
  convolution_layer_blocked(*synapse,*neuron_i,*neuron_n2);
  end_roi();


  cout << "blocked computation complete!\n";  

  compare((VTYPE*)*neuron_n,(VTYPE*)*neuron_n2,NYSCL*NXSCL*Nn);
  compare((VTYPE*)*neuron_n,(VTYPE*)*neuron_n_from_dev1D,NYSCL*NXSCL*Nn);

  cout << "done\n";
}


