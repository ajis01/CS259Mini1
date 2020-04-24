#include <iostream>
#include "dnn.hpp"
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <math.h>

using namespace std;

//Define the parameters if not defined externally
#ifndef Nn
  #define Nn 128  // Number of Output Layers
  #define Ni 224  // Number of Input  Layers
#endif

#ifndef Tii
  // Tiling Sizes
  #define Tnn 32  
  #define Tii 32
  //#define Tn 5
  //#define Ti 25
  #define Tn 16
  #define Ti 16
#endif
//#define  threadsPerBlock  512
//#define  threadsPerBlockPerDim2D  32
//#define  threadsPerBlockPerDim3DBATCH  1
//#define  threadsPerBlockPerDim3D  16
#define  blocksPerGrid    (Ni + threadsPerBlock - 1) / threadsPerBlock
//#define BATCH 1

#define transferThreadsPerBlock Nn
#define  transferBlocksPerGrid    (Nn + transferThreadsPerBlock - 1) / transferThreadsPerBlock 
#define  batchTransferBlocksPerGrid    (BATCH*Nn + transferThreadsPerBlock - 1) / transferThreadsPerBlock 

//Arrays:
VTYPE synapse[Nn][Ni] __attribute__((aligned(64)));
VTYPE neuron_i[Ni] __attribute__((aligned(64)));
VTYPE batch_neuron_i[BATCH][Ni] __attribute__((aligned(64)));
VTYPE neuron_n[Nn] __attribute__((aligned(64)));
VTYPE batch_neuron_n[BATCH][Nn] __attribute__((aligned(64)));
VTYPE batch_neuron_n3D[BATCH][Nn] __attribute__((aligned(64)));
VTYPE batch_neuron_n_gold[BATCH][Nn] __attribute__((aligned(64)));
VTYPE neuron_n2[Nn] __attribute__((aligned(64)));//   neuron_n2_from_dev[Nn] __attribute__((aligned(64)));

void fill_classifier(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni], 
    VTYPE (&neuron_n)[Nn],   VTYPE (&neuron_n2)[Nn], VTYPE (&batch_neuron_i)[BATCH][Ni],
    VTYPE (&batch_neuron_n)[BATCH][Nn],VTYPE (&batch_neuron_n_gold)[BATCH][Nn]) {
  for(int n = 0; n < Nn; ++n) {
    for(int i = 0; i < Ni; ++i) {
      synapse[n][i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
    }
  }
  for(int i = 0; i < Ni; ++i) {
    neuron_i[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
  }
  for(int b = 0; b < BATCH; ++b) {
    for(int i = 0; i < Ni; ++i) {
      batch_neuron_i[b][i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
    }
  }
  for(int n = 0; n < Nn; ++n) {
    neuron_n[n] = 0; //i;
    neuron_n2[n] = 0; //i;
  }
  for(int b = 0; b < BATCH; ++b) {
    for(int i = 0; i < Nn; ++i) {
      batch_neuron_n[b][i] = 0;
    }
  }
  for(int b = 0; b < BATCH; ++b) {
    for(int i = 0; i < Nn; ++i) {
      batch_neuron_n_gold[b][i] = 0;
    }
  }
}

void classifier_layer(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni], VTYPE (&neuron_n)[Nn]) {
  for (int n = 0; n < Nn; n++) {
    VTYPE temp=0;
    for (int i = 0; i < Ni; i++) {
      temp += synapse[n][i] * neuron_i[i];
    }
    neuron_n[n] = transfer(temp);
  }
}

//__global__ void cuda_classifier_layer(VTYPE *dsynapse, VTYPE *dneuron_i, VTYPE *dneuron_n) {
//  for (int n = 0; n < Nn; n++) {
//    VTYPE temp=0;
//    for (int i = 0; i < Ni; i++) {
//      temp += *(dsynapse+ Ni*n + i) * (*(dneuron_i + i));
//    }
//    *(dneuron_n + n) = (temp > 0) ? temp : temp/4;
//  }
//}
__global__ void cuda_classifier_layer_1DBlocks(VTYPE *dsynapse, VTYPE *dneuron_i, VTYPE *dneuron_n) {
    __shared__ VTYPE temp[threadsPerBlock];
    int index = blockDim.x*blockIdx.x + threadIdx.x;
  for (int n = 0; n < Nn; n++) {
    if(index < Ni)
    {
      temp[threadIdx.x] = *(dsynapse + Ni*n + index) * (*(dneuron_i + index));

      __syncthreads();

      if(0 == threadIdx.x)
      {
        VTYPE sum = 0.0;
        for( int i = 0; i < threadsPerBlock; i++ )
          sum += temp[i];
       // *(dneuron_n + n) = (sum > 0) ? sum : sum/4;
          atomicAdd((dneuron_n + n),sum);
      }
  __syncthreads();
  }
  }
}


__global__ void cuda_classifier_layer_2DBlocks(VTYPE *dsynapse, VTYPE *dneuron_i, VTYPE *dneuron_n) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = (blockIdx.y * blockDim.y) + threadIdx.y;
  if(i < Nn && j < Ni) 
  {  
    VTYPE temp = *(dsynapse + i*Ni + j) * (*(dneuron_i + j));
    atomicAdd((dneuron_n + i),temp);
  }
}
__global__ void cuda_classifier_layer_batch_2DBlocks(VTYPE *dsynapse, VTYPE *dneuron_i, VTYPE *dneuron_n) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  int j = (blockIdx.y * blockDim.y) + threadIdx.y;
  if(i < Nn && j < Ni) 
  {  
    for(int b=0; b<BATCH; ++b)
    {
      VTYPE temp = *(dsynapse + i*Ni + j) * (*(dneuron_i + b*Ni + j));
      atomicAdd((dneuron_n + b*Nn + i),temp);
    }
  }
}

__global__ void cuda_classifier_layer_batch_3DBlocks(VTYPE *dsynapse, VTYPE *dneuron_i, VTYPE *dneuron_n) {
  if(!SHARE)
  {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    int b = (blockIdx.z * blockDim.z) + threadIdx.z;
    if((i < Nn) && (j < Ni) && (b < BATCH)) 
    {  
      VTYPE temp = *(dsynapse + i*Ni + j) * (*(dneuron_i + b*Ni + j));
      atomicAdd((dneuron_n + b*Nn + i),temp);
    }
  }

  else
  { 
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    int b = (blockIdx.z * blockDim.z) + threadIdx.z;
    __shared__ VTYPE  temp[threadsPerBlockPerDim3DBATCH][threadsPerBlockPerDim3D][threadsPerBlockPerDim3D];
    if((i < Nn) && (j < Ni) && (b < BATCH)) 
    {  
      temp[threadIdx.x][threadIdx.y][threadIdx.z] = *(dsynapse + i*Ni + j) * (*(dneuron_i + b*Ni + j));

      __syncthreads();

      if(0==threadIdx.x)
      {
        VTYPE sum = 0.0;
        for(int ii=0; ii < threadsPerBlockPerDim3DBATCH; ++ii)
          for(int jj=0; jj < threadsPerBlockPerDim3D; ++jj)
            for(int kk=0; kk < threadsPerBlockPerDim3D; ++kk)
              sum += temp[ii][jj][kk];
        atomicAdd((dneuron_n + b*Nn + i),sum);
      }

      __syncthreads();
    }
  }
}

__global__ void cuda_transfer_array(VTYPE *dneuron_n)
{
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  VTYPE temp = dneuron_n[i];
  temp = (temp > 0) ? temp : temp/4.0;
  dneuron_n[i] = temp;
}

void batch_classifier_layer(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[BATCH][Ni], VTYPE (&neuron_n)[BATCH][Nn]) {
  VTYPE val=0;
  int i,j,k;
  for(i=0; i<BATCH; i++)
  {
  for(j=0; j<Nn; j++)
  {
    for(k=0; k<Ni; k++)
    {
      val = synapse[j][k] * neuron_i[i][k];
      //neuron_n[i][j] = transfer(val);
      neuron_n[i][j] += (val);
    }
    neuron_n[i][j] = transfer(neuron_n[i][j]);
  }
  }
}

bool batch_compare(VTYPE (&neuron_n1)[BATCH][Nn], VTYPE (&curr)[BATCH][Nn])
{
  bool error= false;
  for(int i=0; i<BATCH; i++)
  {
    for(int j=0; j<Nn; j++)
    {
      //cout << "neuron_gold= " << neuron_n1[i][j] << "\tneuron_n=" << curr[i][j] << "\n";
      if(abs(neuron_n1[i][j] - curr[i][j]) > 0.0001f)
      {
        error =  true;
        cout << "return\n"; 
        return error;
      }
    }
  }
  return error;
}

void classifier_layer_blocked(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni], 
                              VTYPE (&neuron_n)[Nn]) {
  VTYPE sum[Nn]={0};
  for (int nnn = 0; nnn < Nn; nnn += Tnn) { // tiling for output neurons;
    for (int iii = 0; iii < Ni; iii += Tii) { // tiling for input neurons;
      for (int nn = nnn; nn < nnn + Tnn; nn += Tn) {
        for (int ii = iii; ii < iii + Tii; ii += Ti) {
          // — Original code —
          for (int n = nn; n < nn + Tn; n++) {
            VTYPE sum_sc=0;
            for (int i = ii; i < ii + Ti; i++) {
              sum_sc += (synapse[n][i] * neuron_i[i]);
            }
            sum[n]+=sum_sc;
          }
        }
      }
    }
    for (int nn = nnn; nn < nnn + Tnn; nn++) {
      neuron_n[nn] = transfer(sum[nn]);
    }
  }
}

int main(int argc, char** argv) {
  cout << "initializing arrays\n";

  fill_classifier(synapse,neuron_i,neuron_n,neuron_n2,batch_neuron_i,batch_neuron_n,batch_neuron_n_gold);

  cout << "starting computation\n";

  begin_roi();
  classifier_layer(synapse,neuron_i,neuron_n);
  end_roi();

  cout << "simple version complete!\n";  

  VTYPE * dev_synapse;  
  VTYPE * dev_neuron_i;  
  VTYPE * dev_neuron_n; 
  VTYPE * dev_neuron_n2D; 
  VTYPE * neuron_n2_from_dev; 
  neuron_n2_from_dev = (VTYPE*) malloc(Nn*sizeof(VTYPE));
  VTYPE * neuron_n2D_from_dev; 
  neuron_n2D_from_dev = (VTYPE*) malloc(Nn*sizeof(VTYPE));

  VTYPE * dev_batch_neuron_i;  
  VTYPE * dev_batch_neuron_n; 
  VTYPE * dev_batch_neuron_n3D; 

  cudaError_t err = cudaSuccess;
  err = cudaMalloc(&dev_synapse,  Ni*Nn*sizeof(VTYPE));
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device synapse (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  err = cudaMalloc(&dev_neuron_i, Ni*sizeof(VTYPE));
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device neuron_i (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  err = cudaMalloc(&dev_neuron_n, Nn*sizeof(VTYPE));
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device dev_neuron_n (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(dev_synapse, synapse, Ni*Nn*sizeof(VTYPE), cudaMemcpyHostToDevice);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy synapse from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(dev_neuron_i, neuron_i, Ni*sizeof(VTYPE), cudaMemcpyHostToDevice);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy neuron_i from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  cout << "begin cuda 1D blocks simple version\n";  
  begin_roi();
  cuda_classifier_layer_1DBlocks <<< blocksPerGrid, threadsPerBlock >>> (dev_synapse,dev_neuron_i,dev_neuron_n);
  cuda_transfer_array <<< transferThreadsPerBlock, transferBlocksPerGrid >>> (dev_neuron_n);
  err = cudaMemcpy(neuron_n2_from_dev, dev_neuron_n, Nn*sizeof(VTYPE), cudaMemcpyDeviceToHost);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy neuron_n2_from_dev from device to host (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
  //for (int n=0; n<Nn; ++n)
  //  *(neuron_n2_from_dev + n) = (*(neuron_n2_from_dev + n) > 0)
  //    ? *(neuron_n2_from_dev + n) : (*(neuron_n2_from_dev + n)) /4.0;
  end_roi();

  cout << "cuda 1D blocks simple version complete!\n";  
  err = cudaMalloc(&dev_neuron_n2D, Nn*sizeof(VTYPE));
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device dev_neuron_n2D (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  dim3 threadsPerBlock2(threadsPerBlockPerDim2D, threadsPerBlockPerDim2D);
  dim3 numBlocks((Nn + threadsPerBlock2.x - 1)/threadsPerBlock2.x,  /* for instance 512/8 = 64*/
              (Ni + threadsPerBlock2.y -1)/threadsPerBlock2.y);
  cout << "begin cuda 2D blocks simple version\n";  
  begin_roi();
  cuda_classifier_layer_2DBlocks <<< numBlocks, threadsPerBlock2 >>>   (dev_synapse,dev_neuron_i,dev_neuron_n2D);   
  err = cudaMemcpy(neuron_n2D_from_dev, dev_neuron_n2D, Nn*sizeof(VTYPE), cudaMemcpyDeviceToHost);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy neuron_n2D_from_dev from device to host (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
  for (int n=0; n<Nn; ++n)
    *(neuron_n2D_from_dev + n) = (*(neuron_n2D_from_dev + n) > 0)
      ? *(neuron_n2D_from_dev + n) : (*(neuron_n2D_from_dev + n)) /4.0;
  end_roi();
  cout << "cuda 2D blocks simple version complete!\n";  

  err = cudaMalloc(&dev_batch_neuron_i, BATCH*Ni*sizeof(VTYPE));
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device batch_neuron_i (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  err = cudaMalloc(&dev_batch_neuron_n, BATCH*Nn*sizeof(VTYPE));
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device dev_batch_neuron_n (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(dev_batch_neuron_i, batch_neuron_i, BATCH*Ni*sizeof(VTYPE), cudaMemcpyHostToDevice);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy batch_neuron_i from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  cout << "begin cuda 2D blocks batch golden version\n";  
  begin_roi();
  batch_classifier_layer(synapse,batch_neuron_i,batch_neuron_n_gold);
  end_roi();
  cout << "cuda 2D blocks batch golden version complete!\n";  

  cout << "begin cuda 2D blocks batch version\n";  
  begin_roi();
  cuda_classifier_layer_batch_2DBlocks <<< numBlocks, threadsPerBlock2 >>>
    (dev_synapse,dev_batch_neuron_i,dev_batch_neuron_n);   
  err = cudaMemcpy(batch_neuron_n, dev_batch_neuron_n, BATCH*Nn*sizeof(VTYPE), cudaMemcpyDeviceToHost);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy dev_batch_neuron_n from device to host (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
  for (int b=0; b<BATCH; ++b)
    for (int n=0; n<Nn; ++n)
      batch_neuron_n[b][n] = (batch_neuron_n[b][n] > 0) ? batch_neuron_n[b][n]
        : batch_neuron_n[b][n]/4.0;
  end_roi();
  cout << "cuda 2D blocks batch version complete!\n";  

  err = cudaMalloc(&dev_batch_neuron_n3D, BATCH*Nn*sizeof(VTYPE));
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device dev_batch_neuron_n3D (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  cout << "begin cuda 3D blocks batch version\n";  
  dim3 threadsPerBlock3(threadsPerBlockPerDim3DBATCH, threadsPerBlockPerDim3D, threadsPerBlockPerDim3D);
  dim3 numBlocksPerGrid3D((Nn + threadsPerBlock3.x - 1)/threadsPerBlock3.x,  
              (Ni + threadsPerBlock3.y -1)/threadsPerBlock3.y, (BATCH + threadsPerBlock3.z -1)/threadsPerBlock3.z);
  begin_roi();
  cuda_classifier_layer_batch_3DBlocks <<< numBlocksPerGrid3D, threadsPerBlock3 >>>
    (dev_synapse,dev_batch_neuron_i,dev_batch_neuron_n3D);   
  err = cudaMemcpy(batch_neuron_n3D, dev_batch_neuron_n3D, BATCH*Nn*sizeof(VTYPE), cudaMemcpyDeviceToHost);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy dev_batch_neuron_n3D from device to host (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
  for (int b=0; b<BATCH; ++b)
    for (int n=0; n<Nn; ++n)
      batch_neuron_n3D[b][n] = (batch_neuron_n3D[b][n] > 0) ? batch_neuron_n3D[b][n]
        : batch_neuron_n3D[b][n]/4.0;
  end_roi();
  cout << "cuda 3D blocks batch version complete!\n";  

  begin_roi();
  classifier_layer_blocked(synapse,neuron_i,neuron_n2);  
  end_roi();

  cout << "blocked computation complete!\n";  

  compare(neuron_n,neuron_n2,Nn);
  compare(neuron_n2_from_dev,neuron_n2,Nn);
  compare(neuron_n2D_from_dev,neuron_n2,Nn);
  if(!(batch_compare(batch_neuron_n_gold,batch_neuron_n)))
    cout << "Batch results match!\n"; 
  else
    cout << "Batch results do not match!\n"; 

  if(!(batch_compare(batch_neuron_n_gold,batch_neuron_n3D)))
    cout << "Batch results match for 3D!\n"; 
  else
    cout << "Batch results do not match for 3D!\n"; 
    //cout << "\033[1;32mbold red text\033[0m\n";
  cout << "done\n";
  
}

