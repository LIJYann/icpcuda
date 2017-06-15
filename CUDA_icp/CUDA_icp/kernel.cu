
#include "cuda_runtime.h"

#include "device_launch_parameters.h"

#include<thrust/device_vector.h>

#include<thrust/sort.h>

#include "cuda_function.h"

#include <math.h>

#include<time.h>

#define  BLOCK_NUM 32
#define  THREAD_NUM 1024

using namespace cuda_function;


__global__ void Distance(float px, float py, float pz, float*qx, float *qy, float *qz, int qsize, float *dist)

{

	int tid = threadIdx.x;

	int bid = blockIdx.x;

		for (int j = tid + bid*THREAD_NUM; j < qsize; j += BLOCK_NUM*THREAD_NUM)
		{
			dist[j] = pow(px - *(qx + j), 2) + pow(py - *(qy + j), 2) + pow(pz - *(qz + j), 2);
			 //printf("%d:  %f\n", j, dist[j]);
			//if (dist[j] - 0 < 0.0000001) { printf("%d: %f\n", j, dist[j]); }
		}
		__syncthreads();
}


void cuda_function::FindCorrespondantPoint(PointSet &P, PointSet &Q, PointSet &X, float &E)
{
	float *qx, *qy, *qz,*dist;
	E = 0;
	cudaMalloc((void**)&qx, Q.size * sizeof(float));
	cudaMemcpy(qx, Q.x, Q.size * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&qy, Q.size * sizeof(float));
	cudaMemcpy(qy, Q.y, Q.size * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&qz, Q.size * sizeof(float));
	cudaMemcpy(qz, Q.z, Q.size * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dist, Q.size * sizeof(float));
	float *distance;
	distance = new float[Q.size];
	/*printf("%f\n", px[0]);
	Distance << <BLOCK_NUM, THREAD_NUM >> > (P.x[0], P.y[0], P.z[0], qx, qy, qz, Q.size, dist);
	cudaDeviceSynchronize();
	cudaMemcpy(distance, dist, Q.size * sizeof(float), cudaMemcpyDeviceToHost);
	printf("%f,%f\n", dist[0], distance[0]);*/
	for (int i = 0; i < P.size; i++)
	{
		//clock_t s=clock();
		//printf("%d:\n", i);
		Distance << <BLOCK_NUM, THREAD_NUM >> > (P.x[i], P.y[i], P.z[i], qx, qy, qz, Q.size, dist);
		cudaDeviceSynchronize();
		//printf("calculation cost %f seconds\n", (float)(clock() - s) / CLOCKS_PER_SEC);		
		cudaMemcpy(&*distance, dist, Q.size * sizeof(float), cudaMemcpyDeviceToHost);

		//printf("memcopy cost %f seconds\n", (float)(clock() - s) / CLOCKS_PER_SEC);
		float mindist = 1000000;
		int pos;
		for (int j = 0; j < Q.size; j++)
		{
			if (mindist > distance[j]) {
				mindist = distance[j];
				pos = j;
			}
		}
		
		//printf("comparation cost %f seconds\n", (float)(clock() - s) / CLOCKS_PER_SEC);
		E += mindist/P.size;// printf("%f, %d\n", mindist,pos);
		//printf("this00 cost %f seconds\n", (float)(clock() - s) / CLOCKS_PER_SEC);
		X.x[i] = Q.x[pos];
		X.y[i] = Q.y[pos];
		X.z[i] = Q.z[pos];
		//printf("this01 cost %f seconds\n", (float)(clock() - s) / CLOCKS_PER_SEC);
		
	}
	cudaFree(qx);
	cudaFree(qy);
	cudaFree(qz);
	cudaFree(dist);
}
