#pragma once

#include "cuda_runtime.h"
#define DATASIZE 40000
#include "device_launch_parameters.h"

typedef struct PointSet

{

	float *x;

	float *y;

	float *z;

	int size;

};
namespace cuda_function
{
		void FindCorrespondantPoint(PointSet &P, PointSet &Q, PointSet &X, float &E);

}