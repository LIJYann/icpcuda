#include <time.h>

#include <stdio.h>

#include<algorithm>

#include<Eigen/Eigenvalues>
#include<fstream>
#include<math.h>
#include <iostream>

#define DATASIZE 40000


using namespace Eigen;

using namespace std;

typedef struct PointSet
{
	float *x;
	float *y;
	float *z;
	int size;
};
typedef struct Rotation
{
	float x1, x2, x3, y1, y2, y3, z1, z2, z3;
};


void InputData(char* FILEPATH, PointSet &P)

{

	FILE *fp = NULL;

	float  f;
	P.size = 0;
	//*(P.x) = '\0';
	P.x = new float[DATASIZE];
	P.y = new float[DATASIZE];
	P.z = new float[DATASIZE];

	fp = fopen(FILEPATH, "r");

	while (fscanf(fp, "%f", &f) != EOF)

	{

		P.x[P.size] = f;

		fscanf(fp, "%f", &f);

		P.y[P.size] = f;

		fscanf(fp, "%f", &f);

		P.z[P.size] = f;
		P.size++;
	}



	if (fclose(fp)) { printf("error in closing the file"); }

}

void CalculateMeanPoint(PointSet &P, PointSet &mean)

{
	mean.x = new float;
	mean.y = new float;
	mean.z = new float;

	mean.x[0] = 0;
	mean.y[0] = 0;
	mean.z[0] = 0;

	for (int i = 0; i<P.size; i++)

	{

		mean.x[0] += P.x[i] / P.size;

		mean.y[0] += P.y[i] / P.size;

		mean.z[0] += P.z[i] / P.size;

	}

	mean.size = 1;

}

void MovingPointSet(PointSet &P, PointSet &T)

{

	for (int i = 0; i<P.size; i++)

	{

		P.x[i] += T.x[0];

		P.y[i] += T.y[0];

		P.z[i] += T.z[0];

	}

}

void FindCorrespondantPoint(PointSet &P, PointSet &Q, PointSet &X, float &E)
{
	E = 0;
	for (int i = 0; i < P.size; i++)
	{
		float mindist = 100000;
		int pos = 0;
		for (int j = 0; j < Q.size; j++)
		{
			float dist = pow(P.x[i] - Q.x[j], 2) + pow(P.y[i] - Q.y[j], 2) + pow(P.z[i] - Q.z[j], 2);
			if (mindist > dist) { mindist = dist; pos = j; }
		}
		E += mindist / P.size;
		X.x[i] = Q.x[pos];
		X.y[i] = Q.y[pos];
		X.z[i] = Q.z[pos];
	}
}

void CalculateRotation(PointSet &P, PointSet &X, Rotation &R, PointSet &P_mean, PointSet &X_mean)
{
	float cov[9] = { 0,0,0,0,0,0,0,0,0 };
	for (int i = 0; i < P.size; i++)
	{
		cov[0] += (P.x[i] - P_mean.x[0])*(X.x[i] - X_mean.x[0]) / P.size;//(1,1)
		cov[1] += (P.x[i] - P_mean.x[0])*(X.y[i] - X_mean.y[0]) / P.size;//(1,2)
		cov[2] += (P.x[i] - P_mean.x[0])*(X.z[i] - X_mean.z[0]) / P.size;//(1,3)
		cov[3] += (P.y[i] - P_mean.y[0])*(X.x[i] - X_mean.x[0]) / P.size;//(2,1)
		cov[4] += (P.y[i] - P_mean.y[0])*(X.y[i] - X_mean.y[0]) / P.size;//(2,2)
		cov[5] += (P.y[i] - P_mean.y[0])*(X.z[i] - X_mean.z[0]) / P.size;//(2,3)
		cov[6] += (P.z[i] - P_mean.z[0])*(X.x[i] - X_mean.x[0]) / P.size;//(3,1)
		cov[7] += (P.z[i] - P_mean.z[0])*(X.y[i] - X_mean.y[0]) / P.size;//(3,2)
		cov[8] += (P.z[i] - P_mean.z[0])*(X.z[i] - X_mean.z[0]) / P.size;//(3,3)
	}
	Matrix3f B;
	B << cov[0], cov[1], cov[2], cov[3], cov[4], cov[5], cov[6], cov[7], cov[8];
	//cout<<B<<endl;
	//build 4*4 symetric matrix
	float Q[16];
	//first line
	Q[0] = cov[0] + cov[4] + cov[8];
	Q[1] = cov[5] - cov[7];
	Q[2] = cov[6] - cov[2];
	Q[3] = cov[1] - cov[3];
	//second line
	Q[4] = cov[5] - cov[7];
	Q[5] = cov[0] - cov[4] - cov[8];
	Q[6] = cov[1] + cov[3];
	Q[7] = cov[6] + cov[2];
	//third line
	Q[8] = cov[6] - cov[2];
	Q[9] = cov[1] + cov[3];
	Q[10] = cov[4] - cov[0] - cov[8];
	Q[11] = cov[5] + cov[7];
	//last line
	Q[12] = cov[1] - cov[3];
	Q[13] = cov[2] + cov[6];
	Q[14] = cov[5] + cov[7];
	Q[15] = cov[8] - cov[0] - cov[4];

	Matrix4f A;
	A << Q[0], Q[1], Q[2], Q[3], Q[4], Q[5], Q[6], Q[7], Q[8], Q[9], Q[10], Q[11], Q[12], Q[13], Q[14], Q[15];
	//cout<<A<<endl;
	EigenSolver<Matrix4f> es(A);

	Matrix4f D = es.pseudoEigenvalueMatrix();
	Matrix4f V = es.pseudoEigenvectors();

	float biggestValue = A(0, 0);
	int pos = 0;
	for (int i = 1; i<4; i++) {
		if (biggestValue<D(i, i)) { biggestValue = D(i, i); pos = i; }
	}
	float q[] = { V(0,pos), V(1,pos), V(2,pos), V(3,pos) };

	//calculate rotation matrix with unit quaternion

	R.x1 = q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3];//(1,1)

	R.y1 = 2 * (q[1] * q[2] - q[0] * q[3]);//(1,2)

	R.z1 = 2 * (q[1] * q[3] + q[0] * q[2]);//(1,3)

	R.x2 = 2 * (q[1] * q[2] + q[0] * q[3]);//(2,1)

	R.y2 = q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3];//(2,2)

	R.z2 = 2 * (q[2] * q[3] - q[0] * q[1]);//(2,3)

	R.x3 = 2 * (q[1] * q[3] - q[0] * q[2]);//(3,1)

	R.y3 = 2 * (q[2] * q[3] + q[0] * q[1]);//(3,2)

	R.z3 = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3];//(3,3)

}

void Rotate(PointSet &P, Rotation &R)
{
	for (int i = 0; i < P.size; i++)
	{
		float x = P.x[i] * R.x1 + P.y[i] * R.y1 + P.z[i] * R.z1;
		float y = P.x[i] * R.x2 + P.y[i] * R.y2 + P.z[i] * R.z2;
		float z = P.x[i] * R.x3 + P.y[i] * R.y3 + P.z[i] * R.z3;
		P.x[i] = x;
		P.y[i] = y;
		P.z[i] = z;
	}
}

int main()

{
	float E = 1;
	PointSet P, Q, X, P_mean, Q_mean, X_mean, T;
	Rotation R;
	clock_t start = clock();
	char *FILEPATH = "C:\\Users\\LIJ\\Desktop\\icp0\\data\\bunny.asc";
	InputData(FILEPATH, P);
	FILEPATH = "C:\\Users\\LIJ\\Desktop\\icp0\\data\\bunny_perturbed.asc";
	InputData(FILEPATH, Q);
	printf("scanning cost %f seconds\n", (float)(clock() - start) / CLOCKS_PER_SEC);
	int itcount = 0;

	X.x = new float[P.size];
	X.y = new float[P.size];
	X.z = new float[P.size];


	start = clock();
	CalculateMeanPoint(Q, Q_mean);
	printf("CalculateMeanPoint for Q cost %f seconds\n", (float)(clock() - start) / CLOCKS_PER_SEC);

	while (E > 0.00001&&itcount < 5)
	{
		start = clock();
		FindCorrespondantPoint(P, Q, X, E);
		printf("FindCorrespondantPoint cost %f seconds\n", (float)(clock() - start) / CLOCKS_PER_SEC);

		start = clock();
		CalculateMeanPoint(P, P_mean);
		printf("CaluculateMeanPoint for P cost %f seconds\n", (float)(clock() - start) / CLOCKS_PER_SEC);

		start = clock();
		CalculateMeanPoint(X, X_mean);
		printf("CalculateMeanPoint for X cost %f seconds\n", (float)(clock() - start) / CLOCKS_PER_SEC);

		start = clock();
		CalculateRotation(P, X, R, P_mean, X_mean);
		printf("CalculateRotation cost %f seconds\n", (float)(clock() - start) / CLOCKS_PER_SEC);

		start = clock();
		Rotate(P, R);
		printf("Rotate cost %f  seconds\n", (float)(clock() - start) / CLOCKS_PER_SEC);

		T.x = new float;
		T.y = new float;
		T.z = new float;
		T.x[0] = Q_mean.x[0] - P_mean.x[0];
		T.y[0] = Q_mean.y[0] - P_mean.y[0];
		T.z[0] = Q_mean.z[0] - P_mean.z[0];

		start = clock();
		MovingPointSet(P, T);
		printf("MovingPointSet cost %f seconds\n", (float)(clock() - start) / CLOCKS_PER_SEC);

		X.size = 0;
		itcount++;
		printf("%d iteration: %f\n", itcount, E);
	}

	start = clock();
	FILEPATH = "C:\\Users\\LIJ\\Desktop\\icp0\\data\\bunny_changed_VS.asc";
	FILE *fp = fopen(FILEPATH, "w");
	for (int i = 0; i<P.size; i++)
	{
		fprintf(fp, "%f ", P.x[i]);
		fprintf(fp, "%f ", P.y[i]);
		fprintf(fp, "%f\n", P.z[i]);
	}
	fclose(fp);
	fp = NULL;
	printf("output data cost %f seconds\n", (float)(clock() - start) / CLOCKS_PER_SEC);


	return 0;

}