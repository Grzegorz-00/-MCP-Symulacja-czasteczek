#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include "Engine.h"
#include <ctime>

int main(int argc, char **argv)
{
	clock_t stop, start;

	Engine sym1(5000,20000,0.025);
	start = clock();
	sym1.start();

	stop = clock();
	float time = (float)(stop-start)/CLOCKS_PER_SEC;
	std::cout << "CPU: " << time << std::endl;

	sym1.reset();
	start = clock();

	sym1.startCUDA();

	stop = clock();
	time = (float)(stop-start)/CLOCKS_PER_SEC;
	std::cout << "CUDA: " << time << std::endl;

	Engine sym2(1000,5000,0.025);

	sym2.savePositionHistToFile("posHist.txt",30);
	sym2.saveDistanceHistToFile("distHist.txt",30);

	sym2.reset();

	int numIters = 500;
	float *averPosition = new float[numIters];
	float *averDistance = new float[numIters];
	for(int i = 0;i<numIters;i++)
	{
		sym2.startCUDA();
		averPosition[i] = sym2.getAveragePosition();
		averDistance[i] = sym2.getAverageDistance();
		sym2.reset();
	}
	Engine::saveHistToFile("averPosHist.txt",averPosition,numIters,10);
	Engine::saveHistToFile("averDistHist.txt",averDistance,numIters,10);

	delete averPosition;
	delete averDistance;

	return 0;
}
