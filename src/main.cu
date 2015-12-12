#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include "Engine.h"
#include <ctime>

int main(int argc, char **argv)
{
	clock_t stop, start;

	std::cout << "Pomiar czasu dla 10000 cząsteczek i róznej ilości kroków" << std::endl;
	std::ofstream fileTiming;
	fileTiming.open("timing.txt");
	int iters = 30;
	int startNum = 500;
	int stepSize = 100;
	float* cpuTimingTable = new float[iters];
	float* cudaTimingTable = new float[iters];
	bool breakSize = false;
	for(int i = 0;i<iters;i++)
	{
		int currentStepSize = startNum + i*stepSize;

		Engine sym1(currentStepSize,10000,0.025);
		start = clock();
		sym1.start();
		stop = clock();
		cpuTimingTable[i] = (float)(stop-start)/CLOCKS_PER_SEC;


		sym1.reset();
		start = clock();
		sym1.startCUDA();
		stop = clock();
		cudaTimingTable[i] = (float)(stop-start)/CLOCKS_PER_SEC;
		std::cout << currentStepSize  << " CPU: " << cpuTimingTable[i] << " \t CUDA: " << cudaTimingTable[i] << std::endl;
		fileTiming << currentStepSize << " "  << cpuTimingTable[i] << " " << cudaTimingTable[i] << std::endl;
		if(!breakSize && cudaTimingTable[i] < cpuTimingTable[i])
		{
			breakSize = true;
			std::cout << "Punkt styczny" << std::endl;
		}
	}

	delete[] cpuTimingTable;
	delete[] cudaTimingTable;
	fileTiming.close();

	std::cout << "Generowanie histogramów dla 1000 kroków i 5000 cząsteczek" << std::endl;
	Engine sym2(1000,5000,0.025);
	sym2.startCUDA();
	sym2.savePositionHistToFile("posHist.txt",30);
	sym2.saveDistanceHistToFile("distHist.txt",30);
	sym2.saveFlipNumHistToFile("flipDirHist.txt",16);
	sym2.saveDistFlipNumDepToFile("distFlipNumDep.txt");
	sym2.reset();

	std::cout << "Generowanie histogramów średnich wartości dla 1000 kroków i 5000 cząsteczek przy 500 próbach" << std::endl;
	int numIters = 500;
	float *averPosition = new float[numIters];
	float *averDistance = new float[numIters];
	for(int i = 0;i<numIters;i++)
	{
		sym2.start();
		averPosition[i] = sym2.getAveragePosition();
		averDistance[i] = sym2.getAverageDistance();
		sym2.reset();
	}
	Engine::saveHistToFile("averPosHist.txt",averPosition,numIters,10);
	Engine::saveHistToFile("averDistHist.txt",averDistance,numIters,10);

	delete[] averPosition;
	delete[] averDistance;

	std::cout << "Zakończono" << std::endl;
	return 0;
}
