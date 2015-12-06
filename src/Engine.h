#ifndef ENGINE_H_
#define ENGINE_H_

#include <random>
#include <iostream>
#include <fstream>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

class Engine
{
public:
	Engine(int duration, int size, float acceleration);
	~Engine();
	void reset();
	void start();
	void startCUDA();
	void print();
	void savePositionToFile(std::string filename);
	void saveDistanceToFile(std::string filename);
	void savePositionHistToFile(std::string filename, int bids);
	void saveDistanceHistToFile(std::string filename, int bids);
	float getAveragePosition();
	float getAverageDistance();
	static void saveHistToFile(std::string filename, float* data, int dataSize, int bids);
	static void saveHistToFileRange(std::string filename, float* data, int dataSize, int bids, float min, float max);


private:
	void step();
	float *_positionTable;
	float *_distanceTable;
	float *_velocityTable;
	int _size;
	int _duration;
	float _acceleration;

};



#endif
