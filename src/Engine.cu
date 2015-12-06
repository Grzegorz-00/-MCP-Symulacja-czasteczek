#include "Engine.h"

__global__ void setup_curand(curandState* state, unsigned long seed)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, idx, 0, &state[idx]);
}

__device__ float generate_curand(curandState* globalState, int idx)
{
	curandState localState = globalState[idx];
	float random = curand_uniform(&localState);
	globalState[idx] = localState;
	return random;
}

__global__ void stepCUDA(float* positionTable ,float* distanceTable, float* velocityTable ,int  size, int duration, float acceleration, curandState *globalState)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < size)
	{
		for(int j = 0;j<duration;j++)
		{
			distanceTable[i] += fabs(velocityTable[i]);
			positionTable[i] += velocityTable[i];
			if(positionTable[i] > 0)
			{
				velocityTable[i] -= acceleration;
			}
			else if(positionTable[i] < 0)
			{
				velocityTable[i] += acceleration;
			}

			if(generate_curand(globalState,i) >= 0.5)
			{
				velocityTable[i] = -velocityTable[i];
			}
		}

	}

}

Engine::Engine(int duration, int size, float acceleration)
{
	_duration = duration;
	_size = size;
	_acceleration = acceleration;
	_positionTable = new float[_size];
	_distanceTable = new float[_size];
	_velocityTable = new float[_size];

	std::default_random_engine generator(std::random_device{}());
	std::normal_distribution<float> distribution(1,2.5);

	for(int i = 0;i<_size;i++)
	{
		_velocityTable[i] = distribution(generator);
		_distanceTable[i] = 0;
		_positionTable[i] = 0;
	}
}
Engine::~Engine()
{
	delete _positionTable;
	delete _distanceTable;
	delete _velocityTable;
}

void Engine::reset()
{
	std::default_random_engine generator(std::random_device{}());
	std::normal_distribution<float> distribution(1,2.5);

	for(int i = 0;i<_size;i++)
	{
		_velocityTable[i] = distribution(generator);
		_distanceTable[i] = 0;
		_positionTable[i] = 0;
	}
}

void Engine::start()
{
	srand(time(NULL));
	step();

}

void Engine::startCUDA()
{

	int block_size = 128;
	int block_num = (_size + block_size - 1)/block_size;
	float *kernVelocityTable;
	float *kernDistanceTable;
	float *kernPositionTable;
	curandState *kernGlobalState;

	cudaMalloc((void**)&kernVelocityTable,sizeof(float)*_size);
	cudaMalloc((void**)&kernDistanceTable,sizeof(float)*_size);
	cudaMalloc((void**)&kernPositionTable,sizeof(float)*_size);
	cudaMalloc((void**)&kernGlobalState,sizeof(float)*_size);

	cudaMemcpy(kernVelocityTable,_velocityTable,sizeof(float)*_size,cudaMemcpyHostToDevice);
	cudaMemcpy(kernDistanceTable,_distanceTable,sizeof(float)*_size,cudaMemcpyHostToDevice);
	cudaMemcpy(kernPositionTable,_positionTable,sizeof(float)*_size,cudaMemcpyHostToDevice);

	setup_curand<<<block_num,block_size>>>(kernGlobalState,time(NULL));


	stepCUDA <<<block_num,block_size>>>(kernPositionTable, kernDistanceTable, kernVelocityTable,  _size, _duration, _acceleration, kernGlobalState);


	cudaMemcpy(_distanceTable,kernDistanceTable,sizeof(float)*_size,cudaMemcpyDeviceToHost);
	cudaMemcpy(_positionTable,kernPositionTable,sizeof(float)*_size,cudaMemcpyDeviceToHost);

	cudaFree(kernDistanceTable);
	cudaFree(kernPositionTable);
	cudaFree(kernVelocityTable);
	cudaFree(kernGlobalState);

}

void Engine::print()
{
	std::cout << "distance vector" << std::endl;
	for(int i = 0;i<_size;i++)
	{
		std::cout << _distanceTable[i] << " ";
	}

	std::cout << std::endl << "position vector" << std::endl;
	for(int i = 0;i<_size;i++)
	{
		std::cout << _positionTable[i] << " ";
	}

	std::cout << std::endl << "velocity vector" << std::endl;
	for(int i = 0;i<_size;i++)
	{
		std::cout << _velocityTable[i] << " ";
	}
}

void Engine::savePositionToFile(std::string filename)
{
	std::ofstream file;
	file.open(filename);
	for(int i = 0;i<_size;i++)
	{
		file << _positionTable[i] << std::endl;
	}
	file.close();

}
void Engine::saveDistanceToFile(std::string filename)
{

	std::ofstream file;
	file.open(filename);
	for(int i = 0;i<_size;i++)
	{
		file << _distanceTable[i] << std::endl;
	}
	file.close();
}

void Engine::step()
{
	for(int i = 0;i<_size;i++)
	{
		for(int j = 0;j<_duration;j++)
		{
			_distanceTable[i] += fabs(_velocityTable[i]);
			_positionTable[i] += _velocityTable[i];
			if(_positionTable[i] > 0)
			{
				_velocityTable[i] -= _acceleration;
			}
			else if(_positionTable[i] < 0)
			{
				_velocityTable[i] += _acceleration;
			}

			if(rand()%2 == 0)
			{
				_velocityTable[i] = -_velocityTable[i];
			}
		}
	}
}

float Engine::getAveragePosition()
{
	float averPosition = 0;
	for(int i = 0;i<_size;i++)
	{
		averPosition += _positionTable[i];
	}
	averPosition /= _size;
	return averPosition;

}
float Engine::getAverageDistance()
{
	float averDistance = 0;
	for(int i = 0;i<_size;i++)
	{
		averDistance += _distanceTable[i];
	}
	averDistance /= _size;
	return averDistance;
}

void Engine::savePositionHistToFile(std::string filename, int bids)
{
	saveHistToFile(filename,_positionTable,_size,bids);
}
void Engine::saveDistanceHistToFile(std::string filename, int bids)
{
	saveHistToFile(filename,_distanceTable,_size,bids);
}

void Engine::saveHistToFile(std::string filename, float* data, int dataSize, int bids)
{
	std::ofstream file;
	file.open(filename);
	float min = data[0];
	float max = data[0];
	for(int i = 1;i<dataSize;i++)
	{
		if(data[i] > max)max = data[i];
		else if(data[i] < min)min = data[i];
	}
	float stepSize = (max-min)/bids;

	for(int i = 0;i<bids-1;i++)
	{
		float minRange = min + i*stepSize;
		float maxRange = min + (i+1)*stepSize;
		int rangeOccurs = 0;
		for(int j = 0;j<dataSize;j++)
		{
			if(data[j] >= minRange && data[j] < maxRange)rangeOccurs++;
		}

		file << (minRange+maxRange)/2 << " " << rangeOccurs << std::endl;
	}
	file.close();
}

void Engine::saveHistToFileRange(std::string filename, float* data, int dataSize, int bids, float min, float max)
{
	std::ofstream file;
	file.open(filename);

	float stepSize = (max-min)/bids;

	for(int i = 0;i<bids-1;i++)
	{
		float minRange = min + i*stepSize;
		float maxRange = min + (i+1)*stepSize;
		int rangeOccurs = 0;
		for(int j = 0;j<dataSize;j++)
		{
			if(data[j] >= minRange && data[j] < maxRange)rangeOccurs++;
		}

		file << (minRange+maxRange)/2 << " " << rangeOccurs << std::endl;
	}
	file.close();
}
