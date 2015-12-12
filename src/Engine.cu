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

__global__ void stepCUDA(float* positionTable ,float* distanceTable, float* velocityTable , int* numFlipTable, int  size, int duration, float acceleration, curandState *globalState)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < size)
	{
		distanceTable[i] = 0;
		positionTable[i] = 0;
		numFlipTable[i] = 0;

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
				numFlipTable[i]++;

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
	_flipDirTable = new int[_size];

	std::default_random_engine generator(std::random_device{}());
	std::normal_distribution<float> distribution(1,2.5);

	for(int i = 0;i<_size;i++)
	{
		_velocityTable[i] = distribution(generator);
		_distanceTable[i] = 0;
		_positionTable[i] = 0;
		_flipDirTable[i] = 0;
	}
}
Engine::~Engine()
{
	delete _positionTable;
	delete _distanceTable;
	delete _velocityTable;
	delete _flipDirTable;
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
		_flipDirTable[i] = 0;
	}
}

void Engine::start()
{
	srand(time(NULL));
	step();

}

void Engine::notifyCudaAllocError()
{
	std::cout << "CUDA Alloc problem" << std::endl;
	_errorOccur = true;
}

void Engine::notifyCudaCpyError()
{
	std::cout << "CUDA Memcpy problem" << std::endl;
	_errorOccur = true;
}
void Engine::startCUDA()
{

	int block_size = 128;
	int block_num = (_size + block_size - 1)/block_size;
	float *kernVelocityTable;
	float *kernDistanceTable;
	float *kernPositionTable;
	int *kernFlipDirTable;
	curandState *kernGlobalState;

	if(cudaMalloc((void**)&kernVelocityTable,sizeof(float)*_size)!=cudaSuccess)notifyCudaAllocError();
	if(cudaMalloc((void**)&kernDistanceTable,sizeof(float)*_size)!=cudaSuccess)notifyCudaAllocError();
	if(cudaMalloc((void**)&kernPositionTable,sizeof(float)*_size)!=cudaSuccess)notifyCudaAllocError();
	if(cudaMalloc((void**)&kernFlipDirTable,sizeof(int)*_size)!=cudaSuccess)notifyCudaAllocError();
	if(cudaMalloc((void**)&kernGlobalState,sizeof(float)*_size)!=cudaSuccess)notifyCudaAllocError();

	if(_errorOccur == false)
	{
		if(cudaMemcpy(kernVelocityTable,_velocityTable,sizeof(float)*_size,cudaMemcpyHostToDevice)!=cudaSuccess)notifyCudaCpyError();
	}

	if(_errorOccur == false)
	{
		setup_curand<<<block_num,block_size>>>(kernGlobalState,time(NULL));
		stepCUDA <<<block_num,block_size>>>(kernPositionTable, kernDistanceTable, kernVelocityTable, kernFlipDirTable, _size, _duration, _acceleration, kernGlobalState);
	}

	if(cudaMemcpy(_distanceTable,kernDistanceTable,sizeof(float)*_size,cudaMemcpyDeviceToHost)!=cudaSuccess)notifyCudaCpyError();
	if(cudaMemcpy(_positionTable,kernPositionTable,sizeof(float)*_size,cudaMemcpyDeviceToHost)!=cudaSuccess)notifyCudaCpyError();
	if(cudaMemcpy(_flipDirTable,kernFlipDirTable,sizeof(int)*_size,cudaMemcpyDeviceToHost)!=cudaSuccess)notifyCudaCpyError();

	cudaFree(kernDistanceTable);
	cudaFree(kernPositionTable);
	cudaFree(kernVelocityTable);
	cudaFree(kernFlipDirTable);
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

void Engine::saveDistFlipNumDepToFile(std::string filename)
{
	std::ofstream file;
	file.open(filename);
	for(int i = 0;i<_size;i++)
	{
		file << _distanceTable[i] << " " << _flipDirTable[i] << std::endl;
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
				_flipDirTable[i]++;
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

float Engine::getAverageFlipNum()
{
	float averFlipNum = 0;
	for(int i = 0;i<_size;i++)
	{
		averFlipNum += _flipDirTable[i];
	}
	averFlipNum /= _size;
	return averFlipNum;
}

void Engine::savePositionHistToFile(std::string filename, int bids)
{
	saveHistToFile<float>(filename,_positionTable,_size,bids);
}
void Engine::saveDistanceHistToFile(std::string filename, int bids)
{
	saveHistToFile<float>(filename,_distanceTable,_size,bids);
}

void Engine::saveFlipNumHistToFile(std::string filename, int bids)
{
	saveHistToFile<int>(filename,_flipDirTable,_size,bids);
}

template<class T>void Engine::saveHistToFile(std::string filename, T* data, int dataSize, int bids)
{
	std::ofstream file;
	file.open(filename);
	T min = data[0];
	T max = data[0];
	for(int i = 1;i<dataSize;i++)
	{
		if(data[i] > max)max = data[i];
		else if(data[i] < min)min = data[i];
	}
	double stepSize = (double)(max-min)/bids;

	for(int i = 0;i<bids-1;i++)
	{
		double minRange = min + i*stepSize;
		double maxRange = min + (i+1)*stepSize;
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
