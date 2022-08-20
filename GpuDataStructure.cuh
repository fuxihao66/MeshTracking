#pragma once
#ifndef GPU_DATA_STRUCTURE_H_
#define GPU_DATA_STRUCTURE_H_

#include <vector>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "khash.h"




struct Triangle {
    float3 point0;
    float3 point1;
    float3 point2;
};

//Tomas Akenine-Möller
inline __device__ bool TrianglePairIntersectionTest(const Triangle& t0, const Triangle& t1) {

}
inline __device__ void GetTriangle(float* triangleBuffer, uint32_t triangleIndex, Triangle& tri) {

}
// use template to support 2d and 3d
//template<typename ValType>
struct TriangleSpatialHashMap {
    struct MapData {
        uint32_t usedSlotNum;
        uint32_t triIndex;
    };
private:
    float* cudaMemoryBuffer;
    uint32_t mEntriesPerCell;
    uint32_t mXSize;
    uint32_t mYSize;
    uint32_t mZSize;

    float mLeft;
    float mRight;
    float mNear;
    float mFar;
    float mBottom;
    float mUp;

    uint32_t IndexMapping(float x, float y, float z) {
        int grid_x = ;
        int grid_y = ;
        int grid_z = ;

        return grid_x*mYSize*mZSize* mEntriesPerCell + grid_y *mZSize * mEntriesPerCell + grid_z *mEntriesPerCell ;
    }
public:
    TriangleSpatialHashMap(float far, float near, float left, float right, float bottom, float up, float gridSize, uint32_t entriesPerCell):
        mLeft(left), mRight(right), mNear(near), mFar(far), mBottom(bottom), mUp(up)
    {
        
        mXSize = ceil((near - far) / gridSize);
        mYSize = ceil((right - left) / gridSize);
        mZSize = ceil((up - bottom) / gridSize);
        mEntriesPerCell = entriesPerCell;
        cudaError_t cudaStatus = cudaMalloc((void**)&cudaMemoryBuffer, (mXSize * mYSize * mZSize) * (mEntriesPerCell + 1) * sizeof(MapData));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
        }
    }

    void InsertToCell(float x, float y, float z) {
        uint32_t index = IndexMapping(x, y, z);
        while (true) {
            uint32_t oldUsedSlotNum = atomicAdd(cudaMemoryBuffer[index].usedSlotNum, 1);
            if (oldUsedSlotNum >= mEntriesPerCell) {
                atomicExch(cudaMemoryBuffer[index].usedSlotNum, entriesPerCell);
            }
            else {
                cudaMemoryBuffer[index + oldUsedSlotNum + 1] = ;
                break;
            }
            index = (index + (mEntriesPerCell + 1)) & ();
        }
        

    }
    void InsertTriangle(Triangle& tri) {

        // find all cells

        for () {
            InsertToCell();
        }
    }

   

    // 该三角形所
    bool TestIntersected( float* triangleBuffer, uint32_t triangleIndex) {
        Triangle triangleToTestIntersect;
        GetTriangle(triangleBuffer, triangleIndex, triangleToTestIntersect);


    }

    // 这里有两种做法
    // 1. intersected buffer不做atomic，每个thread的那个三角形找到相交三角形则直接+1，然后退出
    // 2. intersected buffer做atomic，每个thread的三角形以及和它相交的所有三角形都更新到intersected buffer中
    // given triangle index and triangle list, update triangles that intersected with this triangle to intersected buffer
    void UpdateIntersectedTriangles(float* triangleBuffer, uint32_t triangleIndex, bool* intersectedBuffer) {
        bool intersected = TestIntersected(triangleBuffer, triangleIndex);
        intersectedBuffer[triangleIndex] = intersected ? 1 : 0;  // 第1种做法
    }

    

};



inline uint64_t EdgeToKey(const uint32_t startVertex, const uint32_t endVertex) {
    uint64_t temp = startVertex;
    return temp << 32 + endVertex;
}

#define DEFAULT_MAP_SIZE 0xFFFFFFFF
//#define MAX_SEARCH_RANGE 1000

/*
 * https://github.com/nosferalatu/SimpleGPUHashTable
*/

#define INVALID_KEY_VAL 0xFFFFFFFFFFFFFFFF




template<typename ValType>
struct KeyValueMapGPU {
    struct Key {
        //bool valid;
        uint64_t key; // use -1 to indicate validality ( 0xFFFFFFFFFFFFFFFF )
    };
private:
    Key* cudaKeyBuffer;
    ValType* cudaValBuffer;

    uint64_t mMapSize;

    uint64_t Hash(uint64_t key) {
        return khash64_fn(key, your_random_number) & (mMapSize - 1);
    }
public:

    KeyValueMapGPU(const uint32_t mapSize = -1) {
        Init(mapSize);
    }
    void Init(const uint32_t mapSize = -1) {
        uint32_t keyByte2Alloc = 0;
        uint32_t valByte2Alloc = 0;
        if (mapSize == -1) {
            keyByte2Alloc = DEFAULT_MAP_SIZE * sizeof(Key);
            valByte2Alloc = DEFAULT_MAP_SIZE * sizeof(ValType);
        }
        else {
            keyByte2Alloc = mapSize * sizeof(Key);
            valByte2Alloc = mapSize * sizeof(ValType);
        }

        cudaMalloc(&cudaKeyBuffer, keyByte2Alloc);
        cudaMalloc(&cudaValBuffer, valByte2Alloc);
        
        mMapSize = mapSize;
    }

    void Insert(uint64_t key, ValType value) {
        uint32_t indexOfBuffer = Hash(key);

        while (true) {
            uint64_t old = atomicCAS(cudaKeyBuffer[indexOfBuffer].key, INVALID_KEY_VAL, key);
            if (old == INVALID_KEY_VAL || old == key) {
                cudaValBuffer[indexOfBuffer] = value;
                break;
            }
            indexOfBuffer = (indexOfBuffer + 1) & (mMapSize - 1);
        }
        
    }

};

#endif