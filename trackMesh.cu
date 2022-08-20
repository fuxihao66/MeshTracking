#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"


#include "GpuDataStructure.cuh"


#include <stdio.h>


//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

struct Vertex {
    float3 Position;
    float3 Velocity;
};


__global__ void AdvectMeshKernel(float* meshVertices, const float timeStep, const int vertexNumber) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //c[i] = a[i] + b[i];
    if (i < vertexNumber) {
        meshVertices[6 * i + 0] += timeStep * meshVertices[6 * i + 3];
        meshVertices[6 * i + 1] += timeStep * meshVertices[6 * i + 4];
        meshVertices[6 * i + 2] += timeStep * meshVertices[6 * i + 5];
    }
}

__global__ void Build3DHashTable(float* meshVertices, unsigned int* meshIndices, unsigned int triangleNum, TriangleSpatialHashMap* hashMap){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < triangleNum) {
        unsigned int vertexIndex0 = meshIndices[3 * i + 0];
        unsigned int vertexIndex1 = meshIndices[3 * i + 1];
        unsigned int vertexIndex2 = meshIndices[3 * i + 2];

        // 第0个顶点的xyz
        meshVertices[6 * vertexIndex0 + 0];
        meshVertices[6 * vertexIndex0 + 1];
        meshVertices[6 * vertexIndex0 + 2];

        // 第1个顶点的xyz
        meshVertices[6 * vertexIndex1 + 0];
        meshVertices[6 * vertexIndex1 + 1];
        meshVertices[6 * vertexIndex1 + 2];

        // 第2个顶点的xyz
        meshVertices[6 * vertexIndex2 + 0];
        meshVertices[6 * vertexIndex2 + 1];
        meshVertices[6 * vertexIndex2 + 2];

        // 计算出包含这个三角形的所有cell
        // 每个cell的entry里面加入这个三角形
        /*for each cell{
            hashMap
        }*/

        hashMap->InsertTriangle();
    }
}


__global__ void UpdateDeletionList(float* meshVertices, unsigned int* meshIndices, bool* deleteBuffer, unsigned int triangleNum, TriangleSpatialHashMap* hashMap){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < triangleNum) {
        /*unsigned int vertexIndex0 = meshIndices[3 * i + 0];
        unsigned int vertexIndex1 = meshIndices[3 * i + 1];
        unsigned int vertexIndex2 = meshIndices[3 * i + 2];*/

        //// 第0个顶点的xyz
        //meshVertices[6 * vertexIndex0 + 0];
        //meshVertices[6 * vertexIndex0 + 1];
        //meshVertices[6 * vertexIndex0 + 2];

        //// 第1个顶点的xyz
        //meshVertices[6 * vertexIndex1 + 0];
        //meshVertices[6 * vertexIndex1 + 1];
        //meshVertices[6 * vertexIndex1 + 2];

        //// 第2个顶点的xyz
        //meshVertices[6 * vertexIndex2 + 0];
        //meshVertices[6 * vertexIndex2 + 1];
        //meshVertices[6 * vertexIndex2 + 2];

        // 计算出包含这个三角形的所有cell
        // 每个cell的entry里面的所有三角形与这个三角形进行相交测试
        

        hashMap->UpdateIntersectedTriangles(meshVertices, i, deleteBuffer)

        //for (possible grid) {
        //    if grid intersect{
        //        for (triIndex in grid) {
        //            unsigned int triVertexIndex0 = meshIndices[3 * triIndex];
        //            unsigned int triVertexIndex1 = meshIndices[3 * triIndex + 1];
        //            unsigned int triVertexIndex2 = meshIndices[3 * triIndex + 2];

        //            // 根据顶点判断


        //            // 根据相交判断
        //            bool hasIntersection = TriangleIntersection(vertexIndex0, vertexIndex1, vertexIndex2, triVertexIndex0, triVertexIndex1, triVertexIndex2, meshVertices);
        //            if (hasIntersection) {
        //                atomicExch(deleteBuffer[triIndex], 1);
        //                atomicExch(deleteBuffer[i], 1);
        //            }
        //        }
        //    }
        //}
    }
}
__global__ void RemoveTopologicalNoise(float* triangleVertices, uint32_t* meshIndices, uint32_t triangleNum, bool* deletionList) {
    int triangleIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangleIndex < triangleNum) {
        if (deletionList[triangleIndex]) {
            unsigned int vertexIndex0 = meshIndices[3 * triangleIndex + 0];
            unsigned int vertexIndex1 = meshIndices[3 * triangleIndex + 1];
            unsigned int vertexIndex2 = meshIndices[3 * triangleIndex + 2];
            // 第0个顶点的xyz
            float v0x = triangleVertices[6 * vertexIndex0 + 0];
            float v0y = triangleVertices[6 * vertexIndex0 + 1];
            float v0z = triangleVertices[6 * vertexIndex0 + 2];

            // 第1个顶点的xyz
            float v1x = triangleVertices[6 * vertexIndex1 + 0];
            float v1y = triangleVertices[6 * vertexIndex1 + 1];
            float v1z = triangleVertices[6 * vertexIndex1 + 2];

            // 第2个顶点的xyz
            float v2x = triangleVertices[6 * vertexIndex2 + 0];
            float v2y = triangleVertices[6 * vertexIndex2 + 1];
            float v2z = triangleVertices[6 * vertexIndex2 + 2];
        }
    }
}

__global__ void InsideVolumeTest() {

}
/*
__global__ void InsideVolumeKernel(int* c, const int* a, const int* b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
__global__ void DeleteTriangleAndGenBoundaryKernel(int* c, const int* a, const int* b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
__device__ void ManifoldEnforceThreadFunction() {
}
__global__ void ManifoldEnforcementKernel(int* c, const int* a, const int* b){
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
    ManifoldEnforceThreadFunction << < >> > ();   // kernel中调用kernel
}
__global__ void IdentifyHoleKernel(int* c, const int* a, const int* b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
__global__ void PairHoleKernel(int* c, const int* a, const int* b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
__global__ void FillHoleKernel(int* c, const int* a, const int* b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
__global__ void DeleteTriangleAndUpdateBoundaryKernel(int* c, const int* a, const int* b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
__global__ void ImproveMeshKernel(int* c, const int* a, const int* b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
__global__ void RemoveTrianglesKernel(int* c, const int* a, const int* b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
__global__ void RemoveVerticesKernel(int* c, const int* a, const int* b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
*/


#define num_relax_iterations 5

cudaError_t MeshTrack(float* cudaVertexBuffer, unsigned int* cudaIndexBuffer, unsigned int* deleteBuffer, unsigned int vertexNum, unsigned int triangleNum, TriangleSpatialHashMap* hashMap, ) {
    // 1. detect intersection
    {
        dim3 ThreadPerBlock(1024, 1, 1);
        dim3 NumBlock(ceil(triangleNum/float(ThreadPerBlock.x)), 1, 1);
        Build3DHashTable<<<NumBlock, ThreadPerBlock>>>(cudaVertexBuffer, cudaIndexBuffer, deleteBuffer, triangleNum, hashMap);
        UpdateDeletionList<<<NumBlock, ThreadPerBlock>>>();
    }

    // relaxation
    for (int i = 0; i < num_relax_iterations; i++) {
        dim3 ThreadPerBlock(1024, 1, 1);
        dim3 NumBlock(ceil(triangleNum / float(ThreadPerBlock.x)), 1, 1);
        RemoveTopologicalNoise<<<NumBlock, ThreadPerBlock>>>();
    }

    
    // 2. detect inside volume
    InsideVolumeTest<< < >> > ();
    // 3. delete and generate boundary list
    DeleteTriangleAndGenBoundaryKernel << < >> > ();
    // 4. loop
    while (1) {
        ManifoldEnforcementKernel << < >> > ();
        if () {
            break;
        }
    }
    ImproveMeshKernel << < >> > ();
    RemoveTrianglesKernel << < >> > ();
    RemoveVerticesKernel << < >> > ();
    
}


void BuildGeometry(std::vector<Vertex>& vertices, std::vector<unsigned int>& indices,
        unsigned int& indexNum, unsigned int& vertexNum){
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;
    std::string filePath = "data/wooden_sphere.obj";
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filePath.c_str(),
        NULL, true);

    vertexNum = attrib.vertices.size() / 3 * 2;
    vertices.resize(vertexNum);
    for (size_t v = 0; v < attrib.vertices.size() / 3; v++) {
        vertices[v].Position.x = attrib.vertices[3 * v + 0];
        vertices[v].Position.y = attrib.vertices[3 * v + 1] - 0.52f;
        vertices[v].Position.z = attrib.vertices[3 * v + 2];
        vertices[v].Velocity.x = 0.0f;
        vertices[v].Velocity.y = 0.5f;
        vertices[v].Velocity.z = 0.0f;
    }

    for (size_t v = attrib.vertices.size() / 3; v < vertexNum; v++) {
        vertices[v].Position.x = attrib.vertices[3 * (v - attrib.vertices.size() / 3) + 0];
        vertices[v].Position.y = attrib.vertices[3 * (v - attrib.vertices.size() / 3) + 1] + 0.52f;
        vertices[v].Position.z = attrib.vertices[3 * (v - attrib.vertices.size() / 3) + 2];
        vertices[v].Velocity.x = 0.0f;
        vertices[v].Velocity.y = -0.5f;
        vertices[v].Velocity.z = 0.0f;
    }


    indices.resize(shapes[0].mesh.num_face_vertices.size() * 3 * 3);
    for (size_t i = 0; i < shapes.size(); i++) {

        size_t index_offset = 0;

        // For each face
        for (size_t f = 0; f < shapes[i].mesh.num_face_vertices.size(); f++) {
            size_t fnum = shapes[i].mesh.num_face_vertices[f];

            // For each vertex in the face
            for (size_t v = 0; v < fnum; v++) {
                tinyobj::index_t idx = shapes[i].mesh.indices[index_offset + v];
                indices[index_offset + v] = idx.vertex_index;
                indexNum += 1;
            }

            index_offset += fnum;
        }
    }

    for (size_t i = 0; i < indexNum; i++) {
        indices[indexNum + i] = indices[i] + attrib.vertices.size() / 3;
    }
    indexNum *= 2;
}

void 

int main()
{

    /*const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };*/
    const unsigned int MaxFrameCount = 10;
    const float timeStep = 0.01f;

    const float leftBound = -1.5f;  // y
    const float rightBound = 1.5f;
    const float bottomBound = -1.5f; //z
    const float upBound = 1.5f;
    const float farBound = -1.5f; //x
    const float nearBound = 1.5f;

    const float gridSize = 0.01f;   // 需要是最长边的3倍
    const unsigned int entrySize = 100; // 每个cell最多存多少个三角形




    // 生成水滴 
    // TODO: 把cpu和gpu端资源全部封装起来
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    unsigned int vertexNum = 0;
    unsigned int indexNum = 0;

    BuildGeometry(vertices, indices, vertexNum, indexNum);

    cudaError_t cudaStatus = cudaSetDevice(0);
    float* cudaVertexBuffer;
    unsigned int* cudaIndexBuffer;

    {//分配mesh资源    
    // TODO: 分配多两倍的空间  用于增长
        cudaStatus = cudaMalloc((void**)&cudaVertexBuffer, 3 * vertexNum * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
        }
        cudaStatus = cudaMalloc((void**)&cudaIndexBuffer, 3 * indexNum * sizeof(unsigned int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
        }

        cudaStatus = cudaMemcpy(cudaVertexBuffer, (void*)vertices.data(), vertexNum * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
        }

        cudaStatus = cudaMemcpy(cudaIndexBuffer, (void*)indices.data(), indexNum * sizeof(unsigned int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
        }
    }
    
    unsigned int* deleteBuffer;
    
    {// 分配中间变量空间
        cudaStatus = cudaMalloc((void**)&deleteBuffer, indexNum * sizeof(unsigned int));  // 这里的空间是三倍三角形的数量，以防后续三角形增加
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
        }
    }
    IntersectionHashMap* spatialMap = new IntersectionHashMap(farBound, nearBound, leftBound, rightBound, bottomBound, upBound, gridSize, entrySize);


    // 开始处理
    for (int i = 0; i < MaxFrameCount; i++) {
        // advect mesh 
        dim3 NumBlock;
        dim3 ThreadPerBlock;
        ThreadPerBlock.x = 1024;
        ThreadPerBlock.y = 1;
        ThreadPerBlock.z = 1;
        NumBlock.x = ceil(vertexNum / float(ThreadPerBlock.x));
        NumBlock.y = 1;
        NumBlock.z = 1;
        AdvectMeshKernel<<<NumBlock, ThreadPerBlock >>>(cudaVertexBuffer, timeStep, vertexNum);
        
        // begin mesh operate
        cudaStatus = MeshTrack(cudaVertexBuffer, cudaIndexBuffer, deleteBuffer, vertexNum, indexNum / 3, spatialMap);

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d\n", cudaStatus);
            return 1;
        }

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            return 1;
        }
    }


   

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}