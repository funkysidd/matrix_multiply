#include "ArgsParser.h"
#include "MatrixMultiplyRunner.h"

// Block parameters
static constexpr uint BM = 128; // rows
static constexpr uint BN = 128; // cols
static constexpr uint BK = 16;  // common dimension
static constexpr uint TM = 8;   // Thread tile dimension in Y
static constexpr uint TN = 4;   // Thread tile dimension in X

// Block size in A : BM x BK
// Block size in B : BK x BN

// Warp parameters
static constexpr uint THREADS_PER_WARP = 32;   // This is a constant on NVIDIA GPUs
static constexpr uint THREADS_PER_BLOCK = 128; // This could be manipulated
static constexpr uint WM = 64;                 // Warp tile dimensions in Y
static constexpr uint WN = 64;                 // Warp tile dimensions in X

// static constexpr uint WARP_TILES_IN_Y = BM / WM; // Warp tiles in Y
static constexpr uint WARP_TILES_IN_X = BN / WN; // Warp tiles in X

static constexpr uint THREAD_TILES_PER_THREAD_X = 4; // Thread tiles per thread in X
static constexpr uint THREAD_TILES_PER_THREAD_Y =
    (WM * WN) / (THREADS_PER_WARP * TM * TN * THREAD_TILES_PER_THREAD_X); // Thread tiles per thread in Y

// Warp sub-tile: A warp tile is multiple thread tiles, within a warp, that a thread operates upon.

static constexpr uint WARP_SUB_TILE_DIMENSION_Y = THREAD_TILES_PER_THREAD_Y * TM; // Warp sub-tile dimension in Y
static constexpr uint WARP_SUB_TILE_DIMENSION_X = THREAD_TILES_PER_THREAD_X * TN; // Warp sub-tile dimension in X

// static constexpr uint WARP_SUB_TILES_IN_Y = WM / WARP_SUB_TILE_DIMENSION_Y; // Warp sub-tiles in Y
static constexpr uint WARP_SUB_TILES_IN_X = WN / WARP_SUB_TILE_DIMENSION_X; // Warp sub-tiles in X

#define CELLS_PER_COLUMN_LOADED_BY_THREAD 4 // Cells/column loaded by a thread
static constexpr uint THREAD_LOADS_PER_ROW_BLOCK_A = BK / CELLS_PER_COLUMN_LOADED_BY_THREAD;
static constexpr uint THREAD_LOADS_PER_ROW_BLOCK_B = BN / CELLS_PER_COLUMN_LOADED_BY_THREAD;
static constexpr uint32_t THREAD_ROW_STRIDE_A = THREADS_PER_BLOCK / THREAD_LOADS_PER_ROW_BLOCK_A;
static constexpr uint32_t THREAD_ROW_STRIDE_B = THREADS_PER_BLOCK / THREAD_LOADS_PER_ROW_BLOCK_B;

#define OPT_WRITE_MAT_C_VECTORIZED 0 // Optimization enabling vectorized writes to matrix C

__global__ void matMul(float *matA, float *matB, float *matC, const uint M, const uint N, const uint K)
{
    __shared__ float sharedBlockA_T[BK][BM]; // Transposed, i.e., as opposed to BM x BK
    __shared__ float sharedBlockB[BK][BN];

    float threadData[WARP_SUB_TILE_DIMENSION_Y][WARP_SUB_TILE_DIMENSION_X] = { 0.0f };
    float rowA[WARP_SUB_TILE_DIMENSION_Y];
    float colB[WARP_SUB_TILE_DIMENSION_X];

    dim2 blockCellOffsetMatA = { 0, blockIdx.y * BM };
    dim2 blockCellOffsetMatB = { blockIdx.x * BN, 0 };
    dim2 blockCellOffsetMatC = { blockIdx.x * BN, blockIdx.y * BM };

    dim2 baseCellIdxBlockA = { (threadIdx.x % THREAD_LOADS_PER_ROW_BLOCK_A) * CELLS_PER_COLUMN_LOADED_BY_THREAD,
                               threadIdx.x / THREAD_LOADS_PER_ROW_BLOCK_A };
    dim2 baseCellIdxBlockB = { (threadIdx.x % THREAD_LOADS_PER_ROW_BLOCK_B) * CELLS_PER_COLUMN_LOADED_BY_THREAD,
                               threadIdx.x / THREAD_LOADS_PER_ROW_BLOCK_B };

    uint warpId = threadIdx.x / THREADS_PER_WARP;
    uint threadIdWithinWarp = threadIdx.x % THREADS_PER_WARP;
    dim2 warpOffsetInSharedBlock = { (warpId % WARP_TILES_IN_X) * WN, (warpId / WARP_TILES_IN_X) * WM };

    dim2 warpSubtileId = { threadIdWithinWarp % WARP_SUB_TILES_IN_X, threadIdWithinWarp / WARP_SUB_TILES_IN_X };
    dim2 warpSubtileOffSetInSharedBlock = { (warpSubtileId.x * WARP_SUB_TILE_DIMENSION_X) + warpOffsetInSharedBlock.x,
                                            (warpSubtileId.y * WARP_SUB_TILE_DIMENSION_Y) + warpOffsetInSharedBlock.y };

    for (uint blockIt = 0; blockIt < K; blockIt += BK)
    {
        for (uint row = 0; row < BM; row += THREAD_ROW_STRIDE_A)
        {
            // Load `CELLS_PER_COLUMN_LOADED_BY_THREAD` (i.e., float4)  cells at baseCellIdxBlockA
            dim2 cellIdxMatA = { blockCellOffsetMatA.x + baseCellIdxBlockA.x,
                                 blockCellOffsetMatA.y + baseCellIdxBlockA.y + row };
            float4 dataMatA = reinterpret_cast<float4 *>(&matA[cellIdxMatA.y * K + cellIdxMatA.x])[0];
            // Note: Given we store transposed results, we need the next four lines.
            sharedBlockA_T[baseCellIdxBlockA.x + 0][baseCellIdxBlockA.y + row] = dataMatA.x;
            sharedBlockA_T[baseCellIdxBlockA.x + 1][baseCellIdxBlockA.y + row] = dataMatA.y;
            sharedBlockA_T[baseCellIdxBlockA.x + 2][baseCellIdxBlockA.y + row] = dataMatA.z;
            sharedBlockA_T[baseCellIdxBlockA.x + 3][baseCellIdxBlockA.y + row] = dataMatA.w;
        }

        for (uint row = 0; row < BK; row += THREAD_ROW_STRIDE_B)
        {
            // Load `CELLS_PER_COLUMN_LOADED_BY_THREAD` (i.e., float4) cells at baseCellIdxBlockB
            dim2 cellIdxMatB = { blockCellOffsetMatB.x + baseCellIdxBlockB.x,
                                 blockCellOffsetMatB.y + baseCellIdxBlockB.y + row };
            reinterpret_cast<float4 *>(&sharedBlockB[baseCellIdxBlockB.y + row][baseCellIdxBlockB.x])[0] =
                reinterpret_cast<float4 *>(&matB[cellIdxMatB.y * N + cellIdxMatB.x])[0];
        }

        __syncthreads();

        for (uint i = 0; i < BK; ++i)
        {
            for (uint row = 0; row < WARP_SUB_TILE_DIMENSION_Y; ++row)
            {
                rowA[row] = sharedBlockA_T[i][warpSubtileOffSetInSharedBlock.y + row];
            }

            for (uint col = 0; col < WARP_SUB_TILE_DIMENSION_X; ++col)
            {
                colB[col] = sharedBlockB[i][warpSubtileOffSetInSharedBlock.x + col];
            }

            for (uint tItY = 0; tItY < THREAD_TILES_PER_THREAD_Y; ++tItY)
            {
                for (uint tItX = 0; tItX < THREAD_TILES_PER_THREAD_X; ++tItX)
                {
                    // We are now within a thread tile witihn a warp sub-tile
                    dim2 threadTileOffset = { (tItX * TN), (tItY * TM) };

                    for (uint row = 0; row < TM; ++row)
                    {
                        for (uint col = 0; col < TN; ++col)
                        {
                            dim2 dst = { threadTileOffset.x + col, threadTileOffset.y + row };
                            threadData[dst.y][dst.x] += rowA[dst.y] * colB[dst.x];
                        }
                    }
                }
            }
        }

        __syncthreads();

        blockCellOffsetMatA.x += BK;
        blockCellOffsetMatB.y += BK;
    }

    blockCellOffsetMatC.x += warpSubtileOffSetInSharedBlock.x;
    blockCellOffsetMatC.y += warpSubtileOffSetInSharedBlock.y;

#if OPT_WRITE_MAT_C_VECTORIZED
    for (int tItY = 0; tItY < THREAD_TILES_PER_THREAD_Y; ++tItY)
    {
        for (int tItX = 0; tItX < THREAD_TILES_PER_THREAD_X; ++tItX)
        {
            // We are now within a thread tile witihn a warp sub-tile
            dim2 threadTileOffset = { (tItX * TN), (tItY * TM) };

            for (uint row = 0; row < TM; ++row)
            {
                uint tmpY = blockCellOffsetMatC.y + threadTileOffset.y + row;
                for (uint col = 0; col < TN; col += CELLS_PER_COLUMN_LOADED_BY_THREAD)
                {
                    dim2 cellIdxMatC = { blockCellOffsetMatC.x + threadTileOffset.x + col, tmpY };
                    reinterpret_cast<float4 *>(&matC[cellIdxMatC.y * N + cellIdxMatC.x])[0] =
                        reinterpret_cast<float4 *>(&threadData[threadTileOffset.y + row][threadTileOffset.x + col])[0];
                }
            }
        }
    }
#else
    for (uint tItY = 0; tItY < THREAD_TILES_PER_THREAD_Y; ++tItY)
    {
        for (uint tItX = 0; tItX < THREAD_TILES_PER_THREAD_X; ++tItX)
        {
            // We are now within a thread tile witihn a warp sub-tile
            dim2 threadTileOffset = { (tItX * TN), (tItY * TM) };
            for (uint row = 0; row < TM; ++row)
            {
                for (uint col = 0; col < TN; ++col)
                {
                    dim2 cellIdxMatC = { blockCellOffsetMatC.x + threadTileOffset.x + col,
                                         blockCellOffsetMatC.y + threadTileOffset.y + row };
                    matC[cellIdxMatC.y * N + cellIdxMatC.x] =
                        threadData[threadTileOffset.y + row][threadTileOffset.x + col];
                }
            }
        }
    }
#endif
}

class Runner : public MatrixMultiplyRunner
{
  public:
    Runner() : MatrixMultiplyRunner() {}

  protected:
    void InitThreadBlockDimensions() override { m_threadBlockDims = { BN, BM }; }

    void InitNumberOfThreads() override { m_threads = { THREADS_PER_BLOCK }; }

    void LaunchKernel() { matMul<<<m_gridDims, m_threads>>>(m_matADevice, m_matBDevice, m_matCDevice, M(), N(), K()); }
};

int main(int argc, char *argv[])
{
    static_assert(CELLS_PER_COLUMN_LOADED_BY_THREAD == sizeof(float4) / sizeof(float));
    static_assert(THREAD_TILES_PER_THREAD_Y != 0);

    ArgsParser parser;
    if (!parser.Parse(argc, argv))
    {
        parser.Usage(argv[0]);
        return 1;
    }

    int cudaDeviceCount = -1;
    CUDA_CHECK(cudaGetDeviceCount(&cudaDeviceCount));

    if (cudaDeviceCount < 1)
    {
        fprintf(stderr, "No CUDA device found, exiting!\n");
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(0));

    const std::array<uint, 3> &dims = parser.GetDimenions();

    Runner runner;
    runner.Init(dims[0], dims[1], dims[2], 1.0f, 1.0f);
    for (int i = 0; i < parser.GetIterations(); ++i)
    {
        runner.Run(parser.GetExpectedResult());
    }
    runner.Destroy();

    return 0;
}