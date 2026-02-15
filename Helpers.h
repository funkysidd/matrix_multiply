#pragma once

#include <vector_types.h>

#include <cstdint>

struct dim2
{
    uint32_t x;
    uint32_t y;
};

struct Matrix
{
    float* data;
    dim3 dims;
    dim3 blocks;
};

void computeMappings(
    const Matrix* a,
    const Matrix* b,
    int *aMapping,
    int *bMapping
)
{
    int cIdx = 0;
    for (int row = 0; row < a->blocks.y; ++row)
    {
        for (int col = 0; col < a->blocks.x; ++col)
        {
            int aIdx = row * a->blocks.x + col;

            // The incoming `col` becomes the row that we look at in in matrix `b`.
            int bBase= col * b->blocks.x;
            for (int i = 0; i < b->blocks.x; ++i)
            {
                int bIdx = bBase + i;
                aMapping[cIdx] = aIdx;
                bMapping[cIdx] = bIdx;
                cIdx++;
            }
        }
    }
}