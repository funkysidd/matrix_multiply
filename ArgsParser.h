#pragma once

#include <string>
#include <array>

class ArgsParser
{
public:
    ArgsParser(int argc, char *argv[])
    {
        for (int i = 1; i < argc; )
        {
            if (std::string(argv[i]) == "--iterations" && (i+1) < argc)
            {

                m_iterations = std::stoi(argv[i+1]);
                i += 2;
            }
            else if (std::string(argv[i]) == "--dims" && (i+3) < argc)
            {

                uint m = std::stoi(argv[i+1]);
                uint n = std::stoi(argv[i+2]);
                uint k = std::stoi(argv[i+3]);

                m_dims = { m, n, k };
                i += 4;
            }
            else
            {
                i += 1;
            }
        }
    }

    int GetIterations() const { return m_iterations; }
    const std::array<uint, 3>& GetDimenions() const { return m_dims; }

private:
    int m_iterations{ 1 };
    std::array<uint, 3> m_dims{ 128, 128, 128 };
};
