#pragma once

#include <array>
#include <string>

class ArgsParser
{
  public:
    ArgsParser() = default;

    void Usage(const char *progName)
    {
        fprintf(stderr, "Usage: %s --dims <m> <n> <k> --iterations <num> --expected-result <val>\n", progName);
    }

    bool Parse(int argc, char *argv[])
    {
        for (int i = 1; i < argc;)
        {
            if (std::string(argv[i]) == "--dims" && (i + 3) < argc)
            {

                uint m = std::stoi(argv[i + 1]);
                uint n = std::stoi(argv[i + 2]);
                uint k = std::stoi(argv[i + 3]);

                m_dims = { m, n, k };
                i += 4;
            }
            else if (std::string(argv[i]) == "--iterations" && (i + 1) < argc)
            {

                m_iterations = std::stoi(argv[i + 1]);
                i += 2;
            }
            else if (std::string(argv[i]) == "--expected-result" && (i + 1) < argc)
            {

                m_expectedResult = std::stof(argv[i + 1]);
                i += 2;
            }
            else
            {
                return false;
            }
        }

        return true;
    }

    int GetIterations() const { return m_iterations; }
    const std::array<uint, 3> &GetDimenions() const { return m_dims; }
    float GetExpectedResult() const { return m_expectedResult; }

  private:
    int m_iterations{ 1 };
    std::array<uint, 3> m_dims{ 4096, 4086, 4096 };
    float m_expectedResult{ 4096.0f };
};
